"""Actual pretrained 0.8B LoRA on RGB plus causal audio-derived visual tokens."""
import argparse
from datetime import datetime, timezone
import json
import importlib.metadata
from pathlib import Path
import time

from scripts.training.action_head import forward_action_logits
from scripts.training.audio_cue_data import load_audio_dataset
from scripts.training.audio_vision import prepare_audio
from scripts.training.bandit_train import trainable_fingerprint
from scripts.training.checkpoint import restore_checkpoint, save_checkpoint
from scripts.training.data import require, sha256_file
from scripts.training.feedback_learning import class_from_action, index_for_step, supervised_loss
from scripts.training.hf_backend import RGBBackend, environment


def mask_for_step(step):
    require(type(step) is int and step >= 0, 'invalid global step for mask dropout')
    return step % 8 == 7


def training_source_paths():
    names = ('audio_vision_train.py', 'audio_vision.py', 'audio_cue.py', 'audio_cue_data.py',
             'audio.py', 'action_head.py', 'bandit_train.py', 'checkpoint.py', 'data.py',
             'feedback.py', 'feedback_learning.py', 'formatting.py', 'headless.py',
             'hf_backend.py', 'metrics.py', 'rollout.py')
    return [Path(__file__).with_name(name) for name in names] + [Path(__file__).resolve().parents[2]/'controller/control_core.py']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--steps', type=int, choices=range(1, 1001), default=2)
    parser.add_argument('--save-every', type=int, choices=range(1, 101), default=2)
    parser.add_argument('--seed', type=int, default=20261029)
    args = parser.parse_args()
    dataset, manifest, digests, audio_index = load_audio_dataset(args.dataset)
    rows = dataset['train']
    require(manifest['source_kinds'] == ['recorded']
            and all(r['game_id'] == 'audio-cue-reach-v1' and r['frame_id'] == 0 and not r['history'] for r in rows),
            'audio training requires recorded own reach initial sensors with no history')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    backend = RGBBackend(args.model, rank=8, learning_seed=args.seed, deterministic=True)
    torch = backend.torch
    parameters = [p for p in backend.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=.01)
    identity = {'model': backend.identity['fingerprint'], 'dataset': digests, 'seed': args.seed,
                'initialization_fingerprint': trainable_fingerprint(backend.model),
                'rank': 8, 'alpha': 16, 'dropout': 0., 'dtype': 'bfloat16',
                'optimizer': 'AdamW', 'lr': 1e-4, 'weight_decay': .01, 'clip': 1., 'batch_size': 1,
                'temperature': 2., 'mask_dropout': 'global_step_zero_based_mod8_equals7',
                'inputs': 'currentRGB+currentSTFTimage+presentmask; no history/teacher/engine state',
                'spectrogram_packages': {name: importlib.metadata.version(name) for name in ('numpy', 'Pillow')},
                'determinism': backend.determinism, 'packages': environment()['packages'],
                'code': {p.name: sha256_file(p) for p in training_source_paths()}}
    step = restore_checkpoint(args.resume, backend.model, optimizer, identity=identity) if args.resume else 0
    require(step < args.steps, 'restored step must be below requested total steps')
    start_step = step
    config = {'schema_version': 1, 'identity': identity, 'model': backend.identity,
              'environment': environment(), 'source_sha256': manifest['source_sha256'],
              'started_utc': datetime.now(timezone.utc).isoformat(), 'start_step': step, 'total_steps': args.steps,
              'trainable_parameters': sum(p.numel() for p in parameters), 'observations_per_epoch': len(rows),
              'claim': 'actual0.8Bvisionencoder+fixedaudioSTFTimagefusion on procedural task; not native audio or real-game improvement'}
    (args.output/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    backend.model.train()
    while step < args.steps:
        row = rows[index_for_step(len(rows), step, args.seed)]
        sensor_row = {'system_prompt': row['system_prompt'], 'rgb': dict(row['rgb']), **audio_index[row['sample_id']]}
        backend.synchronize()
        began = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        mask = mask_for_step(step)
        batch, evidence = prepare_audio(backend, sensor_row, manifest['data_root'], args.output, force_missing=mask)
        raw, candidates = forward_action_logits(backend, batch)
        logits = raw[candidates].float()
        target = class_from_action(row['action'])
        loss = supervised_loss(logits, target, temperature=2.)
        require(bool(torch.isfinite(loss)), 'nonfinite audio training loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
        require(bool(torch.isfinite(norm)), 'nonfinite audio training gradient')
        optimizer.step()
        backend.synchronize()
        step += 1
        record = {'step': step, 'sample_id': row['sample_id'], 'rgb_pixel_sha256': row['rgb']['pixel_sha256'],
                  'target_class': target, 'loss': float(loss.detach()), 'gradient_norm': float(norm),
                  'raw_class_logits': logits.detach().cpu().tolist(), 'sensor_evidence': evidence,
                  'mask_dropout': mask, 'step_ms': (time.perf_counter()-began)*1000,
                  'peak_cuda_allocated_mib': torch.cuda.max_memory_allocated()/1024**2}
        with (args.output/'steps.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, allow_nan=False)+'\n')
        print(json.dumps(record), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            checkpoint = args.output/f'checkpoint-{step:06}'
            save_checkpoint(checkpoint, backend.model, optimizer, step=step, identity=identity)
            backend.model.save_pretrained(checkpoint/'adapter', safe_serialization=True)
    result = {'completed_optimizer_steps': step, 'start_step': start_step, 'new_updates': step-start_step,
              'elapsed_seconds': time.perf_counter()-started, 'checkpoint': f'checkpoint-{step:06}',
              'peak_cuda_allocated_mib': torch.cuda.max_memory_allocated()/1024**2,
              'claim': config['claim']}
    (args.output/'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
