"""Matched first-key SFT or actual one-step contextual-bandit LoRA learning.

This is not sequential RL: each visible-horizon1 environment resets after one
sampled action. The bandit objective never receives an expert class or state.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

from controller.control_core import to_action_text
from scripts.training.action_head import LABELS, forward_action_logits, template_action
from scripts.training.checkpoint import restore_checkpoint, save_checkpoint
from scripts.training.data import checked_rgb, read_jsonl, require, sha256_file
from scripts.training.feedback import REWARD_VERSION, audit_feedback_episodes
from scripts.training.feedback_learning import (bandit_loss, class_from_action, index_for_step,
                                                student_row, supervised_loss)
from scripts.training.headless import HeadlessGame
from scripts.training.hf_backend import RGBBackend, environment, load_dataset
from scripts.training.rollout import run_episode


def trainable_fingerprint(model):
    import torch
    digest = hashlib.sha256()
    for name, value in sorted(model.named_parameters()):
        if value.requires_grad:
            tensor = value.detach().cpu().contiguous()
            digest.update(f'{name}:{tensor.dtype}:{tuple(tensor.shape)}:'.encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def checked_resets(rows, episodes):
    """Audit reset provenance outside the student; never return teacher labels/state."""
    by_id = {e['episode_id']: e for e in episodes}
    result = {}
    for row in rows:
        require(row['game_id'] == 'headless-reach-v1' and row['frame_id'] == 0,
                'training requires own reach initial observations')
        source = by_id[row['episode_id']]
        match = re.fullmatch('headless-reach-v1-seed-([0-9]+)', source['episode_id'])
        require(match is not None and len(source['frames']) == 1, 'invalid training reset identity')
        require(source['frames'][0]['terminal_reason'] == 'goal'
                and abs(source['frames'][0]['reward'] - .99) < 1e-9, 'training expert transition must actually succeed')
        seed = int(match[1])
        state = source['frames'][0]['feedback']['state_before']
        require(state == HeadlessGame(seed=seed, max_steps=1).snapshot(),
                'source RGB must represent the supported visible horizon1 reset')
        result[row['sample_id']] = seed
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('sft', 'bandit'), required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--steps', type=int, choices=range(1, 1001), default=4)
    parser.add_argument('--save-every', type=int, choices=range(1, 101), default=4)
    parser.add_argument('--rank', type=int, choices=(2, 4, 8, 16), default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=20261028)
    parser.add_argument('--temperature', type=float, default=2.)
    parser.add_argument('--baseline', type=float, default=.19)
    args = parser.parse_args()
    require(0 < args.lr <= .01 and .1 <= args.temperature <= 10 and -1.01 <= args.baseline <= .99,
            'invalid learning rate/temperature/constant baseline')
    dataset, manifest, digest = load_dataset(args.dataset)
    require(manifest['source_kinds'] == ['recorded'], 'this matched reward study requires recorded own-environment corpus')
    episodes = list(read_jsonl(manifest['source_path']))
    audit_feedback_episodes(episodes, manifest['data_root'])
    rows = dataset['train']
    resets = checked_resets(rows, episodes)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    backend = RGBBackend(args.model, rank=args.rank, learning_seed=args.seed, deterministic=True)
    torch = backend.torch
    parameters = [p for p in backend.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=.01)
    code = {p.name: sha256_file(p) for p in (
        Path(__file__), Path(__file__).with_name('feedback_learning.py'),
        Path(__file__).with_name('action_head.py'), Path(__file__).with_name('rollout.py'),
        Path(__file__).with_name('headless.py'), Path(__file__).with_name('checkpoint.py'),
        Path(__file__).with_name('hf_backend.py'), Path(__file__).with_name('feedback.py'),
        Path(__file__).with_name('formatting.py'), Path(__file__).with_name('data.py'),
        Path(__file__).with_name('metrics.py'), Path(__file__).resolve().parents[2] / 'controller/control_core.py')}
    identity = {'model_fingerprint': backend.identity['fingerprint'], 'dataset_manifest_sha256': digest,
                'arm': args.arm, 'seed': args.seed, 'rank': args.rank, 'lora_alpha': args.rank * 2,
                'target_modules': ['q_proj', 'v_proj'], 'lora_dropout': 0., 'lr': args.lr,
                'optimizer': 'AdamW', 'weight_decay': .01, 'grad_clip': 1., 'batch_size': 1, 'dtype': 'bfloat16',
                'temperature': args.temperature, 'constant_baseline': args.baseline if args.arm == 'bandit' else None,
                'entropy_regularization': 0., 'history_len': 0, 'environment_horizon': 1,
                'reward_contract': REWARD_VERSION, 'classes': list(LABELS),
                'initialization_fingerprint': trainable_fingerprint(backend.model),
                'determinism': backend.determinism, 'packages': environment()['packages'], 'code': code}
    step = restore_checkpoint(args.resume, backend.model, optimizer, identity=identity) if args.resume else 0
    require(step < args.steps, 'restored step must be below total requested steps')
    start_step = step
    config = {'schema_version': 1, 'identity': identity, 'model': backend.identity,
              'environment': environment(), 'started_utc': datetime.now(timezone.utc).isoformat(),
              'start_step': step, 'total_steps': args.steps,
              'trainable_parameters': sum(p.numel() for p in parameters),
              'source_sha256': manifest['source_sha256'], 'observations_per_epoch': len(rows),
              'student_inputs': ['RGB', 'fixed visible rules'], 'teacher_in_student_inputs': False,
              'teacher_in_objective': args.arm == 'sft',
              'reward_source': 'actual200ms environment step' if args.arm == 'bandit' else None,
              'claim': 'one-step contextual-bandit reward learning vs first-key supervised learning; not sequential RL',
              'budget_note': 'both share an expert-curated frozen reset corpus; SFT uses labels offline, bandit adds actual interactions; equal updates are not equal total cost'}
    (args.output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    backend.model.train()
    while step < args.steps:
        row = rows[index_for_step(len(rows), step, args.seed)]
        backend.synchronize()
        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        actual_reward, selected, teacher_class = None, None, None
        if args.arm == 'sft':
            batch, _, _ = backend.prepare(student_row(row), manifest['data_root'])
            raw_logits, candidates = forward_action_logits(backend, batch)
            logits = raw_logits[candidates].float()
            teacher_class = class_from_action(row['action'])
            loss = supervised_loss(logits, teacher_class, temperature=args.temperature)
        else:
            captured = {}

            def policy(live_row):
                # A bandit gets the same initial RGB as the paired SFT observation, not
                # its teacher action or source engine annotations. This is an audit check.
                pixels = checked_rgb(live_row['rgb'], args.output.resolve())
                require(pixels['pixel_sha256'] == row['rgb']['pixel_sha256']
                        and live_row['system_prompt'] == row['system_prompt'], 'paired student inputs differ')
                batch, _, timing = backend.prepare(student_row(live_row), args.output)
                forward_started = time.perf_counter()
                raw_logits, candidates = forward_action_logits(backend, batch)
                values = raw_logits[candidates].float()
                distribution = torch.distributions.Categorical(logits=values / args.temperature)
                sample = int(distribution.sample())
                captured.update(logits=values, selected=sample)
                text = to_action_text(template_action(LABELS[sample]))
                backend.synchronize()
                return {'completion': text, 'generated_tokens': 0,
                        'timing_ms': {**timing, 'inference': (time.perf_counter() - forward_started) * 1000},
                        'decoder_record': {'text_source': 'sampled five-class training template',
                                           'temperature': args.temperature, 'sampled_class': LABELS[sample],
                                           'raw_class_logits': values.detach().cpu().tolist(),
                                           'sampling_probabilities': distribution.probs.detach().cpu().tolist()}}

            episode = run_episode(policy, args.output, variant='reach', seed=resets[row['sample_id']],
                                  max_steps=1, history_len=0, episode_tag=f'update-{step + 1:06}')
            require(len(episode['frames']) == 1, 'bandit interaction must have exactly one actual action')
            frame = episode['frames'][0]
            actual_reward = frame['reward']
            selected = captured['selected']
            require(frame['actual_action']['text'] == to_action_text(template_action(LABELS[selected])),
                    'sampled policy action differs from actual execution')
            with (args.output / 'feedback-episodes.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(episode) + '\n')
            logits = captured['logits']
            loss = bandit_loss(logits, selected, actual_reward, temperature=args.temperature, baseline=args.baseline)
        require(bool(torch.isfinite(loss)), 'nonfinite loss; experiment stopped')
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
        require(bool(torch.isfinite(gradient_norm)), 'nonfinite gradient; experiment stopped')
        optimizer.step()
        backend.synchronize()
        step += 1
        record = {'step': step, 'sample_id': row['sample_id'], 'rgb_pixel_sha256': row['rgb']['pixel_sha256'],
                  'loss': float(loss.detach()), 'gradient_norm': float(gradient_norm),
                  'teacher_class': teacher_class, 'sampled_class': selected, 'actual_reward': actual_reward,
                  'raw_class_logits': logits.detach().cpu().tolist(),
                  'step_ms': (time.perf_counter() - step_started) * 1000,
                  'peak_cuda_allocated_mib': torch.cuda.max_memory_allocated() / 1024 ** 2}
        with (args.output / 'steps.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            checkpoint = args.output / f'checkpoint-{step:06}'
            save_checkpoint(checkpoint, backend.model, optimizer, step=step, identity=identity)
            backend.model.save_pretrained(checkpoint / 'adapter', safe_serialization=True)
    feedback = (audit_feedback_episodes(read_jsonl(args.output / 'feedback-episodes.jsonl'), args.output)
                if args.arm == 'bandit' else None)
    result = {'completed_optimizer_steps': step, 'start_step': start_step, 'new_updates': step - start_step,
              'new_online_interactions': step - start_step if args.arm == 'bandit' else 0,
              'new_expert_label_uses': step - start_step if args.arm == 'sft' else 0,
              'elapsed_seconds': time.perf_counter() - started, 'feedback_audit': feedback,
              'peak_cuda_allocated_mib': torch.cuda.max_memory_allocated() / 1024 ** 2,
              'checkpoint': f'checkpoint-{step:06}', 'claim': config['claim']}
    (args.output / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
