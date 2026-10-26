"""Bounded batch-one assistant-action SFT using PEFT LoRA and existing local weights."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time

from scripts.training.checkpoint import restore_checkpoint, save_checkpoint
from scripts.training.hf_backend import RGBBackend, environment, load_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--steps', type=int, choices=range(1, 1001), default=4,
                        help='Total optimizer steps, including restored steps')
    parser.add_argument('--save-every', type=int, choices=range(1, 101), default=4)
    parser.add_argument('--rank', type=int, choices=(2, 4, 8, 16), default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=20261026)
    parser.add_argument('--history-len', type=int, choices=range(0, 33), default=0)
    args = parser.parse_args()
    if not 0 < args.lr <= .01:
        parser.error('lr must be finite and in (0,.01]')
    dataset, manifest, digest = load_dataset(args.dataset)
    if args.history_len > manifest['history_len']:
        parser.error('requested history exceeds audited history')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    backend = RGBBackend(args.model, rank=args.rank, learning_seed=args.seed)
    torch = backend.torch
    parameters = [p for p in backend.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=.01)
    identity = {'model_fingerprint': backend.identity['fingerprint'],
                'dataset_manifest_sha256': digest, 'seed': args.seed, 'rank': args.rank,
                'lora_alpha': args.rank * 2, 'lora_dropout': 0.,
                'target_modules': ['q_proj', 'v_proj'], 'lr': args.lr,
                'weight_decay': .01, 'history_len': args.history_len,
                'dtype': 'bfloat16', 'batch_size': 1, 'optimizer': 'AdamW',
                'packages': environment()['packages']}
    step = restore_checkpoint(args.resume, backend.model, optimizer, identity=identity) if args.resume else 0
    if step >= args.steps:
        parser.error('restored step must be below requested total steps')
    config = {'schema_version': 1, 'identity': identity, 'model': backend.identity,
              'environment': environment(), 'started_utc': datetime.now(timezone.utc).isoformat(),
              'start_step': step, 'total_steps': args.steps,
              'trainable_parameters': sum(p.numel() for p in parameters),
              'source_kinds': manifest['source_kinds'],
              'claims': 'synthetic pipeline smoke' if manifest['source_kinds'] == ['synthetic']
                        else 'supervised adaptation; task improvement requires separate evaluation'}
    (args.output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    training = dataset['train']
    backend.model.train()
    train_log = args.output / 'steps.jsonl'
    while step < args.steps:
        epoch = step // len(training)
        indices = list(range(len(training)))
        random.Random(args.seed + epoch).shuffle(indices)
        row = training[indices[step % len(training)]]
        backend.synchronize()
        start = time.perf_counter()
        batch, count, timing = backend.prepare(row, manifest['data_root'],
                                               history_len=args.history_len, supervised=True)
        optimizer.zero_grad(set_to_none=True)
        loss = backend.loss(batch, count)
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite training loss; experiment stopped')
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError('nonfinite gradient; experiment stopped')
        optimizer.step()
        backend.synchronize()
        step += 1
        record = {'step': step, 'sample_id': row['sample_id'], 'loss': loss.item(),
                  'target_tokens': count, 'gradient_norm': float(gradient_norm),
                  'step_ms': (time.perf_counter() - start) * 1000,
                  'peak_cuda_allocated_mib': torch.cuda.max_memory_allocated() / 1024 ** 2}
        with train_log.open('a', encoding='utf-8') as file:
            file.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            checkpoint = args.output / f'checkpoint-{step:06}'
            save_checkpoint(checkpoint, backend.model, optimizer, step=step, identity=identity)
            backend.model.save_pretrained(checkpoint / 'adapter', safe_serialization=True)
    result = {'completed_optimizer_steps': step, 'elapsed_seconds': time.perf_counter() - started,
              'peak_cuda_allocated_mib': torch.cuda.max_memory_allocated() / 1024 ** 2,
              'checkpoint': f'checkpoint-{step:06}', 'evidence_level': config['claims']}
    (args.output / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
