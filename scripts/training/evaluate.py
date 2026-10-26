"""Frozen local RGB saved-frame evaluation; no desktop input or online success claim."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from scripts.training.hf_backend import RGBBackend, environment, load_dataset
from scripts.training.metrics import action_signature, summarize


def evaluate(backend, rows, root, *, history_len=0, max_new_tokens=64):
    records = []
    backend.model.eval()
    for row in rows:
        total_start = time.perf_counter()
        batch, _, timing = backend.prepare(row, root, history_len=history_len)
        completion, tokens, inference_ms = backend.generate(batch, max_new_tokens=max_new_tokens)
        parse_started = time.perf_counter()
        signature = action_signature(completion)
        action_match = signature is not None and signature == action_signature(row['action'])
        timing.update(inference=inference_ms, parse=(time.perf_counter() - parse_started) * 1000,
                      offline_total=(time.perf_counter() - total_start) * 1000,
                      screenshot=None, transport=None, queue=None, execution=None)
        # Loss is evaluated separately and excluded from the decision-latency measurement.
        supervised, count, _ = backend.prepare(row, root, history_len=history_len, supervised=True)
        with backend.torch.inference_mode():
            loss = backend.loss(supervised, count).item()
        records.append({'sample_id': row['sample_id'], 'game_id': row['game_id'],
                        'source_kind': row['source']['kind'], 'rgb_sha256': row['rgb']['sha256'],
                        'strict_valid': signature is not None, 'action_match': action_match,
                        'completion': completion, 'expected_action': row['action'],
                        'generated_tokens': tokens, 'loss': loss, 'target_tokens': count,
                        'timing_ms': timing})
        print(json.dumps({'sample': row['sample_id'], 'match': action_match,
                          'strict': signature is not None, 'loss': loss,
                          'offline_ms': timing['offline_total']}), flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--split', choices=('train', 'validation', 'test', 'all'), default='test')
    parser.add_argument('--max-samples', type=int, choices=range(1, 129), default=8)
    parser.add_argument('--max-new-tokens', type=int, choices=range(1, 257), default=64)
    parser.add_argument('--history-len', type=int, choices=range(0, 33), default=0)
    args = parser.parse_args()
    dataset, manifest, digest = load_dataset(args.dataset)
    if args.history_len > manifest['history_len']:
        parser.error('requested history exceeds audited history')
    args.output.mkdir(parents=True, exist_ok=False)
    backend = RGBBackend(args.model, adapter=args.adapter)
    metadata = {'schema_version': 1, 'model': backend.identity, 'adapter': backend.adapter_identity,
                'dataset_manifest_sha256': digest,
                'environment': environment(), 'arguments': {k: str(v) if isinstance(v, Path) else v
                                                            for k, v in vars(args).items()},
                'started_utc': datetime.now(timezone.utc).isoformat(),
                'generation': {'do_sample': False, 'stop': '<|action_end|>',
                               'reasoning': False, 'history_len': args.history_len},
                'latency_boundary': 'local disk decode through model generation and action parse; no screenshot/transport/queue/executor present',
                'cache': 'no warmup excluded; sequential cold-first then warm weights; no cross-request KV cache',
                'source_kinds': manifest['source_kinds']}
    (args.output / 'config.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    summaries = {}
    for split in dataset if args.split == 'all' else [args.split]:
        records = evaluate(backend, dataset[split][:args.max_samples], manifest['data_root'],
                           history_len=args.history_len, max_new_tokens=args.max_new_tokens)
        (args.output / f'{split}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
        summaries[split] = summarize(records)
    summaries['peak_cuda_allocated_mib'] = backend.torch.cuda.max_memory_allocated() / 1024 ** 2
    (args.output / 'summary.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
