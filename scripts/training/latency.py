"""Fixed base+adapter PEFT merge benchmark with paired order and quality checks."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import time

from scripts.training.data import require
from scripts.training.hf_backend import RGBBackend, environment, load_dataset
from scripts.training.metrics import action_signature


def percentile(values, percentile):
    values = sorted(values)
    return values[math.ceil(percentile * len(values)) - 1]


def summarize_pairs(rows, *, budget_ms=200.):
    require(rows and math.isfinite(budget_ms) and budget_ms > 0, 'positive budget and rows required')
    pairs = defaultdict(dict)
    for row in rows:
        require(row['arm'] in ('unmerged', 'merged'), 'unknown comparison arm')
        require(row['arm'] not in pairs[row['pair_id']], 'duplicate arm in pair')
        pairs[row['pair_id']][row['arm']] = row
    require(all(set(pair) == {'unmerged', 'merged'} for pair in pairs.values()), 'incomplete arm pair')
    require(all(pair['unmerged']['sample_id'] == pair['merged']['sample_id'] for pair in pairs.values()),
            'paired arms must use the same sample')
    arms = {}
    for arm in ('unmerged', 'merged'):
        selected = [row for row in rows if row['arm'] == arm]
        timing = {}
        for stage in selected[0]['timing_ms']:
            values = [row['timing_ms'][stage] for row in selected if row['timing_ms'][stage] is not None]
            require(all(math.isfinite(v) and v >= 0 for v in values), 'invalid timing')
            timing[stage] = {'p50': statistics.median(values), 'p95': percentile(values, .95)} if values else None
        arms[arm] = {'requests': len(selected), 'errors': sum(bool(r['error']) for r in selected),
                     'strict_valid_rate': sum(r['strict_valid'] for r in selected) / len(selected),
                     'action_match_rate': sum(r['action_match'] for r in selected) / len(selected),
                     'budget_overrun_rate': sum(r['timing_ms']['offline_total'] > budget_ms for r in selected) / len(selected),
                     'timing_ms': timing}
    quality = all(not r['error'] and r['strict_valid'] and r['action_match'] for r in rows)
    ratios = [p['unmerged']['timing_ms']['offline_total'] / p['merged']['timing_ms']['offline_total']
              for p in pairs.values() if p['merged']['timing_ms']['offline_total'] > 0]
    return {'pairs': len(pairs), 'arms': arms, 'quality_preserved_on_static_probes': quality,
            'paired_median_speedup': statistics.median(ratios) if quality and ratios else None,
            'completion_agreement_rate': sum(not p['unmerged']['error'] and not p['merged']['error']
                                             and p['unmerged']['strict_valid'] and p['merged']['strict_valid']
                                             and p['unmerged']['completion'] == p['merged']['completion']
                                             for p in pairs.values()) / len(pairs),
            'processing_budget_ms': budget_ms,
            'p95_method': 'nearest-rank small-sample empirical quantile',
            'evidence_level': 'static saved-frame execution comparison; no online task/stale-rate claim'}


def measure(backend, row, root, arm, pair_id, *, max_new_tokens):
    start = time.perf_counter()
    timing = {'decode': None, 'encode': None, 'host_to_device': None, 'inference': None,
              'parse': None, 'screenshot': None, 'transport': None, 'queue': None, 'execution': None}
    completion, error, token_count, signature = '', None, 0, None
    try:
        batch, _, stages = backend.prepare(row, root)
        timing.update(stages)
        completion, token_count, inference_ms = backend.generate(batch, max_new_tokens=max_new_tokens)
        timing['inference'] = inference_ms
        parsed_at = time.perf_counter()
        signature = action_signature(completion)
        timing['parse'] = (time.perf_counter() - parsed_at) * 1000
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    timing['offline_total'] = (time.perf_counter() - start) * 1000
    return {'pair_id': pair_id, 'arm': arm, 'sample_id': row['sample_id'],
            'completion': completion, 'generated_tokens': token_count, 'error': error,
            'strict_valid': signature is not None,
            'action_match': signature is not None and signature == action_signature(row['action']),
            'timing_ms': timing}


def compare_logits(first, second, row, root):
    torch = first.torch
    outputs, losses = [], []
    with torch.inference_mode():
        for backend in (first, second):
            batch, count, _ = backend.prepare(row, root, supervised=True)
            labels = batch.pop('labels')[:, -count:]
            out = backend.model(**batch, use_cache=False, logits_to_keep=count + 1).logits[:, :-1].float()
            outputs.append(out.cpu())
            losses.append(float(torch.nn.functional.cross_entropy(out.reshape(-1, out.shape[-1]), labels.reshape(-1))))
    return {'sample_id': row['sample_id'], 'max_abs_logit_difference': float((outputs[0] - outputs[1]).abs().max()),
            'mean_abs_logit_difference': float((outputs[0] - outputs[1]).abs().mean()),
            'unmerged_target_loss': losses[0], 'merged_target_loss': losses[1],
            'note': 'safe bf16 merge can round effective weights; bitwise equivalence is not assumed'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, choices=range(1, 9), default=2)
    parser.add_argument('--repeats', type=int, choices=range(1, 7), default=3)
    parser.add_argument('--max-new-tokens', type=int, choices=range(1, 129), default=64)
    args = parser.parse_args()
    dataset, manifest, digest = load_dataset(args.dataset)
    rows = dataset['test'][:args.samples]
    args.output.mkdir(parents=True, exist_ok=False)
    first = RGBBackend(args.model, adapter=args.adapter)
    second = RGBBackend(args.model, adapter=args.adapter)
    require(first.identity == second.identity and first.adapter_identity == second.adapter_identity,
            'benchmark policy identities must match')
    merge_start = time.perf_counter()
    second.model = second.model.merge_and_unload(safe_merge=True)
    second.model.eval()
    second.synchronize()
    merge_ms = (time.perf_counter() - merge_start) * 1000
    config = {'schema_version': 1, 'started_utc': datetime.now(timezone.utc).isoformat(),
              'model': first.identity, 'adapter': first.adapter_identity, 'environment': environment(),
              'dataset_manifest_sha256': digest, 'sample_ids': [r['sample_id'] for r in rows],
              'repeats': args.repeats, 'max_new_tokens': args.max_new_tokens,
              'policy': 'greedy empty-think/history0; exact base+adapter identity, bf16 merge rounding measured',
              'order': 'alternating per pair; both models resident throughout; warmups saved separately',
              'merge_ms': merge_ms, 'timing_boundary': 'saved file decode through action parse; no screenshot/network/queue/executor'}
    (args.output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    warmups = [measure(backend, rows[0], manifest['data_root'], arm, 'warmup',
                       max_new_tokens=args.max_new_tokens)
               for arm, backend in (('unmerged', first), ('merged', second))]
    (args.output / 'warmups.json').write_text(json.dumps(warmups, indent=2), encoding='utf-8')
    require(all(not r['error'] for r in warmups), 'warmup failed; inspect saved warmup errors')
    records = []
    for repeat in range(args.repeats):
        for index, row in enumerate(rows):
            order = [('unmerged', first), ('merged', second)]
            if (repeat + index) % 2:
                order.reverse()
            for arm, backend in order:
                result = measure(backend, row, manifest['data_root'], arm, f'{repeat}:{index}',
                                 max_new_tokens=args.max_new_tokens)
                records.append(result)
                with (args.output / 'requests.jsonl').open('a', encoding='utf-8') as file:
                    file.write(json.dumps(result) + '\n')
                print(json.dumps({k: result[k] for k in ('pair_id', 'arm', 'error', 'action_match', 'timing_ms')}), flush=True)
    numerical = [compare_logits(first, second, row, manifest['data_root']) for row in rows]
    summary = {**summarize_pairs(records), 'numerical_checks': numerical,
               'peak_cuda_allocated_mib': first.torch.cuda.max_memory_allocated() / 1024 ** 2}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
