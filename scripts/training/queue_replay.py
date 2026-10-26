"""CPU virtual queue replay over saved paired decoder service times.

Arrivals and action occupancy are assumptions, not observed online latency. No
inference, desktop input, clock waiting, task reward or quality estimate occurs.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from scripts.training.data import require, sha256_file
from scripts.training.decoding_benchmark import ARMS, summarize_pairs
from scripts.training.latency import percentile


def simulate_queue(service_ns, *, interval_ns, action_ns, policy, start_budget_ns, end_budget_ns):
    """One worker serializes service and action; latest drops only pending frames."""
    require(isinstance(service_ns, list) and 0 < len(service_ns) <= 100_000
            and all(type(v) is int and v > 0 for v in service_ns), 'positive integer service times required')
    require(all(type(v) is int and v > 0 for v in (interval_ns, start_budget_ns, end_budget_ns))
            and type(action_ns) is int and action_ns >= 0, 'integer timing parameters required')
    require(policy in ('fifo', 'latest'), 'unknown scheduling policy')
    ready, next_index, events, dropped = 0, 0, [], []
    while next_index < len(service_ns):
        index = next_index if policy == 'fifo' else min(len(service_ns) - 1,
                                                       max(next_index, ready // interval_ns))
        dropped.extend(range(next_index, index))
        captured = index * interval_ns
        started = max(ready, captured)
        action_start = started + service_ns[index]
        action_end = action_start + action_ns
        events.append({'frame_index': index, 'captured_ns': captured, 'worker_start_ns': started,
                       'service_ns': service_ns[index], 'queue_ns': started - captured,
                       'action_start_ns': action_start, 'action_end_ns': action_end,
                       'capture_to_start_ns': action_start - captured,
                       'capture_to_end_ns': action_end - captured})
        ready, next_index = action_end, index + 1
    n = len(service_ns)
    timely_start = sum(e['capture_to_start_ns'] <= start_budget_ns for e in events)
    timely_end = sum(e['capture_to_end_ns'] <= end_budget_ns for e in events)
    timing = {}
    for field in ('queue_ns', 'capture_to_start_ns', 'capture_to_end_ns'):
        values = [e[field] / 1_000_000 for e in events]
        timing[field.removesuffix('_ns') + '_ms'] = {'p50': statistics.median(values),
                                                     'p95': percentile(values, .95)}
    return {'policy': policy, 'arrivals': n, 'processed': len(events), 'dropped': dropped,
            'events': events, 'drain_end_ns': ready, 'selected_events_timing': timing,
            'on_time_start_arrivals': timely_start, 'on_time_end_arrivals': timely_end,
            'on_time_start_fraction_all_arrivals': timely_start / n,
            'on_time_end_fraction_all_arrivals': timely_end / n}


def replay_benchmark(benchmark, *, interval_ns=200_000_000, action_ns=200_000_000,
                     start_budget_ns=200_000_000, end_budget_ns=400_000_000, source_root=None):
    """Validate all pairs and their original schedule before any simulation."""
    root = Path(benchmark)
    snapshots = {name: (root / name).read_bytes()
                 for name in ('config.json', 'requests.jsonl', 'summary.json')}
    config = json.loads(snapshots['config.json'])
    rows = [json.loads(line) for line in snapshots['requests.jsonl'].decode('utf-8').splitlines()]
    measured = json.loads(snapshots['summary.json'])
    summary = summarize_pairs(rows)
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    require(canonical({k: v for k, v in measured.items() if k != 'peak_cuda_allocated_mib'}) == canonical(summary),
            'saved paired summary differs from all requests')
    require(summary['all_paired_actions_agree'] and all(not row['error'] for row in rows),
            'failed or disagreeing measured requests cannot be silently omitted')
    require(type(config.get('repeats')) is int and config['repeats'] > 0
            and isinstance(config.get('samples'), list) and config['samples'], 'complete schedule required')
    for key in ('model', 'adapter', 'environment', 'source_code_sha256'):
        require(config.get(key), 'fixed model/adapter/environment/source identity required')
    for name, expected in config['source_code_sha256'].items():
        require(isinstance(name, str) and Path(name).name == name, 'unexpected source file path')
        path = (Path(source_root) if source_root is not None else Path(__file__).parent) / name
        require(path.is_file() and sha256_file(path) == expected, 'measured source code identity changed')
    require(len({sample['id'] for sample in config['samples']}) == len(config['samples']),
            'duplicate sample in frozen source schedule')
    selected = {arm: [] for arm in ARMS}
    cursor = 0
    for repeat in range(config['repeats']):
        for index, sample in enumerate(config['samples']):
            order = ARMS if (repeat + index) % 2 == 0 else tuple(reversed(ARMS))
            for arm in order:
                require(cursor < len(rows), 'missing measured request')
                row = rows[cursor]
                require(row['pair_id'] == f'{repeat}:{index}' and row['arm'] == arm
                        and row['sample_id'] == sample['id'] and row['rgb_sha256'] == sample['rgb_sha256'],
                        'measured request order/input differs from frozen schedule')
                duration = row['timing_ms']['offline_total']
                require(type(duration) in (int, float) and math.isfinite(duration) and duration > 0,
                        'invalid measured service time')
                selected[arm].append(max(1, round(duration * 1_000_000)))
                cursor += 1
    require(cursor == len(rows), 'extra measured requests outside schedule')
    assumptions = {'interval_ns': interval_ns, 'action_ns': action_ns,
                   'start_budget_ns': start_budget_ns, 'end_budget_ns': end_budget_ns,
                   'worker': 'one non-preemptive serial service+action worker',
                   'service': 'recorded offline_total for each selected frame, exogenous to virtual cadence',
                   'unmeasured': ['screenshot', 'network', 'additional online logging', 'continuous world'],
                   'clock': 'synthetic integer nanoseconds; equality at release is eligible',
                   'deadline_note': 'start200ms/end400ms are equivalent when action is fixed200ms',
                   'p95': 'nearest-rank over selected processed events, not all arrivals'}
    cells = {arm: {policy: simulate_queue(times, interval_ns=interval_ns, action_ns=action_ns,
                                          policy=policy, start_budget_ns=start_budget_ns,
                                          end_budget_ns=end_budget_ns)
                    for policy in ('fifo', 'latest')} for arm, times in selected.items()}
    return {'schema_version': 1, 'source_sha256': {name: hashlib.sha256(data).hexdigest()
                for name, data in snapshots.items()},
            'measured_source_sha256': config['source_code_sha256'],
            'replay_source_sha256': {name: sha256_file(Path(__file__).parent / name)
                                    for name in ('queue_replay.py', 'decoding_benchmark.py',
                                                 'latency.py', 'metrics.py', 'data.py')},
            'model': config['model'], 'adapter': config['adapter'], 'environment': config['environment'],
            'assumptions': assumptions, 'cells': cells,
            'claim': 'conditional CPU queue simulation only; no live speedup, online freshness or task-quality result'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='new JSON report; never overwrite')
    parser.add_argument('--source-root', type=Path,
                        help='read-only source snapshot matching measured hashes; never executed')
    args = parser.parse_args()
    require(not args.output.exists(), 'output report already exists')
    report = replay_benchmark(args.benchmark, source_root=args.source_root)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({arm: {policy: {k: v for k, v in result.items() if k not in ('events', 'dropped')}
                           for policy, result in cells.items()} for arm, cells in report['cells'].items()}, indent=2))


if __name__ == '__main__':
    main()
