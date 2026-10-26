"""Paired full-generation vs restricted first-key decoding, with one fixed model."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import time

from scripts.training.action_head import FirstKeyBackend, LABELS, template_action
from scripts.training.data import read_jsonl, require, sha256_file
from scripts.training.feedback import audit_feedback_episodes
from scripts.training.hf_backend import RGBBackend, environment
from scripts.training.latency import percentile
from scripts.training.metrics import action_signature
from scripts.training.rollout import filter_completion
from controller.control_core import to_action_text


ARMS = ('full', 'first-key')


def summarize_pairs(rows, *, budget_ms=200.):
    require(rows and math.isfinite(budget_ms) and budget_ms > 0, 'positive budget and rows required')
    pairs = defaultdict(dict)
    for row in rows:
        require(row['arm'] in ARMS and row['arm'] not in pairs[row['pair_id']], 'unknown or duplicate arm')
        require(row['timing_ms']['offline_total'] > 0 and all(
            value is None or math.isfinite(value) and value >= 0 for value in row['timing_ms'].values()),
            'invalid decision timing')
        pairs[row['pair_id']][row['arm']] = row
    require(all(set(pair) == set(ARMS) for pair in pairs.values()), 'incomplete decoder pair')
    require(all(all(pair['full'][key] == pair['first-key'][key] for key in ('sample_id', 'rgb_sha256'))
                for pair in pairs.values()), 'decoder pair input mismatch')
    result = {'pairs': len(pairs), 'arms': {}}
    for arm in ARMS:
        selected = [r for r in rows if r['arm'] == arm]
        timing = {}
        for key in sorted({key for r in selected for key in r['timing_ms']}):
            values = [r['timing_ms'][key] for r in selected if r['timing_ms'].get(key) is not None]
            timing[key] = {'p50': statistics.median(values), 'p95': percentile(values, .95)} if values else None
        result['arms'][arm] = {'requests': len(selected), 'errors': sum(bool(r['error']) for r in selected),
                               'strict_valid_rate': sum(r['strict_valid'] for r in selected) / len(selected),
                               'processing_budget_overrun_rate': sum(r['timing_ms']['offline_total'] > budget_ms
                                                                     for r in selected) / len(selected),
                               'timing_ms': timing}
    valid = all(not r['error'] and r['strict_valid'] for r in rows)
    pairs_valid = [p for p in pairs.values() if all(not r['error'] and r['strict_valid'] for r in p.values())]
    executed_agreement = sum(action_signature(p['full']['actual_action']) ==
                             action_signature(p['first-key']['actual_action']) for p in pairs_valid) / len(pairs)
    proposal_agreement = sum(action_signature(p['full']['completion']) ==
                             action_signature(p['first-key']['completion']) for p in pairs_valid) / len(pairs)
    restricted = {action_signature(to_action_text(template_action(label))) for label in LABELS}
    result.update(executed_action_agreement_rate=executed_agreement,
                  proposal_action_agreement_rate=proposal_agreement,
                  all_paired_actions_agree=valid and executed_agreement == 1.,
                  full_proposals_in_restricted_space_rate=sum(
                      not p['full']['error'] and action_signature(p['full']['completion']) in restricted
                      for p in pairs.values()) / len(pairs),
                  paired_median_speedup_observed=statistics.median(
                      p['full']['timing_ms']['offline_total'] / p['first-key']['timing_ms']['offline_total']
                      for p in pairs.values()) if valid else None,
                  processing_budget_ms=budget_ms, p95_method='nearest-rank empirical small-sample quantile',
                  claim='decoder latency/agreement only; altered action space needs independent task return evaluation',
                  format_note='first-key strict validity is safe-template construction, not learned format ability')
    return result


def measure(backend, row, root, arm, pair_id, *, max_new_tokens):
    started = time.perf_counter()
    completion, tokens, error, actual, details = '', 0, None, None, None
    timing = {'decode': None, 'encode': None, 'host_to_device': None, 'inference': None,
              'parse_filter': None, 'screenshot': None, 'transport': None, 'queue': None, 'execution': None}
    strict = False
    try:
        batch, _, stages = backend.prepare(row, root)
        timing.update(stages)
        generate = RGBBackend.generate if arm == 'full' else FirstKeyBackend.generate
        completion, tokens, timing['inference'] = generate(backend, batch, max_new_tokens=max_new_tokens)
        parsed_at = time.perf_counter()
        action, info = filter_completion(completion)
        actual, strict = to_action_text(action), info['strict_valid']
        timing['parse_filter'] = (time.perf_counter() - parsed_at) * 1000
        details = backend.last_head_record if arm == 'first-key' else None
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    timing['offline_total'] = (time.perf_counter() - started) * 1000
    return {'pair_id': pair_id, 'sample_id': row['sample_id'], 'rgb_sha256': row['rgb']['sha256'],
            'arm': arm, 'completion': completion, 'actual_action': actual, 'generated_tokens': tokens,
            'strict_valid': strict, 'error': error, 'decoder_record': details,
            'text_source': 'model autoregressive generation' if arm == 'full' else '5way whole-window template',
            'timing_ms': timing}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--rollout', type=Path, required=True, help='Completed full-decoder run; initial RGBs only')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, choices=range(1, 7), default=3)
    parser.add_argument('--max-new-tokens', type=int, choices=range(1, 129), default=64)
    args = parser.parse_args()
    source_config = json.loads((args.rollout / 'config.json').read_text(encoding='utf-8'))
    source_status = json.loads((args.rollout / 'run-status.json').read_text(encoding='utf-8'))
    require(source_status['state'] == 'completed' and source_status['failed_episodes'] == 0
            and source_status['completed_episodes'] == source_status['planned_episodes'], 'source rollout incomplete')
    require(source_config['arguments']['history_len'] == 0 and source_config['generation']['decoder'] == 'full',
            'benchmark requires frozen full-decoder history0 source')
    episodes = list(read_jsonl(args.rollout / 'episodes.jsonl'))
    require(0 < len(episodes) <= 48 and len(episodes) == source_status['completed_episodes'],
            'source episode count mismatch or too large')
    audit_feedback_episodes(episodes, args.rollout)
    rows = [{'sample_id': e['episode_id'] + ':0', 'rgb': e['frames'][0]['rgb'],
             'system_prompt': e['system_prompt'], 'history': []} for e in episodes]
    args.output.mkdir(parents=True, exist_ok=False)
    backend = FirstKeyBackend(args.model, adapter=args.adapter)
    require(backend.identity == source_config['model'] and backend.adapter_identity == source_config['adapter'],
            'benchmark must keep source policy weights fixed')
    actual_environment = environment()
    require(actual_environment == source_config['environment'], 'benchmark hardware/software differs from source rollout')
    config = {'schema_version': 1, 'started_utc': datetime.now(timezone.utc).isoformat(),
              'model': backend.identity, 'adapter': backend.adapter_identity, 'environment': actual_environment,
              'source_episodes_sha256': sha256_file(args.rollout / 'episodes.jsonl'),
              'source_config_sha256': sha256_file(args.rollout / 'config.json'),
              'source_code_sha256': {p.name: sha256_file(p) for p in (
                  Path(__file__), Path(__file__).with_name('action_head.py'),
                  Path(__file__).with_name('hf_backend.py'), Path(__file__).with_name('rollout.py'))},
              'samples': [{'id': row['sample_id'], 'rgb_sha256': row['rgb']['sha256']} for row in rows],
              'repeats': args.repeats, 'max_new_tokens_full': args.max_new_tokens,
              'order': 'one resident model/adapter, one recorded warmup each; alternating per pair',
              'timing_boundary': 'decode saved RGB through prepare/model/selection/text parse/filter; no render/network/queue/execution',
              'primary_latency': 'offline_total; inference substage boundaries differ in text serialization inclusion',
              'changed_policy': '5way whole-window action space, forced mouse-zero prefix and text template; weights unchanged'}
    (args.output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    warmups = [measure(backend, rows[0], args.rollout, arm, 'warmup', max_new_tokens=args.max_new_tokens) for arm in ARMS]
    (args.output / 'warmups.json').write_text(json.dumps(warmups, indent=2), encoding='utf-8')
    require(all(not r['error'] for r in warmups), 'warmup failed; inspect saved errors')
    records = []
    for repeat in range(args.repeats):
        for index, row in enumerate(rows):
            order = ARMS if (repeat + index) % 2 == 0 else tuple(reversed(ARMS))
            for arm in order:
                record = measure(backend, row, args.rollout, arm, f'{repeat}:{index}', max_new_tokens=args.max_new_tokens)
                records.append(record)
                with (args.output / 'requests.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(record) + '\n')
                print(json.dumps({'pair': record['pair_id'], 'arm': arm, 'error': record['error'],
                                  'offline_ms': record['timing_ms']['offline_total']}), flush=True)
    summary = {**summarize_pairs(records), 'peak_cuda_allocated_mib': backend.torch.cuda.max_memory_allocated() / 1024 ** 2}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
