"""Strict saved-trajectory metrics; static matching is not game task success."""
import math
import re
import statistics

from scripts.training.data import strict_action


def action_signature(value):
    value = value.strip()
    try:
        strict_action(value)
    except ValueError:
        return None
    body = re.fullmatch(r'<\|action_start\|>(.*?)<\|action_end\|>', value, re.S)[1]
    parts = body.split(';')
    return (tuple(map(int, parts[0].split())),
            tuple(tuple(sorted(group.split())) for group in parts[1:]))


def summarize(rows):
    if not rows:
        raise ValueError('cannot summarize empty evaluation')
    timings = {}
    for key in rows[0]['timing_ms']:
        values = sorted(r['timing_ms'][key] for r in rows if r['timing_ms'][key] is not None)
        timings[key] = ({'p50': statistics.median(values),
                         'p95': values[math.ceil(.95 * len(values)) - 1]} if values else None)
    tokens = sum(r['target_tokens'] for r in rows)
    return {'samples': len(rows),
            'strict_valid_rate': sum(r['strict_valid'] for r in rows) / len(rows),
            'action_exact_match_rate': sum(r['action_match'] for r in rows) / len(rows),
            'token_weighted_loss': sum(r['loss'] * r['target_tokens'] for r in rows) / tokens,
            'timing_ms': timings,
            'p95_method': 'nearest rank; small static samples do not estimate online tail latency',
            'evidence_level': 'saved/synthetic action matching only; no game success or generalization claim'}
