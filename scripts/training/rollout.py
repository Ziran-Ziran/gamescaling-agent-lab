"""RGB-only policy rollouts with real rewards in the own paused headless toy world."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import time

from controller.control_core import SafeAction, sanitize_action, to_action_text
from scripts.training.data import require, sha256_file
from scripts.training.feedback import audit_feedback_episodes
from scripts.training.headless import GAME_IDS, HeadlessGame
from scripts.training.metrics import action_signature


def filter_completion(completion):
    """Never repair malformed text; record strict parsing separately from filtering."""
    signature = action_signature(completion)
    if signature is None:
        action = SafeAction((0., 0., 0.), ((),) * 6)
        return action, {'strict_valid': False, 'fallback': True, 'filtered': False}
    mouse, steps = signature
    action = sanitize_action(mouse, [list(s) for s in steps], allow_mouse=False)
    return action, {'strict_valid': True, 'fallback': False,
                    'filtered': signature != action_signature(to_action_text(action))}


def _append_json(path, value):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')
        stream.flush()


def run_episode(policy, output, *, variant, seed, max_steps=8, history_len=0, clock=None):
    """Policy receives only a new prompt/RGB/history dictionary; never the game object."""
    require(type(history_len) is int and 0 <= history_len <= 5, 'invalid rollout history')
    root = Path(output)
    game = HeadlessGame(variant=variant, seed=seed, max_steps=max_steps,
                        **({'clock_ns': clock.ns, 'sleep': clock.sleep} if clock else {}))
    episode = {'schema_version': 1, 'episode_id': game.episode_id, 'game_id': GAME_IDS[variant],
               'game_build': 'procedural-point-grid-v1', 'task': variant,
               'system_prompt': game.system_prompt(), 'input_policy': 'player_observation',
               'source': {'id': 'headless-rgb-policy-rollout-v1',
                          'kind': 'synthetic' if clock else 'recorded',
                          'license': 'locally-generated-geometric-environment', 'training_allowed': True,
                          'student_inputs': ['rgb', 'actual_completed_action_history'],
                          'environment_scope': 'own paused procedural toy, not commercial games',
                          'clock_scope': 'injected-test-clock' if clock else 'real perf_counter_ns'},
               'clock': {'id': game.clock_id, 'kind': 'monotonic', 'unit': 'ns'},
               'action_contract': {'version': 'gamescaling-200ms-v1',
                                   'executor': 'headless-point-grid-six-substeps-v1',
                                   'screen_size': [256, 256]}, 'frames': []}
    _append_json(root / 'episode-status.jsonl', {'status': 'started',
                 'episode': {k: v for k, v in episode.items() if k != 'frames'}})
    try:
        _run_steps(policy, root, game, episode, history_len)
    except BaseException as exc:
        _append_json(root / 'episode-status.jsonl', {'episode_id': game.episode_id, 'status': 'failed',
                     'completed_actions': len(episode['frames']), 'error_type': type(exc).__name__})
        raise
    _append_json(root / 'episode-status.jsonl', {'episode_id': game.episode_id, 'status': 'completed',
                 'completed_actions': len(episode['frames'])})
    return episode


def _run_steps(policy, root, game, episode, history_len):
    history = []
    while not game.done:
        loop_started = time.perf_counter()
        observation = game.observe()
        rendered_at = time.perf_counter()
        image = root / f'{game.episode_id}-{observation["frame_id"]:03}.png'
        require(not image.exists(), 'rollout frame already exists')
        observation['image'].save(image)
        rgb = {'path': image.name, 'sha256': sha256_file(image)}
        saved_at = time.perf_counter()
        # Deliberately build this allowlisted student payload instead of passing observation,
        # episode, game, teacher action, return, termination flags or audit truth.
        row = {'system_prompt': game.system_prompt(), 'rgb': dict(rgb),
               'history': [{'rgb': dict(item['rgb']), 'action': item['action']}
                           for item in history[-history_len:]] if history_len else []}
        decision = policy(row)
        parsed_at = time.perf_counter()
        action, decision_info = filter_completion(decision['completion'])
        filtered_at = time.perf_counter()
        _append_json(root / 'decisions.jsonl', {'episode_id': game.episode_id,
                     'frame_id': observation['frame_id'], 'rgb': rgb,
                     'completion': decision['completion'], 'policy': decision_info,
                     'planned_action': to_action_text(action), 'completed_execution': False})
        result = game.step(action, observed_ns=observation['captured_ns'])
        execution = result['actual_action']
        timing = dict(decision['timing_ms'])
        timing.update(render=(rendered_at - loop_started) * 1000,
                      png_encode_write_hash=(saved_at - rendered_at) * 1000,
                      policy_call=(parsed_at - saved_at) * 1000,
                      parse_filter=(filtered_at - parsed_at) * 1000,
                      execution=(execution['end_ns'] - execution['start_ns']) / 1e6,
                      observation_to_action_start=(execution['start_ns'] - observation['captured_ns']) / 1e6,
                      observation_to_action_end=(execution['end_ns'] - observation['captured_ns']) / 1e6,
                      loop_including_feedback_audit=(time.perf_counter() - loop_started) * 1000,
                      transport=None, queue=None)
        frame = {'frame_id': observation['frame_id'], 'captured_ns': observation['captured_ns'],
                 'rgb': rgb, 'proposed_action': decision['completion'],
                 'filtered_action': to_action_text(action), **result,
                 'policy': {**decision_info, 'generated_tokens': decision['generated_tokens']},
                 'timing_ms': timing}
        episode['frames'].append(frame)
        _append_json(root / 'transitions.jsonl', {'episode_id': game.episode_id, 'frame': frame})
        # game.step has already validated completed action physics and timing.
        history.append({'rgb': rgb, 'action': execution['text']})
    audit_feedback_episodes([episode], root)


def _timings(frames):
    report = {}
    for key in sorted({key for f in frames for key in f['timing_ms']}):
        values = sorted(f['timing_ms'][key] for f in frames if f['timing_ms'].get(key) is not None)
        report[key] = {'p50': statistics.median(values),
                       'p95': values[math.ceil(.95 * len(values)) - 1]} if values else None
    return report


def summarize_episodes(episodes):
    require(bool(episodes), 'cannot summarize no episodes')
    report = {}
    for variant in GAME_IDS:
        group = [e for e in episodes if e['game_id'] == GAME_IDS[variant]]
        if not group:
            continue
        frames = [f for e in group for f in e['frames']]
        reasons = Counter(e['frames'][-1]['terminal_reason'] for e in group)
        report[variant] = {'episodes': len(group), 'actions': len(frames), 'reasons': dict(reasons),
                           'success_rate': reasons['goal'] / len(group),
                           'mean_return': statistics.mean(sum(f['reward'] for f in e['frames']) for e in group),
                           'mean_steps': len(frames) / len(group),
                           'strict_valid_rate': statistics.mean(f['policy']['strict_valid'] for f in frames),
                           'fallback_rate': statistics.mean(f['policy']['fallback'] for f in frames),
                           'filtered_rate': statistics.mean(f['policy']['filtered'] for f in frames),
                           'timing_ms': _timings(frames)}
    return report


def main():
    from scripts.training.hf_backend import RGBBackend, environment
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes-per-game', type=int, choices=range(1, 17), default=4)
    parser.add_argument('--seed-start', type=int, default=3000)
    parser.add_argument('--max-steps', type=int, choices=range(1, 33), default=8)
    parser.add_argument('--history-len', type=int, choices=range(0, 6), default=0)
    parser.add_argument('--max-new-tokens', type=int, choices=range(1, 257), default=64)
    args = parser.parse_args()
    require(args.seed_start >= 0, 'invalid seed start')
    args.output.mkdir(parents=True, exist_ok=False)
    status = {'state': 'initializing', 'planned_episodes': 3 * args.episodes_per_game,
              'completed_episodes': 0, 'failed_episodes': 0, 'active_episode': None,
              'started_utc': datetime.now(timezone.utc).isoformat()}
    _write_status(args.output, status)
    try:
        _execute(args, status)
    except BaseException as exc:
        status.update(state='failed', error_type=type(exc).__name__,
                      failed_episodes=int(status['active_episode'] is not None))
        raise
    finally:
        status['finished_utc'] = datetime.now(timezone.utc).isoformat()
        _write_status(args.output, status)


def _write_status(output, status):
    path = output / 'run-status.json'
    temporary = output / 'run-status.tmp'
    temporary.write_text(json.dumps(status, indent=2), encoding='utf-8')
    temporary.replace(path)


def _execute(args, status):
    from scripts.training.hf_backend import RGBBackend, environment
    started_utc = datetime.now(timezone.utc).isoformat()
    backend = RGBBackend(args.model, adapter=args.adapter)
    config = {'schema_version': 1, 'started_utc': started_utc, 'model': backend.identity,
              'adapter': backend.adapter_identity, 'environment': environment(),
              'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'generation': {'do_sample': False, 'stop': '<|action_end|>', 'reasoning': False},
              'cache': 'no warmup excluded; first request cold, sequential warm weights; no shared KV cache',
              'task_boundary': 'own headless rule variants; world paused during inference; not real-time commercial gameplay',
              'actor_inputs': 'RGB + fixed visible rules + completed action history only',
              'teacher_or_reward_inputs': False, 'learning_during_rollout': False,
              'timing_boundary': 'render/save/prepare/infer/filter, then actual200ms; observation timestamps start after render',
              'source_code_sha256': {p.name: sha256_file(p) for p in (
                  Path(__file__), Path(__file__).with_name('headless.py'),
                  Path(__file__).with_name('feedback.py'), Path(__file__).with_name('hf_backend.py'),
                  Path(__file__).with_name('formatting.py'), Path(__file__).with_name('metrics.py'),
                  Path('controller/control_core.py'))}}
    (args.output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')

    def policy(row):
        batch, _, timing = backend.prepare(row, args.output, history_len=args.history_len)
        completion, tokens, inference_ms = backend.generate(batch, max_new_tokens=args.max_new_tokens)
        return {'completion': completion, 'generated_tokens': tokens,
                'timing_ms': {**timing, 'inference': inference_ms}}

    episodes = []
    status['state'] = 'running'
    for offset, variant in enumerate(GAME_IDS):
        for seed in range(args.seed_start + offset * 1000,
                          args.seed_start + offset * 1000 + args.episodes_per_game):
            status['active_episode'] = f'{GAME_IDS[variant]}-seed-{seed}'
            _write_status(args.output, status)
            episode = run_episode(policy, args.output, variant=variant, seed=seed,
                                  max_steps=args.max_steps, history_len=args.history_len)
            with (args.output / 'episodes.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(episode) + '\n')
            episodes.append(episode)
            status.update(completed_episodes=len(episodes), active_episode=None)
            _write_status(args.output, status)
            print(json.dumps({'episode': episode['episode_id'], 'steps': len(episode['frames']),
                              'reason': episode['frames'][-1]['terminal_reason'],
                              'return': sum(f['reward'] for f in episode['frames'])}), flush=True)
    summary = {'results': summarize_episodes(episodes),
               'feedback_audit': audit_feedback_episodes(episodes, args.output),
               'peak_cuda_allocated_mib': backend.torch.cuda.max_memory_allocated() / 1024 ** 2,
               'finished_utc': datetime.now(timezone.utc).isoformat(),
               'episodes_sha256': sha256_file(args.output / 'episodes.jsonl'),
               'evidence_level': 'actual RGB policy/reward in paused own toy environment; no RL or commercial-game claim'}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    status['state'] = 'completed'
    print(json.dumps({k: v for k, v in summary.items() if k != 'feedback_audit'}, indent=2))


if __name__ == '__main__':
    main()
