"""Record actual expert environment transitions; never control the desktop.

CLI uses real perf-counter timing and ~200ms wall-clock virtual-input windows.
Clock injection is only for synthetic isolated unit tests and changes provenance.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from controller.control_core import to_action_text
from scripts.training.data import convert, require, sha256_file
from scripts.training.feedback import audit_feedback_episodes
from scripts.training.headless import GAME_IDS, HeadlessGame


def collect(output, *, counts=None, clock_factory=None, max_steps=8):
    counts = counts or {'train': 64, 'validation': 8, 'test': 8}
    require(set(counts) == {'train', 'validation', 'test'}
            and all(type(n) is int and 1 <= n <= 128 for n in counts.values()), 'invalid collection counts')
    require(type(max_steps) is int and 1 <= max_steps <= 32, 'invalid collection horizon')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source = output / 'episodes.jsonl'
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    episodes, splits = [], {}
    plan = (('train', 'reach', 0), ('validation', 'hazard', 1000), ('test', 'keydoor', 2000))
    for split, variant, first_seed in plan:
        splits[split] = [GAME_IDS[variant]]
        for seed in range(first_seed, first_seed + counts[split]):
            clock = clock_factory() if clock_factory else None
            game = HeadlessGame(variant=variant, seed=seed, max_steps=max_steps,
                                **({'clock_ns': clock.ns, 'sleep': clock.sleep} if clock else {}))
            episode = {'schema_version': 1, 'episode_id': game.episode_id,
                       'game_id': GAME_IDS[variant], 'game_build': 'procedural-point-grid-v1',
                       'task': variant, 'system_prompt': game.system_prompt(),
                       'input_policy': 'player_observation',
                       'source': {'id': 'headless-expert-collector-v1',
                                  'kind': 'synthetic' if clock else 'recorded',
                                  'license': 'locally-generated-geometric-environment',
                                  'training_allowed': True,
                                  'environment_scope': 'own procedural toy rule variants, not commercial games',
                                  'teacher_inputs': ['privileged_engine_state_for_BFS'],
                                  'student_inputs': ['rgb'],
                                  'teacher_observability': 'grid occupancy/object cells/inventory rendered; audit-only float and timing state withheld',
                                  'clock_scope': 'injected-test-clock' if clock else 'real perf_counter_ns action windows; physics pauses between actions'},
                       'clock': {'id': game.clock_id, 'kind': 'monotonic', 'unit': 'ns'},
                       'action_contract': {'version': 'gamescaling-200ms-v1',
                                           'executor': 'headless-point-grid-six-substeps-v1',
                                           'screen_size': [256, 256]}, 'frames': []}
            while not game.done:
                observation = game.observe()
                frame_path = output / f'{game.episode_id}-{observation["frame_id"]:03}.png'
                observation['image'].save(frame_path)
                proposed = game.expert_action()
                result = game.step(proposed, observed_ns=observation['captured_ns'])
                frame = {'frame_id': observation['frame_id'], 'captured_ns': observation['captured_ns'],
                         'rgb': {'path': frame_path.name, 'sha256': sha256_file(frame_path)},
                         'proposed_action': to_action_text(proposed), 'filtered_action': to_action_text(proposed),
                         **result}
                episode['frames'].append(frame)
            episodes.append(episode)
            # Append only a complete, auditable episode; interrupted final work stays
            # distinguishable from a valid terminal trajectory.
            audit_feedback_episodes([episode], output)
            with source.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(episode) + '\n')
    split_path = output / 'splits.json'
    split_path.write_text(json.dumps(splits, indent=2), encoding='utf-8')
    report = audit_feedback_episodes(episodes, output)
    report.update(source_sha256=sha256_file(source), elapsed_seconds=time.perf_counter() - started,
                  data_kind='synthetic-test-clock' if clock_factory else 'recorded-own-headless-environment',
                  observed_successes=sum(v['reason'] == 'goal' for v in report['outcomes'].values()))
    (output / 'feedback-audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    manifest = convert(source, split_path, output, output / 'converted', history_len=5)
    (output / 'collection.json').write_text(json.dumps({
        'started_utc': started_utc, 'finished_utc': datetime.now(timezone.utc).isoformat(),
        'counts': counts, 'max_steps': max_steps,
        'game_ids': splits, 'converted_samples': manifest['samples'],
        'feedback_audit_sha256': sha256_file(output / 'feedback-audit.json'),
        'claim': 'expert environment recordings and contract audit, not trained-policy success or RL'}, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'outcomes'}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train', type=int, choices=range(1, 129), default=64)
    parser.add_argument('--validation', type=int, choices=range(1, 129), default=8)
    parser.add_argument('--test', type=int, choices=range(1, 129), default=8)
    parser.add_argument('--max-steps', type=int, choices=range(1, 33), default=8)
    args = parser.parse_args()
    collect(args.output, counts={s: getattr(args, s) for s in ('train', 'validation', 'test')},
            max_steps=args.max_steps)


if __name__ == '__main__':
    main()
