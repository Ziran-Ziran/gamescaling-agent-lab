"""Collect and independently bind dual-color RGB, causal WAV and actual feedback."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from controller.control_core import to_action_text
from scripts.training.audio import read_audio
from scripts.training.audio_cue import (AudioCueGame, GAMES, PROJECTION, SENSOR,
                                        projected_rgb, waveform_bytes)
from scripts.training.data import checked_rgb, convert, read_jsonl, require, sha256_file
from scripts.training.feedback import validate_feedback
from scripts.training.hf_backend import load_dataset


def audit_audio_episodes(episodes, root, *, require_pairs=True):
    root = Path(root).resolve()
    seen, groups, audio_index, returns = set(), {}, {}, []
    for episode in episodes:
        spec = episode['cue_reset']
        game = AudioCueGame(**spec)
        require(episode['episode_id'] == game.base.episode_id and episode['episode_id'] not in seen,
                'audio episode identity or uniqueness mismatch')
        seen.add(episode['episode_id'])
        require(episode['game_id'] == GAMES[spec['variant']]
                and episode['game_build'] == PROJECTION
                and episode['system_prompt'] == game.system_prompt()
                and episode['clock']['id'] == game.base.clock_id, 'audio game/projection/clock mismatch')
        frames = episode['frames']
        require(bool(frames) and frames[0]['feedback']['state_before'] == game.base.snapshot(),
                'audio episode must start at declared reset')
        previous, total = None, 0.
        for index, frame in enumerate(frames):
            feedback = validate_feedback(frame['feedback'])
            require(feedback['episode_id'] == episode['episode_id']
                    and feedback['clock_id'] == episode['clock']['id'], 'audio feedback identity mismatch')
            require(feedback['frame_id'] == frame['frame_id'] == index
                    and feedback['observed_ns'] == frame['captured_ns'], 'audio frame clock mismatch')
            if previous is not None:
                require(previous == feedback['state_before_sha256'], 'broken audio state chain')
            previous = feedback['state_after_sha256']
            actual = frame['actual_action']
            require(actual['text'] == feedback['action_text'] and actual['completed'] is True
                    and actual['start_ns'] == feedback['action_start_ns']
                    and actual['end_ns'] == feedback['action_end_ns'], 'audio actual action binding mismatch')
            require(all(frame[k] == feedback[k] for k in ('reward', 'terminated', 'truncated', 'terminal_reason')),
                    'audio outcome binding mismatch')
            require((feedback['terminated'] or feedback['truncated']) == (index == len(frames)-1),
                    'audio terminal episode boundary mismatch')
            rgb = checked_rgb(frame['rgb'], root)
            image = projected_rgb(feedback['state_before'], spec)
            canonical = f'RGB:{image.width}:{image.height}:'.encode() + image.tobytes()
            require(rgb['pixel_sha256'] == hashlib.sha256(canonical).hexdigest(), 'audio RGB projection mismatch')
            record = frame['audio']
            observation = read_audio(record, root, captured_ns=frame['captured_ns'], clock_id=episode['clock']['id'])
            require(record.get('sensor_version') == SENSOR and observation['present'], 'source audio must be causal and present')
            expected = hashlib.sha256(waveform_bytes(spec['context'], spec['tone'])).hexdigest()
            require(record['sha256'] == expected, 'cue waveform disagrees with virtual sensor reset')
            if index:
                require(record['window_start_ns'] >= frames[index-1]['actual_action']['end_ns'],
                        'audio window overlaps previous execution')
            audio_index[f'{episode["episode_id"]}:{index}'] = {'audio': record,
                'captured_ns': frame['captured_ns'], 'clock_id': episode['clock']['id']}
            total += feedback['reward']
        key = (spec['variant'], spec['context'], spec['colors'])
        groups.setdefault(key, {})[spec['tone']] = checked_rgb(frames[0]['rgb'], root)['pixel_sha256']
        returns.append({'episode': episode['episode_id'], 'return': total,
                        'reason': frames[-1]['terminal_reason'], 'steps': len(frames)})
    require(bool(seen), 'no audio episodes')
    if require_pairs:
        for tones in groups.values():
            require(set(tones) == {0, 1} and tones[0] == tones[1], 'opposite cue pairs need identical initial RGB')
        mirrors = {(v, c) for v, c, _ in groups}
        require(all((v, c, 0) in groups and (v, c, 1) in groups for v, c in mirrors),
                'each context requires counterbalanced color positions')
    return {'schema_version': 1, 'projection': PROJECTION, 'sensor': SENSOR, 'episodes': len(seen),
            'frames': len(audio_index), 'successes': sum(r['reason'] == 'goal' for r in returns),
            'outcomes': returns, 'audio_index': audio_index}


def record_episode(game, root, *, clock_injected=False, policy=None):
    """Keep truth in the ledger. The optional actor gets only current player sensors."""
    root = Path(root)
    episode = {'schema_version': 1, 'episode_id': game.base.episode_id,
               'game_id': GAMES[game.spec['variant']], 'game_build': PROJECTION,
               'task': game.spec['variant'], 'system_prompt': game.system_prompt(),
               'input_policy': 'player_observation', 'cue_reset': dict(game.spec),
               'source': {'id': 'audio-cue-policy-v1' if policy else 'audio-cue-local-collector-v1',
                          'kind': 'synthetic' if clock_injected else 'recorded',
                          'license': 'locally-generated-geometric-environment-and-procedural-WAV',
                          'training_allowed': True, 'student_inputs': ['rgb', 'causal_virtual_audio'],
                          'teacher_inputs': [] if policy else ['privileged_engine_BFS_and_active_target'],
                          'sensor_scope': 'virtual deterministic WAV, no microphone or real game capture',
                          'clock_scope': 'injected-test-clock' if clock_injected else 'actual200ms sensor and action windows'},
               'clock': {'id': game.base.clock_id, 'kind': 'monotonic', 'unit': 'ns'},
               'action_contract': {'version': 'gamescaling-200ms-v1',
                                   'executor': 'headless-point-grid-six-substeps-v1', 'screen_size': [256, 256]},
               'frames': []}
    status_path = root / 'episode-status.jsonl'
    _append(status_path, {'episode_id': game.base.episode_id, 'state': 'started'})
    try:
        while not game.base.done:
            observation = game.observe(root)
            image = root / f'{game.base.episode_id}-{observation["frame_id"]:03}.png'
            require(not image.exists(), 'audio RGB output already exists')
            observation['image'].save(image)
            rgb = {'path': image.name, 'sha256': sha256_file(image)}
            row = {'system_prompt': game.system_prompt(), 'rgb': dict(rgb), 'history': [],
                   'audio': dict(observation['audio']), 'captured_ns': observation['captured_ns'],
                   'clock_id': game.base.clock_id}
            decision = policy(row) if policy else None
            if policy:
                from scripts.training.rollout import filter_completion
                action, info = filter_completion(decision['completion'])
            else:
                action, info = game.base.expert_action(), None
            text = to_action_text(action)
            _append(root / 'decisions.jsonl', {'episode_id': game.base.episode_id, 'frame_id': observation['frame_id'],
                    'proposed_action': decision['completion'] if policy else text, 'completed_execution': False})
            result = game.base.step(action, observed_ns=observation['captured_ns'])
            frame = {'frame_id': observation['frame_id'], 'captured_ns': observation['captured_ns'],
                     'rgb': rgb, 'audio': observation['audio'], 'proposed_action': decision['completion'] if policy else text,
                     'filtered_action': text, **result}
            if policy:
                frame['policy'] = {**info, **decision.get('evidence', {})}
            episode['frames'].append(frame)
            _append(root / 'transitions.jsonl', {'episode_id': game.base.episode_id, 'frame': frame})
        audit_audio_episodes([episode], root, require_pairs=False)
    except BaseException as error:
        _append(status_path, {'episode_id': game.base.episode_id, 'state': 'failed',
                             'completed_actions': len(episode['frames']), 'error_type': type(error).__name__})
        raise
    _append(status_path, {'episode_id': game.base.episode_id, 'state': 'completed',
                         'completed_actions': len(episode['frames'])})
    return episode


def _append(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')


def collect(output, *, contexts=None, clock_factory=None):
    contexts = contexts or {'train': 16, 'validation': 4, 'test': 2}
    require(set(contexts) == {'train', 'validation', 'test'} and all(type(n) is int and 1 <= n <= 32
            for n in contexts.values()), 'invalid audio context counts')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started, episodes, splits = time.perf_counter(), [], {}
    for split, variant, first in (('train', 'reach', 20000), ('validation', 'hazard', 21000), ('test', 'slalom', 22000)):
        splits[split] = [GAMES[variant]]
        for context in range(first, first+contexts[split]):
            for colors in (0, 1):
                for tone in (0, 1):
                    clock = clock_factory() if clock_factory else None
                    game = AudioCueGame(context=context, colors=colors, tone=tone, variant=variant,
                                        **({'clock_ns': clock.ns, 'sleep': clock.sleep} if clock else {}))
                    episode = record_episode(game, output, clock_injected=bool(clock))
                    episodes.append(episode)
                    _append(output / 'episodes.jsonl', episode)
    report = audit_audio_episodes(episodes, output)
    split_file = output / 'splits.json'
    split_file.write_text(json.dumps(splits, indent=2), encoding='utf-8')
    manifest = convert(output/'episodes.jsonl', split_file, output, output/'converted', history_len=0)
    index = output / 'converted/audio-index.json'
    index.write_text(json.dumps(report['audio_index'], indent=2), encoding='utf-8')
    audio_manifest = {'schema_version': 1, 'projection': PROJECTION, 'sensor': SENSOR,
                      'source_sha256': manifest['source_sha256'], 'audio_index_sha256': sha256_file(index)}
    (output/'converted/audio-manifest.json').write_text(json.dumps(audio_manifest, indent=2), encoding='utf-8')
    report = {k: v for k, v in report.items() if k != 'audio_index'}
    report.update(elapsed_seconds=time.perf_counter()-started, source_sha256=manifest['source_sha256'],
                  finished_utc=datetime.now(timezone.utc).isoformat(),
                  claim='actual own virtual-audio expert recordings, not model learning or real-game audio')
    (output/'audio-audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'outcomes'}, indent=2), flush=True)
    return report


def load_audio_dataset(directory):
    directory = Path(directory)
    dataset, manifest, digest = load_dataset(directory)
    extra = json.loads((directory/'audio-manifest.json').read_text(encoding='utf-8'))
    require(extra == {'schema_version': 1, 'projection': PROJECTION, 'sensor': SENSOR,
                      'source_sha256': manifest['source_sha256'],
                      'audio_index_sha256': sha256_file(directory/'audio-index.json')}, 'audio index manifest mismatch')
    index = json.loads((directory/'audio-index.json').read_text(encoding='utf-8'))
    fresh = audit_audio_episodes(read_jsonl(manifest['source_path']), manifest['data_root'])
    require(index == fresh['audio_index'], 'audio index differs from freshly audited source')
    audio_splits = {}
    for split, rows in dataset.items():
        for row in rows:
            digest_audio = index[row['sample_id']]['audio']['sha256']
            require(digest_audio not in audio_splits or audio_splits[digest_audio] == split,
                    'waveform content leakage across game splits')
            audio_splits[digest_audio] = split
    return dataset, manifest, {'rgb': digest, 'audio': sha256_file(directory/'audio-manifest.json')}, index


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    collect(args.output)
