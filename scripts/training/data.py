"""Audit episode JSONL and convert completed actual actions into causal SFT rows.

No model, network, game controller or torch dependency. See docs/data-contract.md.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re


SPLITS = ('train', 'validation', 'test')
KEYS = frozenset(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') +
                 'zero one two three four five six seven eight nine One Two Three Four Five Six Seven Eight Nine Ten Eleven Twelve LB RB MB Up Down Left Right Shift Ctrl Alt Tab Caps Esc Space Enter Back Delete Insert Home End Pause Equal Minus Period Slash Quote'.split())


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def text(value, label):
    require(isinstance(value, str) and bool(value.strip()), f'{label} must be nonempty text')
    return value


def timestamp(value, label):
    require(type(value) is int and value >= 0, f'{label} must be integer nanoseconds')
    return value


def strict_action(value):
    """Accept only exact official action tags, bounds and six steps; never repair."""
    require(isinstance(value, str), 'action must be text')
    match = re.fullmatch(r'<\|action_start\|>(.*?)<\|action_end\|>', value, flags=re.S)
    require(match is not None, 'action must contain only the exact action block')
    parts = match[1].split(';')
    require(len(parts) == 7, 'action requires mouse plus six key steps')
    numbers = parts[0].split()
    require(len(numbers) == 3 and all(re.fullmatch(r'-?\d+', n) for n in numbers),
            'mouse/scroll must be three integers')
    require(all(abs(int(n)) <= bound for n, bound in zip(numbers, (1000, 1000, 5))),
            'mouse/scroll out of range')
    for group in parts[1:]:
        keys = group.split()
        require(len(keys) <= 4 and len(set(keys)) == len(keys) and set(keys) <= KEYS,
                'unknown, duplicate or excessive action keys')
    return value


def checked_rgb(rgb, root):
    from PIL import Image
    require(isinstance(rgb, dict), 'rgb metadata required')
    name = text(rgb.get('path'), 'rgb path')
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and '..' not in relative.parts and '\\' not in name
            and ':' not in name, 'rgb path must stay within data root')
    resolved = (root / name).resolve()
    require(resolved.is_relative_to(root), 'rgb path escapes data root')
    expected = rgb.get('sha256')
    require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected), 'invalid RGB sha256')
    require(sha256_file(resolved) == expected, 'RGB content hash mismatch')
    with Image.open(resolved) as image:
        require(image.width * image.height <= 16_777_216, 'RGB image exceeds audit size limit')
        pixels = image.convert('RGB')
        canonical = f'RGB:{image.width}:{image.height}:'.encode() + pixels.tobytes()
    return {'path': name, 'sha256': expected,
            'pixel_sha256': hashlib.sha256(canonical).hexdigest()}


def audit_episodes(episodes, splits, data_root, *, history_len=5):
    """Return normalized rows and audit counts, or fail closed on ambiguity/leakage."""
    require(type(history_len) is int and 0 <= history_len <= 32, 'invalid history length')
    require(isinstance(splits, dict) and set(splits) == set(SPLITS), 'three splits required')
    game_split = {}
    for split in SPLITS:
        games = splits[split]
        require(isinstance(games, list) and games, 'nonempty game lists required')
        for game in games:
            text(game, 'game_id')
            require(game not in game_split, 'game overlap or duplicate in split assignment')
            game_split[game] = split
    root = Path(data_root).resolve()
    seen_episodes, seen_games, image_splits = set(), set(), {}
    samples, excluded, source_kinds = [], 0, set()
    outcomes, excluded_frames = [], []
    for episode in episodes:
        require(isinstance(episode, dict) and episode.get('schema_version') == 1,
                'unsupported episode schema')
        episode_id = text(episode.get('episode_id'), 'episode_id')
        require(episode_id not in seen_episodes, 'duplicate episode identifier')
        seen_episodes.add(episode_id)
        game = text(episode.get('game_id'), 'game_id')
        require(game in game_split, 'game missing from split assignment')
        seen_games.add(game)
        split = game_split[game]
        source = episode.get('source', {})
        text(source.get('id'), 'source id')
        text(source.get('license'), 'source license')
        require(source.get('training_allowed') is True, 'explicit training permission required')
        require(source.get('kind') in ('synthetic', 'recorded', 'replay'), 'source kind required')
        source_kinds.add(source['kind'])
        require(episode.get('input_policy') == 'player_observation',
                'this RGB converter only supports player_observation')
        clock = episode.get('clock', {})
        text(clock.get('id'), 'clock id')
        require(clock.get('kind') == 'monotonic' and clock.get('unit') == 'ns',
                'one episode-local monotonic nanosecond clock required')
        contract = episode.get('action_contract', {})
        require(contract.get('version') == 'gamescaling-200ms-v1', 'unsupported action contract')
        text(contract.get('executor'), 'executor version')
        size = contract.get('screen_size')
        require(isinstance(size, list) and len(size) == 2
                and all(type(v) is int and v > 0 for v in size), 'screen dimensions required')
        for field in ('game_build', 'task', 'system_prompt'):
            text(episode.get(field), field)
        frames = episode.get('frames')
        require(isinstance(frames, list) and frames, 'episode must contain frames')
        history = []
        last_frame, last_capture, last_action_end = -1, -1, -1
        for index, frame in enumerate(frames):
            require(isinstance(frame, dict), 'frame must be object')
            frame_id = timestamp(frame.get('frame_id'), 'frame_id')
            capture = timestamp(frame.get('captured_ns'), 'captured_ns')
            require(frame_id > last_frame and capture > last_capture, 'frames must strictly increase')
            require(capture >= last_action_end, 'frame precedes previous action completion')
            last_frame, last_capture = frame_id, capture
            terminated, truncated = frame.get('terminated'), frame.get('truncated')
            require(type(terminated) is bool and type(truncated) is bool
                    and not (terminated and truncated), 'invalid terminal flags')
            final = index == len(frames) - 1
            require((terminated or truncated) == final, 'terminal marker must occur exactly at episode end')
            if final:
                text(frame.get('terminal_reason'), 'terminal reason')
            else:
                require(frame.get('terminal_reason') is None, 'nonterminal frame has terminal reason')
            rgb = checked_rgb(frame.get('rgb'), root)
            digest = rgb['pixel_sha256']
            require(digest not in image_splits or image_splits[digest] == split,
                    'image content leakage across game splits')
            image_splits[digest] = split
            action = frame.get('actual_action')
            require(isinstance(action, dict), 'actual action required; proposal is not a label')
            start = timestamp(action.get('start_ns'), 'action start')
            end = timestamp(action.get('end_ns'), 'action end')
            require(capture <= start <= end, 'action/observation time alignment violated')
            require(type(action.get('completed')) is bool, 'actual completion flag required')
            last_action_end = end
            if not action['completed']:
                excluded += 1
                excluded_frames.append({'episode_id': episode_id, 'frame_id': frame_id,
                                        'reason': 'incomplete_actual_execution'})
                history.clear()  # A partially observed execution breaks contiguous history.
                continue
            require(180_000_000 <= end - start <= 220_000_000,
                    'completed action must cover 200ms within 20ms tolerance')
            target = strict_action(action.get('text'))
            sample = {
                'schema_version': 1, 'sample_id': f'{episode_id}:{frame_id}',
                'episode_id': episode_id, 'game_id': game, 'game_build': episode['game_build'],
                'split': split, 'task': episode['task'], 'system_prompt': episode['system_prompt'],
                'source': dict(source), 'input_policy': 'player_observation',
                'clock': dict(clock), 'action_contract': dict(contract),
                'frame_id': frame_id, 'captured_ns': capture, 'rgb': rgb,
                'action_start_ns': start, 'action_end_ns': end, 'action': target,
                'history': [dict(h) for h in history[-history_len:]] if history_len else [],
                'terminated': terminated, 'truncated': truncated,
                'terminal_reason': frame.get('terminal_reason'),
            }
            samples.append(sample)
            history.append({'rgb': rgb, 'action': target, 'frame_id': frame_id,
                            'captured_ns': capture, 'action_end_ns': end})
        outcomes.append({'episode_id': episode_id, 'game_id': game, 'split': split,
                         'terminal_frame_id': frames[-1]['frame_id'],
                         'terminated': frames[-1]['terminated'], 'truncated': frames[-1]['truncated'],
                         'terminal_reason': frames[-1]['terminal_reason']})
    require(seen_games == set(game_split), 'split assignment contains games without episodes')
    counts = Counter(row['split'] for row in samples)
    require(all(counts[s] > 0 for s in SPLITS), 'each split needs completed samples')
    return samples, {'schema_version': 1, 'episodes': len(seen_episodes), 'samples': len(samples),
                     'excluded_incomplete': excluded, 'split_counts': dict(counts),
                     'excluded_frames': excluded_frames, 'episode_outcomes': outcomes,
                     'source_kinds': sorted(source_kinds), 'history_len': history_len,
                     'claims': 'data contract audit only; not a model capability or game-success test'}


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as file:
        for index, line in enumerate(file, 1):
            try:
                yield json.loads(line)
            except ValueError as exc:
                raise ValueError(f'invalid JSON at line {index}') from exc


def convert(source, split_path, data_root, output, *, history_len=5):
    samples, report = audit_episodes(read_jsonl(source),
                                    json.loads(Path(split_path).read_text(encoding='utf-8')),
                                    data_root, history_len=history_len)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for split in SPLITS:
        path = output / f'{split}.jsonl'
        path.write_text(''.join(json.dumps(row, sort_keys=True) + '\n'
                                for row in samples if row['split'] == split), encoding='utf-8')
        files[split] = {'path': path.name, 'sha256': sha256_file(path)}
    manifest = {**report, 'source_sha256': sha256_file(source),
                'splits_sha256': sha256_file(split_path), 'files': files,
                'source_path': str(Path(source).resolve()),
                'splits_path': str(Path(split_path).resolve()),
                'data_root': str(Path(data_root).resolve())}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=Path, required=True)
    parser.add_argument('--splits', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--history-len', type=int, default=5)
    args = parser.parse_args()
    report = convert(args.episodes, args.splits, args.data_root, args.output,
                     history_len=args.history_len)
    print(json.dumps({k: v for k, v in report.items() if k != 'data_root'}, indent=2))


if __name__ == '__main__':
    main()
