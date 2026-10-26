"""Generate an explicitly synthetic static color/action smoke fixture, not gameplay.

Each split has a distinct artificial game id and unique pixels. Rules and visual
distribution are intentionally shared: this tests plumbing, not cross-game skill.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

from scripts.training.data import convert, sha256_file


def generate(output, official_repo, *, seed=20261026, count=8):
    from PIL import Image, ImageDraw
    sys.path.insert(0, str(Path(official_repo).resolve()))
    from gamescaling.prompt import build_system_prompt
    if type(count) is not int or not 1 <= count <= 128:
        raise ValueError('count must be in [1,128]')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rng = random.Random(seed)
    episodes, splits = [], {}
    rule = ('This is a synthetic static color test, not a real game. '
            'If the large central square is red, hold only W in all six steps. '
            'If the large central square is blue, hold only S in all six steps. '
            'Keep mouse movement and scroll zero.')
    for split in ('train', 'validation', 'test'):
        game = f'synthetic-color-{split}-v1'
        splits[split] = [game]
        prompt = build_system_prompt('Synthetic Color Task', rules=rule,
                                     key_bindings={'W': 'red square', 'S': 'blue square'},
                                     mouse_aim=True)
        # One frame per episode: history is deliberately zero for the first experiment.
        for i in range(count):
            key = 'W' if i % 2 == 0 else 'S'
            background = tuple(rng.randrange(25, 70) for _ in range(3))
            image = Image.new('RGB', (256, 256), background)
            draw = ImageDraw.Draw(image)
            draw.rectangle((40, 40, 216, 216), fill=(220, 25, 25) if key == 'W' else (25, 25, 220))
            draw.text((4, 4), f'{split} {i} synthetic', fill=(220, 220, 220))
            name = f'{split}-{i:03}.png'
            image.save(output / name)
            action = '<|action_start|>0 0 0 ; ' + ' ; '.join([key] * 6) + '<|action_end|>'
            episodes.append({
                'schema_version': 1, 'episode_id': f'{game}-{i}', 'game_id': game,
                'game_build': 'generated-static-color-v1', 'task': rule, 'system_prompt': prompt,
                'source': {'id': f'synthetic-generator-v1-seed-{seed}', 'kind': 'synthetic',
                           'license': 'locally-generated-no-third-party-assets', 'training_allowed': True},
                'clock': {'id': f'synthetic-episode-{split}-{i}', 'kind': 'monotonic', 'unit': 'ns'},
                'action_contract': {'version': 'gamescaling-200ms-v1',
                                    'executor': 'synthetic-no-input-v1', 'screen_size': [256, 256]},
                'input_policy': 'player_observation',
                'frames': [{'frame_id': 0, 'captured_ns': 1_000_000_000,
                            'rgb': {'path': name, 'sha256': sha256_file(output / name)},
                            'actual_action': {'text': action, 'start_ns': 1_010_000_000,
                                              'end_ns': 1_210_000_000, 'completed': True},
                            'terminated': True, 'truncated': False,
                            'terminal_reason': 'synthetic-single-decision-complete'}]})
    source, assignment = output / 'episodes.jsonl', output / 'splits.json'
    source.write_text(''.join(json.dumps(e) + '\n' for e in episodes), encoding='utf-8')
    assignment.write_text(json.dumps(splits, indent=2), encoding='utf-8')
    manifest = convert(source, assignment, output, output / 'converted', history_len=0)
    print(json.dumps({'samples': manifest['samples'], 'seed': seed,
                      'evidence_level': 'synthetic pipeline smoke only'}))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--official-repo', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261026)
    parser.add_argument('--count', type=int, default=8)
    args = parser.parse_args()
    generate(args.output, args.official_repo, seed=args.seed, count=args.count)


if __name__ == '__main__':
    main()
