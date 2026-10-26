"""Check design metadata and split boundaries; does not run models or experiments."""
import argparse
import json
from pathlib import Path
import re
import sys


MODALITIES = {'rgb', 'audio', 'depth', 'engine_state'}
STAGES = {'sitp': {'semantic_navigation'}, 'gamescaling': {'sft', 'rl', 'latency'}}


def string_list(value):
    return (isinstance(value, list) and all(isinstance(x, str) and x.strip() for x in value)
            and len(value) == len(set(value)))


def validate_experiment(config, project):
    errors = []
    if not isinstance(config, dict):
        return ['experiment must be an object']
    if config.get('schema_version') != 1 or config.get('project') != project:
        errors.append('project or schema version mismatch')
    if not isinstance(config.get('name'), str) or not re.fullmatch(r'[a-z0-9][a-z0-9-]*', config['name']):
        errors.append('invalid experiment name')
    if config.get('status') != 'design' or config.get('runner_implemented') is not False or config.get('dataset_bound') is not False:
        errors.append('this validator only supports unbound design configurations, not runnable experiments')
    if type(config.get('seed')) is not int or config['seed'] < 0:
        errors.append('seed must be a nonnegative integer')
    if config.get('stage') not in STAGES.get(project, set()):
        errors.append('stage is outside this project')
    lists = {}
    for field in ('observations', 'training_only_observations', 'evaluation_observations', 'metrics'):
        value = config.get(field)
        if not string_list(value):
            errors.append('invalid list: ' + field)
            lists[field] = set()
        else:
            lists[field] = set(value)
    observed, training_only, evaluated = (lists[k] for k in
        ('observations', 'training_only_observations', 'evaluation_observations'))
    if not observed or not evaluated or not lists['metrics']:
        errors.append('observations, evaluation observations and metrics must not be empty')
    if not observed <= MODALITIES:
        errors.append('unknown modality')
    if not training_only <= observed or not evaluated <= observed or training_only & evaluated:
        errors.append('training-only and evaluation observation boundary violated')
    policy = config.get('input_policy')
    if project == 'gamescaling':
        if policy not in ('player_observation', 'privileged'):
            errors.append('invalid player information policy')
        if policy == 'player_observation' and not evaluated <= {'rgb', 'audio'}:
            errors.append('player evaluation cannot use depth or engine state')
        split_unit = 'game'
    else:
        if policy != 'sensor_only' or not observed <= {'rgb', 'depth'}:
            errors.append('SITP sensor-only design must isolate engine labels and truth')
        split_unit = 'scene'
    if config.get('split_unit') != split_unit:
        errors.append('evaluation must be split by ' + split_unit)
    split_ids = config.get('split_ids')
    if not isinstance(split_ids, dict) or set(split_ids) != {'train', 'validation', 'test'}:
        errors.append('train/validation/test split identifiers required')
    else:
        seen = set()
        for label in ('train', 'validation', 'test'):
            value = split_ids[label]
            if not string_list(value) or not value:
                errors.append('invalid split: ' + label)
                continue
            if seen.intersection(value):
                errors.append('overlapping split identifiers')
            seen.update(value)
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    problems = []
    try:
        policy = json.loads((args.root / 'repository-policy.json').read_text(encoding='utf-8'))
        project = policy['project']
        if project not in STAGES:
            raise ValueError('unknown project')
        files = sorted((args.root / 'experiments').glob('*.json'))
        if not files:
            raise ValueError('no design configurations')
        for file in files:
            config = json.loads(file.read_text(encoding='utf-8'))
            problems += [file.name + ': ' + item for item in validate_experiment(config, project)]
        forbidden = ('controller',) if project == 'sitp' else ('ue5',)
        for folder in forbidden:
            if (args.root / folder).exists():
                problems.append('foreign-domain directory is present: ' + folder)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        problems.append('invalid project metadata: ' + str(exc))
    for problem in problems:
        print('BLOCKED: ' + problem, file=sys.stderr)
    if not problems:
        print(f'{project}: {len(files)} design configuration(s) valid; no experiment or training was executed.')
    return int(bool(problems))


if __name__ == '__main__':
    raise SystemExit(main())
