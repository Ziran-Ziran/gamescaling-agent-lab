"""Validate only staged files and staged publication policy; never echo secrets."""
import argparse
import json
import os
from pathlib import PurePosixPath
import re
import subprocess
import sys


BLOCKED_PARTS = {
    'output', 'tmp', 'inputs', 'research', 'content', 'media', 'saved',
    'intermediate', 'binaries', 'deriveddatacache', 'node_modules',
    '__pycache__', '.venv', 'venv', '.ssh', '.aws', '.azure', '.config',
    '.git', 'weights', 'checkpoints', 'datasets', 'logs',
}
PATTERNS = {
    'private key': re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
    'GitHub token': re.compile(rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b'),
    'cloud access key': re.compile(rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'model access token': re.compile(rb'\bhf_[A-Za-z0-9]{30,}\b'),
    'credential in URL': re.compile(rb'https?://[^\s/@:]+:[^\s/@]+@'),
}
TEXT_SUFFIXES = {'.py', '.sh', '.ps1', '.md', '.txt', '.yml', '.yaml', '.json',
                 '.cpp', '.h', '.cs', '.ini', '.uproject'}


def git(*args):
    return subprocess.check_output(['git', *args], stderr=subprocess.PIPE)


def load_policy():
    value = json.loads(git('show', ':repository-policy.json'))
    if not isinstance(value, dict) or value.get('schema_version') != 1:
        raise ValueError('unsupported policy')
    for key in ('allowed_files', 'allowed_trees'):
        paths = value.get(key)
        if not isinstance(paths, list) or not all(isinstance(x, str) and x for x in paths):
            raise ValueError('invalid path list')
        for name in paths:
            p = PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
                raise ValueError('invalid policy path')
    if not all(x.endswith('/') for x in value['allowed_trees']):
        raise ValueError('tree entries require slash')
    return value


def allowed(path, policy):
    p = PurePosixPath(path)
    if p.is_absolute() or '..' in p.parts or any(x.lower() in BLOCKED_PARTS for x in p.parts):
        return False
    name = p.name.lower()
    if name.startswith('.env') or '.local.' in name or 'credentials' in name or 'secrets' in name:
        return False
    if path in policy['allowed_files']:
        return True
    return any(path.startswith(prefix) for prefix in policy['allowed_trees']) and p.suffix in TEXT_SUFFIXES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staged-only', action='store_true')
    args = parser.parse_args()
    try:
        os.chdir(git('rev-parse', '--show-toplevel').decode().strip())
        policy = load_policy()
    except (OSError, subprocess.CalledProcessError, ValueError, UnicodeError):
        print('BLOCKED: missing or invalid staged repository-policy.json', file=sys.stderr)
        return 1
    command = ('diff', '--cached', '--name-only', '--diff-filter=ACMR', '-z') if args.staged_only else ('ls-files', '--cached', '-z')
    paths = sorted(set(git(*command).decode('utf-8').strip('\0').split('\0')) - {''})
    problems = []
    for path in paths:
        if not allowed(path, policy):
            problems.append((path, 'path is outside publication policy'))
        records = git('ls-files', '--stage', '--', path).decode().splitlines()
        if len(records) != 1 or not records[0].startswith(('100644 ', '100755 ')) or records[0].split()[2] != '0':
            problems.append((path, 'only regular stage-zero files may be published'))
            continue
        data = git('show', ':' + path)
        if len(data) > 1024 * 1024:
            problems.append((path, 'file exceeds 1 MiB source-only limit'))
        if b'\0' in data:
            problems.append((path, 'binary content is not allowed'))
        for label, pattern in PATTERNS.items():
            if pattern.search(data):
                problems.append((path, 'possible ' + label))
    for path, reason in problems:
        print(f'BLOCKED: {path}: {reason}', file=sys.stderr)
    print(f'Checked {len(paths)} indexed files; {len(problems)} problem(s).')
    return int(bool(problems))


if __name__ == '__main__':
    raise SystemExit(main())
