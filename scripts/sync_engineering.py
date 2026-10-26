"""Verify or explicitly synchronize a fixed, versioned engineering bundle."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile


CANONICAL_REPOSITORY = 'https://github.com/Ziran-Ziran/gamescaling-agent-lab'
BUNDLE_FILES = (
    '.editorconfig', '.gitattributes', 'SECURITY.md',
    '.github/ISSUE_TEMPLATE/bug_report.md', '.github/ISSUE_TEMPLATE/feature_request.md',
    '.github/pull_request_template.md', 'scripts/check_repository.py',
    'scripts/sync_engineering.py', 'scripts/validate_project.py',
    'tests/test_repository_policy.py', 'tests/test_engineering.py',
)
MANIFEST = 'engineering/manifest.json'


def regular(root, name):
    path = root / name
    for part in (path, *path.parents):
        if part.is_symlink() or (hasattr(os.path, 'isjunction') and os.path.isjunction(part)):
            raise ValueError('linked path is not allowed: ' + name)
        if part == root:
            break
    if not path.is_file():
        raise ValueError('missing regular file: ' + name)
    return path


def load_bundle(root):
    # Do not resolve symlinks before checking the explicitly selected root.
    root = Path(os.path.abspath(root))
    manifest_bytes = regular(root, MANIFEST).read_bytes()
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict) or manifest.get('schema_version') != 1:
        raise ValueError('unsupported manifest')
    if manifest.get('canonical_repository') != CANONICAL_REPOSITORY:
        raise ValueError('unexpected canonical repository')
    if not isinstance(manifest.get('version'), str) or not re.fullmatch(r'\d+\.\d+\.\d+', manifest['version']):
        raise ValueError('invalid bundle version')
    hashes = manifest.get('files')
    if not isinstance(hashes, dict) or set(hashes) != set(BUNDLE_FILES):
        raise ValueError('manifest does not match the fixed bundle file list')
    data = {}
    for name in BUNDLE_FILES:
        content = regular(root, name).read_bytes()
        expected = hashes[name]
        if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected):
            raise ValueError('invalid digest: ' + name)
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError('hash mismatch (local edit or damaged source): ' + name)
        data[name] = content
    data[MANIFEST] = manifest_bytes
    return manifest, data


def atomic_write(path, content):
    # Existing files and parents were checked before any mutation.
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.engineering-', delete=False) as stream:
        pending = Path(stream.name)
        stream.write(content)
    try:
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1], help=argparse.SUPPRESS)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    try:
        local, before = load_bundle(args.root)
        if args.apply and args.source_root is None:
            raise ValueError('--apply requires an explicit reviewed --source-root')
        if args.source_root is None:
            print(f'Engineering bundle {local["version"]}: {len(BUNDLE_FILES)} files verified.')
            return 0
        source, after = load_bundle(args.source_root)
        changed = [name for name in before if before[name] != after[name]]
        if changed and not args.apply:
            print(f'Bundle differences: {len(changed)} file(s); no files changed.', file=sys.stderr)
            return 1
        written = []
        try:
            for name in changed:
                atomic_write(regular(args.root, name), after[name])
                written.append(name)
        except OSError:
            for name in reversed(written):
                atomic_write(regular(args.root, name), before[name])
            raise
        print(f'Engineering bundle {source["version"]}: {len(changed)} file(s) synchronized.')
        return 0
    except (OSError, ValueError, TypeError) as exc:
        print('BLOCKED: ' + str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
