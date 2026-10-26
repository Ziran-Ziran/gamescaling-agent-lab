"""Reaudit a copied RGB v1 dataset; optionally create a NEW location-bound manifest.

CPU only. Original files and checkpoint identities are never rewritten. A content
fingerprint is a migration witness, not permission to resume an old training run.
"""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.training.data import SPLITS, audit_episodes, require


LOCATIONS = frozenset(('source_path', 'splits_path', 'data_root'))


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON object key')
        result[key] = value
    return result


def _json(data):
    return json.loads(data, object_pairs_hook=_unique_object)


def _jsonl(data):
    return [_json(line) for line in data.decode('utf-8').splitlines()]


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _audit(dataset, episodes, splits, data_root):
    dataset = Path(dataset).resolve()
    manifest_bytes = (dataset / 'manifest.json').read_bytes()
    manifest = _json(manifest_bytes)
    require(isinstance(manifest, dict) and type(manifest.get('schema_version')) is int
            and manifest['schema_version'] == 1,
            'unsupported converted manifest schema')
    require(all(isinstance(manifest.get(k), str) and manifest[k].strip() for k in LOCATIONS),
            'original manifest location fields required')
    source_bytes, splits_bytes = Path(episodes).read_bytes(), Path(splits).read_bytes()
    require(_sha(source_bytes) == manifest.get('source_sha256'), 'source hash changed')
    require(_sha(splits_bytes) == manifest.get('splits_sha256'), 'split assignment hash changed')
    rows, audit = audit_episodes(_jsonl(source_bytes), _json(splits_bytes), data_root,
                                 history_len=manifest.get('history_len'))
    files, payloads = {}, {}
    for split in SPLITS:
        name = f'{split}.jsonl'
        info = manifest.get('files', {}).get(split, {})
        require(info.get('path') == name, 'unexpected split file path')
        path = (dataset / name).resolve()
        require(path.is_relative_to(dataset), 'converted file escapes dataset root')
        payloads[name] = path.read_bytes()
        require(_sha(payloads[name]) == info.get('sha256'), 'converted data hash changed')
        require(_canonical(_jsonl(payloads[name])) ==
                _canonical([row for row in rows if row['split'] == split]),
                'converted rows differ from audited source actions/history')
        files[split] = {'path': name, 'sha256': _sha(payloads[name])}
    content = {**audit, 'source_sha256': _sha(source_bytes),
               'splits_sha256': _sha(splits_bytes), 'files': files}
    canonical = _canonical(content)
    require(_canonical({k: v for k, v in manifest.items() if k not in LOCATIONS}) == canonical,
            'manifest audit metadata or fields differ from reconstructed source')
    report = {'schema_version': 1, 'content_identity_version': 'rgb-v1-location-free-v1',
              'manifest_sha256': _sha(manifest_bytes), 'content_sha256': _sha(canonical),
              'samples': audit['samples'], 'split_counts': audit['split_counts'],
              'claims': 'audited corpus relocation only; new manifest requires a new training run'}
    return manifest, payloads, report


def verify_dataset(dataset, episodes, splits, data_root):
    """Audit explicit new locations, preserving every non-location identity field."""
    return _audit(dataset, episodes, splits, data_root)[2]


def materialize_dataset(dataset, episodes, splits, data_root, output, *, expected_content_sha256=None):
    """Create new converted files and a witness after full audit; never overwrite.

    Raw episodes/media remain at their explicit locations and are not copied.
    An I/O failure may leave an incomplete output; it is never silently resumed.
    """
    output = Path(output)
    require(not output.exists(), 'output already exists; never overwrite a dataset')
    manifest, payloads, report = _audit(dataset, episodes, splits, data_root)
    if expected_content_sha256 is not None:
        require(report['content_sha256'] == expected_content_sha256, 'content identity mismatch')
    relocated = {**manifest, 'source_path': str(Path(episodes).resolve()),
                 'splits_path': str(Path(splits).resolve()), 'data_root': str(Path(data_root).resolve())}
    require(any(relocated[k] != manifest[k] for k in LOCATIONS),
            'source locations unchanged; verify only, not a new relocated manifest')
    new_manifest = json.dumps(relocated, indent=2).encode('utf-8')
    require(_sha(new_manifest) != report['manifest_sha256'], 'relocated manifest identity must change')
    witness = {**{k: v for k, v in report.items() if k != 'manifest_sha256'},
               'source_manifest_sha256': report['manifest_sha256'],
               'new_manifest_sha256': _sha(new_manifest)}
    output.mkdir(parents=False, exist_ok=False)
    for name, payload in payloads.items():
        with (output / name).open('xb') as stream:
            stream.write(payload)
    with (output / 'manifest.json').open('xb') as stream:
        stream.write(new_manifest)
    with (output / 'relocation.json').open('x', encoding='utf-8') as stream:
        json.dump(witness, stream, indent=2)
    return witness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--episodes', type=Path, required=True)
    parser.add_argument('--splits', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--materialize', type=Path,
                        help='optional NEW converted directory; not an old-run resume')
    parser.add_argument('--expect-content-sha256', help='optional previously audited content identity')
    args = parser.parse_args()
    if args.materialize is not None:
        report = materialize_dataset(args.dataset, args.episodes, args.splits, args.data_root,
                                     args.materialize, expected_content_sha256=args.expect_content_sha256)
    else:
        report = verify_dataset(args.dataset, args.episodes, args.splits, args.data_root)
        if args.expect_content_sha256 is not None:
            require(report['content_sha256'] == args.expect_content_sha256, 'content identity mismatch')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
