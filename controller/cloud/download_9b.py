"""Download the pinned official checkpoint into the data-disk HF cache."""
import hashlib
import json
import os
from pathlib import Path
import re
import time

ROOT = Path('/root/autodl-tmp/gamescaling-lab')
REPO = 'THUSI-Lab/GameScaling-9B-S3'


def main():
    from huggingface_hub import HfApi, snapshot_download
    ROOT.mkdir(parents=True, exist_ok=True)
    target = ROOT / 'download-manifest.json'
    revision = None
    if target.exists():
        previous = json.loads(target.read_text(encoding='utf-8'))
        if previous.get('repo') != REPO or not re.fullmatch(r'[0-9a-f]{40}', previous.get('revision', '')):
            raise RuntimeError('Existing manifest has a different repository or invalid commit')
        revision = previous['revision']
    info = HfApi().model_info(REPO, revision=revision, files_metadata=True)
    files = []
    for entry in info.siblings:
        lfs = getattr(entry, 'lfs', None)
        sha = getattr(lfs, 'sha256', None)
        if isinstance(lfs, dict):
            sha = lfs.get('sha256', sha)
        files.append(dict(name=entry.rfilename, size=entry.size, sha256=sha))
    manifest = dict(repo=REPO, revision=info.sha, files=files,
                    endpoint=os.environ.get('HF_ENDPOINT', 'https://huggingface.co'),
                    started=time.time(), status='downloading')
    target.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'repo': REPO, 'revision': info.sha,
                      'total_file_bytes': sum(f['size'] or 0 for f in files)}), flush=True)
    snapshot = Path(snapshot_download(REPO, revision=info.sha,
        cache_dir=str(ROOT / 'cache/huggingface/hub'), max_workers=4))
    verified = []
    for entry in files:
        path = snapshot / entry['name']
        if not path.is_file():
            raise RuntimeError(f"Missing downloaded file: {entry['name']}")
        if entry['size'] is not None and path.stat().st_size != entry['size']:
            raise RuntimeError(f"File size mismatch: {entry['name']}")
        if entry['sha256']:
            digest = hashlib.sha256()
            with path.open('rb') as source:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != entry['sha256']:
                raise RuntimeError(f"SHA256 mismatch: {entry['name']}")
            verified.append(entry['name'])
    manifest.update(status='verified', snapshot=str(snapshot),
                    sha256_verified=verified, completed=time.time())
    target.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'verified', 'snapshot': str(snapshot),
                      'sha256_verified': verified}), flush=True)


if __name__ == '__main__':
    main()
