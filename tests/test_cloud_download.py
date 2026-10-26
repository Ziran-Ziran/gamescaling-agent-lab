import contextlib
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from controller.cloud import download_9b


class CloudDownloadTests(unittest.TestCase):
    def run_download(self, root, *, corrupt=False):
        snapshot = root / 'snapshot'
        snapshot.mkdir()
        (snapshot / 'model.safetensors').write_bytes(b'bad!' if corrupt else b'good')
        calls = []

        def model_info(repo, *, revision=None, files_metadata=False):
            calls.append(revision)
            return SimpleNamespace(sha=revision or 'b' * 40, siblings=[SimpleNamespace(
                rfilename='model.safetensors', size=4,
                lfs={'sha256': hashlib.sha256(b'good').hexdigest()})])

        client = SimpleNamespace(HfApi=lambda: SimpleNamespace(model_info=model_info),
                                 snapshot_download=lambda *a, **kw: str(snapshot))
        with patch.object(download_9b, 'ROOT', root), \
                patch.dict('sys.modules', {'huggingface_hub': client}), \
                contextlib.redirect_stdout(io.StringIO()):
            download_9b.main()
        return calls

    def test_resume_keeps_original_commit_when_main_has_changed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = root / 'download-manifest.json'
            manifest.write_text(json.dumps({'repo': download_9b.REPO,
                'revision': 'a' * 40, 'status': 'downloading'}))
            calls = self.run_download(root)
            result = json.loads(manifest.read_text())
            self.assertEqual(calls, ['a' * 40])
            self.assertEqual(result['revision'], 'a' * 40)
            self.assertEqual(result['status'], 'verified')

    def test_same_size_corrupt_weight_never_gets_verified(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(RuntimeError, 'SHA256 mismatch'):
                self.run_download(root, corrupt=True)
            manifest = json.loads((root / 'download-manifest.json').read_text())
            self.assertEqual(manifest['status'], 'downloading')


if __name__ == '__main__':
    unittest.main()
