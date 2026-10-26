"""CPU-only relocation checks; no model, GPU, downloads or real-game inputs."""
import copy
import os
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from scripts.training.data import convert, sha256_file
from scripts.training.hf_backend import load_dataset
from tests.test_training_data import fixture


class DatasetRelocationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.original = self.root / 'original'
        self.original.mkdir()
        episodes, splits = fixture(self.original)
        self.source = self.original / 'episodes.jsonl'
        self.source.write_text(''.join(json.dumps(row) + '\n' for row in episodes), encoding='utf-8')
        self.splits = self.original / 'splits.json'
        self.splits.write_text(json.dumps(splits), encoding='utf-8')
        self.dataset = self.original / 'converted'
        convert(self.source, self.splits, self.original, self.dataset, history_len=2)
        self.moved = self.root / 'moved'
        shutil.copytree(self.original, self.moved)

    def verify(self, root=None):
        from scripts.training.relocate import verify_dataset
        root = root or self.moved
        return verify_dataset(root / 'converted', root / 'episodes.jsonl',
                              root / 'splits.json', root)

    def test_location_changes_preserve_audited_content_identity_without_rewriting_old_manifest(self):
        self.assertIsNotNone(__import__('importlib.util').util.find_spec('scripts.training.relocate'),
                             'relocation verifier is not implemented')
        before = (self.dataset / 'manifest.json').read_bytes()
        first, second = self.verify(self.original), self.verify()
        self.assertEqual(first['content_sha256'], second['content_sha256'])
        self.assertEqual(second['samples'], 9)
        self.assertEqual((self.dataset / 'manifest.json').read_bytes(), before)
        self.assertNotIn(str(self.root), json.dumps(second))

    def test_materialization_is_new_identity_and_loadable_after_original_is_unavailable(self):
        from scripts.training.relocate import materialize_dataset
        old_manifest_sha = sha256_file(self.dataset / 'manifest.json')
        old_fingerprint = self.verify()['content_sha256']
        unavailable = self.root / 'original-unavailable'
        self.original.rename(unavailable)
        output = self.root / 'new-dataset'
        witness = materialize_dataset(self.moved / 'converted', self.moved / 'episodes.jsonl',
                                      self.moved / 'splits.json', self.moved, output)
        rows, manifest, new_manifest_sha = load_dataset(output)
        self.assertEqual(sum(map(len, rows.values())), 9)
        self.assertEqual(witness['source_manifest_sha256'], old_manifest_sha)
        self.assertEqual(witness['new_manifest_sha256'], new_manifest_sha)
        self.assertNotEqual(new_manifest_sha, old_manifest_sha)
        self.assertEqual(witness['content_sha256'], old_fingerprint)
        self.assertEqual(manifest['data_root'], str(self.moved.resolve()))
        self.assertEqual(sha256_file(unavailable / 'converted' / 'manifest.json'), old_manifest_sha)

    def test_source_split_or_rgb_mutation_is_rejected(self):
        for name in ('episodes.jsonl', 'splits.json', 'train-game-0.png'):
            with self.subTest(name=name):
                path = self.moved / name
                original = path.read_bytes()
                path.write_bytes(original + b' ')
                with self.assertRaisesRegex(ValueError, 'hash|identity'):
                    self.verify()
                path.write_bytes(original)

    def test_rehashed_label_tampering_cannot_pass_source_reaudit(self):
        path = self.moved / 'converted' / 'train.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]['action'] = rows[0]['action'].replace('W', 'S')
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        manifest_path = path.parent / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['files']['train']['sha256'] = sha256_file(path)
        manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'rows|source'):
            self.verify()

    def test_manifest_metadata_and_unknown_fields_are_not_silently_dropped(self):
        path = self.moved / 'converted' / 'manifest.json'
        original = json.loads(path.read_text())
        mutations = ({'samples': 999}, {'history_len': 0}, {'schema_version': 2},
                     {'extra_ignored_training_flag': True}, {'excluded_incomplete': 1})
        for change in mutations:
            with self.subTest(change=change):
                path.write_text(json.dumps({**copy.deepcopy(original), **change}), encoding='utf-8')
                with self.assertRaises(ValueError):
                    self.verify()

    def test_split_paths_cannot_escape_and_failed_audit_does_not_materialize(self):
        from scripts.training.relocate import materialize_dataset
        path = self.moved / 'converted' / 'manifest.json'
        manifest = json.loads(path.read_text())
        manifest['files']['train']['path'] = '../outside.jsonl'
        path.write_text(json.dumps(manifest), encoding='utf-8')
        output = self.root / 'rejected'
        with self.assertRaises(ValueError):
            materialize_dataset(path.parent, self.moved / 'episodes.jsonl',
                                self.moved / 'splits.json', self.moved, output)
        self.assertFalse(output.exists())

    def test_existing_output_is_never_overwritten(self):
        from scripts.training.relocate import materialize_dataset
        output = self.root / 'existing'
        output.mkdir()
        marker = output / 'keep.txt'
        marker.write_text('untouched')
        with self.assertRaises((FileExistsError, ValueError)):
            materialize_dataset(self.moved / 'converted', self.moved / 'episodes.jsonl',
                                self.moved / 'splits.json', self.moved, output)
        self.assertEqual(marker.read_text(), 'untouched')

    def test_fresh_conversion_with_different_history_has_different_content_identity(self):
        from scripts.training.relocate import verify_dataset
        alternate = self.root / 'alternate'
        convert(self.source, self.splits, self.original, alternate, history_len=0)
        other = verify_dataset(alternate, self.source, self.splits, self.original)
        self.assertNotEqual(other['content_sha256'], self.verify()['content_sha256'])

    def test_expected_content_mismatch_does_not_create_any_output(self):
        from scripts.training.relocate import materialize_dataset
        output = self.root / 'wrong-identity'
        with self.assertRaisesRegex(ValueError, 'content identity'):
            materialize_dataset(self.moved / 'converted', self.moved / 'episodes.jsonl',
                                self.moved / 'splits.json', self.moved, output,
                                expected_content_sha256='0' * 64)
        self.assertFalse(output.exists())

    def test_converted_file_symlink_outside_dataset_is_rejected(self):
        path = self.moved / 'converted' / 'train.jsonl'
        outside = self.root / 'outside.jsonl'
        path.rename(outside)
        try:
            os.symlink(outside, path)
        except OSError as exc:
            self.skipTest(f'current platform cannot create test symlink: {exc}')
        with self.assertRaisesRegex(ValueError, 'escapes'):
            self.verify()

    def test_duplicate_json_fields_are_rejected(self):
        path = self.moved / 'converted' / 'manifest.json'
        text = path.read_text()
        path.write_text(text.replace('"samples": 9', '"samples": 9, "samples": 9'), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.verify()

    def test_numeric_and_boolean_metadata_are_not_interchangeable(self):
        path = self.moved / 'converted' / 'manifest.json'
        original = json.loads(path.read_text())
        for change in ({'schema_version': True}, {'excluded_incomplete': False}, {'samples': 9.0}):
            with self.subTest(change=change):
                path.write_text(json.dumps({**original, **change}), encoding='utf-8')
                with self.assertRaises(ValueError):
                    self.verify()

    def test_rehashed_boolean_frame_id_cannot_alias_integer_source_frame_id(self):
        path = self.moved / 'converted' / 'train.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]['frame_id'] = False
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        manifest_path = path.parent / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['files']['train']['sha256'] = sha256_file(path)
        manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'rows'):
            self.verify()

    def test_unchanged_source_locations_cannot_be_relabelled_as_new_manifest(self):
        from scripts.training.relocate import materialize_dataset
        output = self.root / 'same-location-copy'
        with self.assertRaisesRegex(ValueError, 'location'):
            materialize_dataset(self.dataset, self.source, self.splits, self.original, output)
        self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
