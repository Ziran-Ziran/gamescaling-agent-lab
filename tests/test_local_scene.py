"""Local-only synthetic fixtures; never include extracted game assets in tests."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/local_scene.py'
SCENE = 'mondstadt-city-7-1-0'
MAP = 'UE5/Content/MondstadtCity/Mondstadt_City_Detailed.umap'


class LocalSceneTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('local_scene', SCRIPT)
        self.scene = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.scene)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.catalog = self.root / 'shared-library/scenes.local.json'
        self.files = {
            'UE5/MondstadtCityDetailed.uproject': json.dumps({'FileVersion': 3, 'EngineAssociation': '5.8'}).encode(),
            MAP: b'synthetic map fixture',
            'UE5/Content/MondstadtCity/Mesh.uasset': b'synthetic mesh fixture',
            'UE5/Config/DefaultInput.ini': b'[/Script/Engine.InputSettings]\n',
        }
        for name, data in self.files.items():
            self.write(self.source / name, data)
        self.write(self.source / 'UE5/Config/DefaultEngine.ini', b'SecurityToken=fixture-do-not-copy\n')
        self.write(self.source / 'UE5/Saved/private.log', b'private cache')
        self.write(self.source / 'package_summary.json', json.dumps({
            'default_static_mesh_actors': 1, 'default_decal_actors': 0,
            'physics_navigation_play_tested': False,
        }).encode())
        self.write(self.source / 'city_ue_saved_validation.json', b'{"status":"success"}')
        self.checksums()

    def write(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def checksums(self, extra=None):
        values = [{'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                  for name, data in self.files.items()]
        if extra:
            values.append(extra)
        self.write(self.source / 'delivery_checksums.json', json.dumps(values).encode())

    def register(self):
        return self.scene.register_scene(self.catalog, self.source, SCENE)

    def prepare(self, repo):
        return self.scene.prepare_scene(repo, self.catalog, SCENE)

    def test_two_repositories_receive_independent_copies_of_shared_source(self):
        self.register()
        first = self.prepare(self.root / 'sitp')
        second = self.prepare(self.root / 'player')
        a = Path(first['workspace']) / MAP
        b = Path(second['workspace']) / MAP
        self.assertEqual(a.read_bytes(), b.read_bytes())
        a.write_bytes(b'local edit')
        self.assertEqual(b.read_bytes(), self.files[MAP])
        self.assertEqual((self.source / MAP).read_bytes(), self.files[MAP])

    def test_prepare_excludes_source_caches_and_generates_clean_map_config(self):
        self.register()
        binding = self.prepare(self.root / 'sitp')
        workspace = Path(binding['workspace'])
        self.assertFalse((workspace / 'UE5/Saved').exists())
        config = (workspace / 'UE5/Config/DefaultEngine.ini').read_text()
        self.assertNotIn('SecurityToken', config)
        self.assertIn('/Game/MondstadtCity/Mondstadt_City_Detailed', config)
        self.assertFalse(binding['physics_navigation_play_tested'])

    def test_repeated_prepare_preserves_local_edits(self):
        self.register()
        repo = self.root / 'sitp'
        binding = self.prepare(repo)
        target = Path(binding['workspace']) / MAP
        target.write_bytes(b'my experiment')
        self.prepare(repo)
        self.assertEqual(target.read_bytes(), b'my experiment')
        report = self.scene.inspect_scene(repo, SCENE)
        self.assertEqual(report['workspace_modified_files'], [MAP])

    def test_changed_source_fails_before_creating_workspace(self):
        self.register()
        (self.source / MAP).write_bytes(b'damaged')
        repo = self.root / 'sitp'
        with self.assertRaises(ValueError):
            self.prepare(repo)
        self.assertFalse((repo / 'local/scenes' / SCENE).exists())

    def test_manifest_path_traversal_is_rejected(self):
        self.checksums({'path': '../escape', 'bytes': 1, 'sha256': '0' * 64})
        with self.assertRaises(ValueError):
            self.register()

    def test_windows_delivery_paths_are_normalized_before_validation(self):
        path = self.source / 'delivery_checksums.json'
        entries = json.loads(path.read_text())
        for entry in entries:
            entry['path'] = entry['path'].replace('/', '\\')
        path.write_text(json.dumps(entries))
        self.register()
        binding = self.prepare(self.root / 'sitp')
        self.assertEqual((Path(binding['workspace']) / MAP).read_bytes(), self.files[MAP])

    def test_unlisted_assets_are_not_silently_accepted(self):
        self.write(self.source / 'UE5/Content/MondstadtCity/Unknown.uasset', b'unknown')
        with self.assertRaises(ValueError):
            self.register()

    def test_changed_checksum_manifest_requires_explicit_new_registration(self):
        self.register()
        file = self.source / 'delivery_checksums.json'
        file.write_bytes(file.read_bytes() + b'\n')
        with self.assertRaises(ValueError):
            self.prepare(self.root / 'sitp')

    def test_unsafe_scene_id_is_rejected(self):
        for name in ('../escape', 'A/B', '', 'CON'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.scene.register_scene(self.catalog, self.source, name)

    def test_register_is_idempotent_but_never_rebinds_existing_id(self):
        first = self.register()
        self.assertEqual(self.register(), first)
        import shutil
        other = self.root / 'different-source'
        shutil.copytree(self.source, other)
        with self.assertRaises(ValueError):
            self.scene.register_scene(self.catalog, other, SCENE)

    def test_inspect_reports_missing_files_without_repairing_them(self):
        self.register()
        repo = self.root / 'sitp'
        binding = self.prepare(repo)
        target = Path(binding['workspace']) / MAP
        target.unlink()
        result = self.scene.inspect_scene(repo, SCENE)
        self.assertEqual(result['workspace_missing_files'], [MAP])
        self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main()
