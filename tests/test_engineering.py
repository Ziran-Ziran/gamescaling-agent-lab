"""Isolated checks for bundle integrity, safe synchronization and research boundaries."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.sync = module('sync_engineering')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixture(self, folder, version='1.0.0', suffix=b'', members=None):
        target = self.root / folder
        manifest = {'schema_version': 1, 'version': version,
                    'canonical_repository': self.sync.CANONICAL_REPOSITORY, 'files': {}}
        for name in members or self.sync.BUNDLE_FILES:
            file = target / name
            file.parent.mkdir(parents=True, exist_ok=True)
            data = b'# fixture\n' + suffix
            file.write_bytes(data)
            manifest['files'][name] = hashlib.sha256(data).hexdigest()
        file = target / 'engineering/manifest.json'
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(json.dumps(manifest), encoding='utf-8')
        return target

    def run_sync(self, target, source=None, apply=False):
        args = [sys.executable, str(ROOT / 'scripts/sync_engineering.py'), '--root', str(target)]
        if source:
            args += ['--source-root', str(source)]
        if apply:
            args += ['--apply']
        return subprocess.run(args, capture_output=True, text=True)

    def test_local_hash_drift_fails(self):
        target = self.fixture('target')
        (target / 'scripts/check_repository.py').write_text('# changed\n')
        self.assertNotEqual(self.run_sync(target).returncode, 0)

    def test_check_only_does_not_write_and_apply_updates_bundle(self):
        target = self.fixture('target')
        source = self.fixture('source', '1.1.0', b'# new\n')
        original = (target / 'engineering/manifest.json').read_bytes()
        self.assertNotEqual(self.run_sync(target, source).returncode, 0)
        self.assertEqual((target / 'engineering/manifest.json').read_bytes(), original)
        self.assertEqual(self.run_sync(target, source, apply=True).returncode, 0)
        self.assertEqual(self.run_sync(target).returncode, 0)
        self.assertEqual((target / 'engineering/manifest.json').read_bytes(),
                         (source / 'engineering/manifest.json').read_bytes())

    def test_broken_source_never_changes_target(self):
        target = self.fixture('target')
        source = self.fixture('source', '1.1.0', b'# new\n')
        (source / self.sync.BUNDLE_FILES[-1]).write_bytes(b'corrupt')
        before = {p: (target / p).read_bytes() for p in self.sync.BUNDLE_FILES}
        self.assertNotEqual(self.run_sync(target, source, apply=True).returncode, 0)
        self.assertEqual(before, {p: (target / p).read_bytes() for p in before})

    def test_legacy_bundle_can_add_only_the_fixed_new_files(self):
        target = self.fixture('target', members=self.sync.BUNDLE_FILES[:11])
        source = self.fixture('source', '1.1.0', b'# new\n')
        result = self.run_sync(target, source, apply=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.run_sync(target).returncode, 0)

    def test_legacy_upgrade_preserves_preexisting_untracked_new_file(self):
        target = self.fixture('target', members=self.sync.BUNDLE_FILES[:11])
        source = self.fixture('source', '1.1.0', b'# new\n')
        existing = target / 'scripts/local_scene.py'
        existing.write_bytes(b'local untracked code')
        self.assertNotEqual(self.run_sync(target, source, apply=True).returncode, 0)
        self.assertEqual(existing.read_bytes(), b'local untracked code')

    def test_local_edits_are_not_overwritten(self):
        target = self.fixture('target')
        source = self.fixture('source', '1.1.0', b'# new\n')
        file = target / self.sync.BUNDLE_FILES[0]
        file.write_bytes(b'local edit')
        self.assertNotEqual(self.run_sync(target, source, apply=True).returncode, 0)
        self.assertEqual(file.read_bytes(), b'local edit')

    def test_arbitrary_manifest_paths_are_rejected(self):
        target = self.fixture('target')
        file = target / 'engineering/manifest.json'
        manifest = json.loads(file.read_text())
        manifest['files']['../escape.py'] = '0' * 64
        file.write_text(json.dumps(manifest))
        self.assertNotEqual(self.run_sync(target).returncode, 0)

    def test_linked_source_file_is_rejected(self):
        target = self.fixture('target')
        file = target / self.sync.BUNDLE_FILES[0]
        outside = self.root / 'outside'
        outside.write_bytes(file.read_bytes())
        file.unlink()
        try:
            file.symlink_to(outside)
        except OSError:
            self.skipTest('symlink creation unavailable on this host')
        self.assertNotEqual(self.run_sync(target).returncode, 0)


class ExperimentContractTests(unittest.TestCase):
    def setUp(self):
        self.validate = module('validate_project')
        self.config = {
            'schema_version': 1, 'project': 'gamescaling', 'name': 'fixture',
            'status': 'design', 'runner_implemented': False, 'dataset_bound': False,
            'seed': 0, 'stage': 'sft', 'observations': ['rgb'],
            'training_only_observations': [], 'evaluation_observations': ['rgb'],
            'input_policy': 'player_observation', 'split_unit': 'game',
            'split_ids': {'train': ['a'], 'validation': ['b'], 'test': ['c']},
            'metrics': ['task_success_rate'],
        }

    def errors(self, config=None):
        return self.validate.validate_experiment(config or self.config, 'gamescaling')

    def test_independent_design_configuration_is_valid(self):
        self.assertEqual(self.errors(), [])

    def test_game_overlap_is_rejected(self):
        self.config['split_ids']['test'] = ['a']
        self.assertTrue(self.errors())

    def test_privileged_evaluation_cannot_claim_player_inputs(self):
        self.config['observations'].append('engine_state')
        self.config['evaluation_observations'].append('engine_state')
        self.assertTrue(self.errors())

    def test_training_only_information_never_enters_evaluation(self):
        self.config['training_only_observations'] = ['rgb']
        self.assertTrue(self.errors())

    def test_unknown_modalities_and_frame_split_are_rejected(self):
        self.config['observations'] = ['unknown']
        self.config['split_unit'] = 'frame'
        self.assertTrue(self.errors())

    def test_bad_json_types_report_errors_without_traceback(self):
        for field, value in [('split_ids', []), ('observations', None), ('seed', True),
                             ('metrics', 'success'), ('project', 'sitp')]:
            with self.subTest(field=field):
                config = copy.deepcopy(self.config)
                config[field] = value
                self.assertTrue(self.errors(config))

    def test_design_validator_does_not_claim_existing_runner(self):
        self.config['runner_implemented'] = True
        self.assertTrue(self.errors())

    def test_unconfigured_sitp_privileged_inputs_are_rejected(self):
        self.config.update(project='sitp', stage='semantic_navigation', split_unit='scene',
                           input_policy='sensor_only', observations=['rgb', 'engine_state'])
        self.assertTrue(self.validate.validate_experiment(self.config, 'sitp'))


if __name__ == '__main__':
    unittest.main()
