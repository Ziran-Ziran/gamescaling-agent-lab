"""Check actual staged data and policy in isolated repositories."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


CHECKER = Path(__file__).resolve().parents[1] / 'scripts' / 'check_repository.py'


class RepositoryPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git('init', '-q')
        self.policy = {'schema_version': 1, 'project': 'fixture',
                       'allowed_files': ['repository-policy.json', '.github/CODEOWNERS'],
                       'allowed_trees': ['scripts/', '.github/']}
        self.stage_policy()

    def git(self, *args):
        return subprocess.run(['git', *args], cwd=self.root, check=True, capture_output=True)

    def stage(self, name, content):
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        self.git('add', '--', name)
        return target

    def stage_policy(self):
        return self.stage('repository-policy.json', json.dumps(self.policy).encode())

    def check(self, *args):
        return subprocess.run([sys.executable, str(CHECKER), *args], cwd=self.root,
                              capture_output=True, text=True)

    def test_shared_source_and_codeowners_are_allowed(self):
        self.stage('scripts/example.py', b'print(1)\n')
        self.stage('.github/CODEOWNERS', b'* @maintainer\n')
        self.assertEqual(self.check().returncode, 0)

    def test_force_added_data_is_blocked_even_if_policy_allows_directory(self):
        for folder in ('output', 'local'):
            with self.subTest(folder=folder):
                self.policy['allowed_trees'].append(folder + '/')
                self.stage_policy()
                self.stage(folder + '/private.txt', b'local data\n')
                self.assertNotEqual(self.check().returncode, 0)

    def test_staged_secret_is_detected_without_echoing_it(self):
        secret = b'gh' + b'p_' + b'X' * 36
        target = self.stage('scripts/example.py', b'token = "' + secret + b'"\n')
        target.write_bytes(b'print(1)\n')
        result = self.check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('possible GitHub token', result.stderr)
        self.assertNotIn(secret.decode(), result.stdout + result.stderr)

    def test_unstaged_secret_is_not_misreported_as_published(self):
        target = self.stage('scripts/example.py', b'print(1)\n')
        target.write_bytes(b'gh' + b'p_' + b'X' * 36)
        self.assertEqual(self.check().returncode, 0)

    def test_unstaged_policy_cannot_expand_publication(self):
        self.stage('private.py', b'print(1)\n')
        policy = {**self.policy, 'allowed_files': self.policy['allowed_files'] + ['private.py']}
        (self.root / 'repository-policy.json').write_text(json.dumps(policy))
        self.assertNotEqual(self.check().returncode, 0)

    def test_binary_and_oversized_source_are_blocked(self):
        for data in (b'print(1)\0\n', b'#' * (1024 * 1024 + 1)):
            with self.subTest(size=len(data)):
                self.stage('scripts/example.py', data)
                self.assertNotEqual(self.check().returncode, 0)

    def test_missing_staged_policy_fails_closed(self):
        self.git('rm', '--cached', 'repository-policy.json')
        self.assertNotEqual(self.check().returncode, 0)

    def test_invalid_staged_policy_fails_closed(self):
        self.stage('repository-policy.json', b'[]')
        self.assertNotEqual(self.check().returncode, 0)

    def test_staged_only_still_validates_policy(self):
        self.stage('repository-policy.json', b'{broken')
        self.assertNotEqual(self.check('--staged-only').returncode, 0)


if __name__ == '__main__':
    unittest.main()
