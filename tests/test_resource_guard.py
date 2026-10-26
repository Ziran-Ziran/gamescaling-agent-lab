import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from scripts.resource_guard import HeavyTaskLock, run_bounded


class ResourceGuardTests(unittest.TestCase):
    def test_busy_lock_is_never_deleted_even_with_old_or_invalid_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            lock = Path(folder) / 'gpu.lock'
            lock.write_text('an old lock from another chat', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                with HeavyTaskLock(lock):
                    self.fail('busy lock acquired')
            self.assertEqual(lock.read_text(encoding='utf-8'), 'an old lock from another chat')

    def test_owner_metadata_and_release_on_exception(self):
        with tempfile.TemporaryDirectory() as folder:
            lock = Path(folder) / 'gpu.lock'
            with self.assertRaisesRegex(RuntimeError, 'failure'):
                with HeavyTaskLock(lock):
                    self.assertIn('token', json.loads(lock.read_text(encoding='utf-8')))
                    raise RuntimeError('failure')
            self.assertFalse(lock.exists())

    def test_replaced_lock_is_not_removed(self):
        with tempfile.TemporaryDirectory() as folder:
            lock = Path(folder) / 'gpu.lock'
            with HeavyTaskLock(lock):
                lock.unlink()
                lock.write_text('replacement', encoding='utf-8')
            self.assertEqual(lock.read_text(encoding='utf-8'), 'replacement')

    def test_child_observes_lock_until_it_exits(self):
        with tempfile.TemporaryDirectory() as folder:
            lock = Path(folder) / 'gpu.lock'
            code = 'import pathlib,sys; sys.exit(0 if pathlib.Path(sys.argv[1]).exists() else 7)'
            result = run_bounded([sys.executable, '-c', code, str(lock)], lock, 5)
            self.assertEqual(result, 0)
            self.assertFalse(lock.exists())

    def test_timeout_releases_only_after_child_stops(self):
        with tempfile.TemporaryDirectory() as folder:
            lock = Path(folder) / 'gpu.lock'
            with self.assertRaises(subprocess.TimeoutExpired):
                run_bounded([sys.executable, '-c', 'import time; time.sleep(30)'], lock, .15)
            self.assertFalse(lock.exists())

    def test_rejects_unbounded_or_excessive_duration(self):
        for seconds in (0, -1, 1201, float('inf'), float('nan'), True):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                run_bounded([sys.executable, '-c', 'pass'], Path('unused.lock'), seconds)

    def test_root_exit_does_not_leave_descendant_alive_after_unlock(self):
        with tempfile.TemporaryDirectory() as folder:
            marker = Path(folder) / 'escaped.txt'
            child = 'import pathlib,time,sys; time.sleep(.5); pathlib.Path(sys.argv[1]).write_text("alive")'
            root = 'import subprocess,sys; subprocess.Popen([sys.executable,"-c",sys.argv[1],sys.argv[2]])'
            run_bounded([sys.executable, '-c', root, child, str(marker)],
                        Path(folder) / 'gpu.lock', 5)
            time.sleep(.7)
            self.assertFalse(marker.exists(), 'descendant survived successful parent exit')


if __name__ == '__main__':
    unittest.main()
