import ctypes
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

try:
    from controller import automatic as a
except ImportError:
    a = None


class AutomaticTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(a, 'automatic launcher is not implemented yet')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / ('a' * 32)
        self.root.mkdir()
        (self.root / 'request').mkdir()
        self.process = Mock()
        self.process.exited.return_value = False
        self.process.returncode.return_value = 0
        self.quiet = patch('sys.stdout')
        self.quiet.start()
        self.addCleanup(self.quiet.stop)
        self.quiet_errors = patch('sys.stderr')
        self.quiet_errors.start()
        self.addCleanup(self.quiet_errors.stop)
        self.patches = [patch.object(a.b, 'prepare_session', return_value=self.root),
                        patch.object(a, 'SHADOW_OUTPUT', Path(self.tmp.name) / 'shadow'),
                        patch.object(a, 'RESULT_TIMEOUT', .015),
                        patch.object(a, 'CLEANUP_TIMEOUT', .015),
                        patch.object(a, 'POLL_SECONDS', .001)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def result(self, status='passed', code=0):
        directory = self.root / 'results' / 'req0001'
        directory.mkdir(parents=True)
        a.b.atomic_json(directory / 'result.json', dict(status=status, returncode=code))

    def run_round(self, argv):
        try:
            return a.main(argv)
        except SystemExit as exc:
            self.fail(f'Supported automatic invocation was rejected: {argv}, exit={exc.code}')

    def test_default_shadow_calls_existing_loop_without_elevation_or_input(self):
        with patch.object(a.run_agent, 'main', return_value=0) as run, \
                patch.object(a, 'launch_worker') as launch, patch.object(a.b, 'prepare_session') as prepare:
            self.assertEqual(a.main([]), 0)
        argv = run.call_args.args[0]
        self.assertEqual(argv[0], 'shadow')
        for key, value in [('--backend', 'cloud'), ('--seconds', '10'), ('--count', '10'), ('--countdown', '0')]:
            self.assertEqual(argv[argv.index(key) + 1], value)
        self.assertIn('--activate', argv)
        self.assertNotIn('--execute', argv)
        launch.assert_not_called()
        prepare.assert_not_called()

    def test_input_requires_execute_before_creating_session(self):
        with patch.object(a.b, 'prepare_session') as prepare:
            for mode in ('explore', 'calibrate'):
                with self.assertRaises(SystemExit) as error:
                    a.main(['--mode', mode])
                self.assertEqual(error.exception.code, 2)
            prepare.assert_not_called()

    def test_elevated_shadow_uses_one_shadow_request_without_execute(self):
        def launch(root, backend):
            request = json.loads((root / 'request' / 'req0001.json').read_text())
            self.assertEqual(request, dict(mode='shadow', seconds=10, count=10))
            self.assertEqual(backend, 'cloud')
            self.assertEqual(len(list((root / 'request').iterdir())), 1)
            self.result()
            return self.process
        self.process.exited.side_effect = lambda: (self.root / 'STOP').exists()
        with patch.object(a, 'launch_worker', side_effect=launch), \
                patch.object(a.run_agent, 'main') as direct_loop:
            self.assertEqual(self.run_round(['--mode', 'shadow', '--elevated']), 0)
        direct_loop.assert_not_called()
        state = json.loads((self.root / 'automatic.json').read_text())
        self.assertTrue(state['elevated_shadow'])
        self.assertTrue(state['worker_closed'])
        self.assertTrue((self.root / 'STOP').exists())
        self.process.close.assert_called_once()

    def test_elevated_flag_rejected_for_input_modes_before_session_preparation(self):
        with patch.object(a.b, 'prepare_session') as prepare:
            for mode in ('explore', 'calibrate'):
                with self.assertRaises(SystemExit) as error:
                    a.main(['--mode', mode, '--execute', '--elevated'])
                self.assertEqual(error.exception.code, 2)
            prepare.assert_not_called()

    def test_elevated_shadow_cancel_and_timeout_write_stop(self):
        for failure in [OSError(1223, 'UAC cancelled'), TimeoutError('UAC timeout')]:
            with self.subTest(failure=type(failure).__name__):
                (self.root / 'STOP').unlink(missing_ok=True)
                (self.root / 'automatic.json').unlink(missing_ok=True)
                (self.root / 'request' / 'req0001.json').unlink(missing_ok=True)
                with patch.object(a, 'launch_worker', side_effect=failure):
                    self.assertNotEqual(self.run_round(['--elevated']), 0)
                self.assertTrue((self.root / 'STOP').exists())
                state = json.loads((self.root / 'automatic.json').read_text())
                self.assertTrue(state['elevated_shadow'])

    def test_invalid_bounds_and_injection_flags_rejected_before_launch(self):
        with patch.object(a, 'launch_worker') as launch:
            for argv in [['--seconds', '11'], ['--seconds', 'nan'], ['--seconds', '0'],
                         ['--count', '31'], ['--count', '0'], ['--backend', 'evil'],
                         ['--output', 'arbitrary'], ['--command', 'whoami'], ['--url', 'https://evil']]:
                with self.subTest(argv=argv), self.assertRaises(SystemExit):
                    a.main(argv)
            launch.assert_not_called()

    def test_input_prepares_only_one_bounded_job_then_stops_and_closes(self):
        def launch(root, backend):
            request = json.loads((root / 'request' / 'req0001.json').read_text())
            self.assertEqual(request, dict(mode='explore', seconds=10, count=10))
            self.assertEqual(backend, 'cloud')
            self.assertEqual(len(list((root / 'request').iterdir())), 1)
            self.result()
            return self.process
        def exited():
            return (self.root / 'STOP').exists()
        self.process.exited.side_effect = exited
        with patch.object(a, 'launch_worker', side_effect=launch):
            self.assertEqual(a.main(['--mode', 'explore', '--execute']), 0)
        self.assertTrue((self.root / 'STOP').exists())
        self.process.close.assert_called_once()
        state = json.loads((self.root / 'automatic.json').read_text())
        self.assertEqual(state['status'], 'passed')
        self.assertTrue(state['worker_closed'])

    def test_cancel_timeout_exception_and_interrupt_write_stop(self):
        for failure in [OSError(1223, 'UAC cancelled'), TimeoutError('UAC timeout'),
                        RuntimeError('launch failed'), KeyboardInterrupt()]:
            with self.subTest(failure=type(failure).__name__):
                (self.root / 'STOP').unlink(missing_ok=True)
                (self.root / 'automatic.json').unlink(missing_ok=True)
                (self.root / 'request' / 'req0001.json').unlink(missing_ok=True)
                with patch.object(a, 'launch_worker', side_effect=failure):
                    self.assertNotEqual(a.main(['--mode', 'explore', '--execute']), 0)
                self.assertTrue((self.root / 'STOP').exists())

    def test_result_timeout_stops_worker_without_termination_and_closes_handle(self):
        with patch.object(a, 'launch_worker', return_value=self.process):
            self.assertNotEqual(a.main(['--mode', 'calibrate', '--execute']), 0)
        self.assertTrue((self.root / 'STOP').exists())
        self.process.close.assert_called_once()
        self.process.terminate.assert_not_called()

    def test_failed_result_or_early_worker_exit_is_nonzero(self):
        self.result(status='stopped', code=2)
        self.process.exited.return_value = True
        with patch.object(a, 'launch_worker', return_value=self.process):
            self.assertNotEqual(a.main(['--mode', 'explore', '--execute']), 0)

    def test_worker_exit_without_result_is_failure(self):
        self.process.exited.return_value = True
        with patch.object(a, 'launch_worker', return_value=self.process):
            self.assertNotEqual(a.main(['--mode', 'explore', '--execute']), 0)
        self.assertTrue((self.root / 'STOP').exists())

    def test_successful_result_but_worker_does_not_close_is_failure(self):
        self.result()
        with patch.object(a, 'launch_worker', return_value=self.process):
            self.assertNotEqual(a.main(['--mode', 'explore', '--execute']), 0)
        self.process.close.assert_called_once()

    def test_successful_result_but_nonzero_worker_exit_is_failure(self):
        self.result()
        self.process.exited.return_value = True
        self.process.returncode.return_value = 1
        with patch.object(a, 'launch_worker', return_value=self.process):
            self.assertNotEqual(a.main(['--mode', 'explore', '--execute']), 0)

    def test_stop_write_failure_still_closes_worker_handle(self):
        with patch.object(Path, 'touch', side_effect=OSError('disk failed')):
            with self.assertRaises(OSError):
                a.stop_worker(self.root, self.process)
        self.process.close.assert_called_once()

    def test_interrupt_during_result_wait_stops_and_closes_worker(self):
        self.process.exited.side_effect = lambda: (self.root / 'STOP').exists()
        with patch.object(a, 'launch_worker', return_value=self.process), \
                patch.object(a, 'wait_result', side_effect=KeyboardInterrupt):
            self.assertEqual(a.main(['--mode', 'explore', '--execute']), 130)
        self.assertTrue((self.root / 'STOP').exists())
        self.process.close.assert_called_once()

    def test_late_uac_handle_is_closed_after_parent_timeout_and_stop(self):
        release, finished = threading.Event(), threading.Event()
        def native(root, backend):
            release.wait(1)
            self.assertTrue((root / 'STOP').exists())
            return self.process
        self.process.close.side_effect = finished.set
        with patch.object(a, '_shell_execute_worker', side_effect=native), patch.object(a, 'UAC_TIMEOUT', .01):
            try:
                self.assertNotEqual(a.main(['--mode', 'explore', '--execute']), 0)
            finally:
                release.set()
            self.assertTrue(finished.wait(1))

    def test_stop_write_failure_during_uac_timeout_still_closes_late_handle(self):
        release, finished = threading.Event(), threading.Event()
        def native(root, backend):
            release.wait(1)
            return self.process
        self.process.close.side_effect = finished.set
        with patch.object(a, '_shell_execute_worker', side_effect=native), \
                patch.object(a, 'UAC_TIMEOUT', .01), \
                patch.object(Path, 'touch', side_effect=OSError('STOP disk failure')):
            try:
                with self.assertRaisesRegex(OSError, 'STOP disk failure'):
                    a.launch_worker(self.root, 'cloud')
            finally:
                release.set()
            self.assertTrue(finished.wait(.25), 'Abandoned late handle must close even if STOP cannot be written')
        self.assertFalse((self.root / 'STOP').exists())
        self.process.close.assert_called_once()

    def test_native_launcher_uses_fixed_python_runas_hidden_and_fixed_cwd(self):
        kernel, shell, ole = Mock(), Mock(), Mock()
        ole.CoInitializeEx.return_value = 0
        def execute(pointer):
            info = pointer._obj
            self.assertEqual(info.lpVerb, 'runas')
            self.assertEqual(info.lpFile, a.sys.executable)
            self.assertEqual(info.lpDirectory, str(a.b.ROOT))
            self.assertEqual(info.nShow, 0)
            self.assertEqual(info.fMask & 0x40, 0x40)
            self.assertEqual(info.lpParameters, '-m controller.bounded_session worker --session ' + self.root.name + ' --backend cloud --activate')
            info.hProcess = 123
            return True
        shell.ShellExecuteExW.side_effect = execute
        with patch.object(a.ctypes, 'WinDLL', side_effect=[shell, kernel, ole]):
            process = a._shell_execute_worker(self.root, 'cloud')
            kernel.WaitForSingleObject.return_value = 0
            self.assertTrue(process.exited())
            process.close()
            process.close()
        kernel.CloseHandle.assert_called_once_with(123)
        ole.CoInitializeEx.assert_called_once_with(None, 6)
        ole.CoUninitialize.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
