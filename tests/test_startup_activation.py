import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from controller import bounded_session as b, run_agent as r, windows_adapter as wa


class ActivationTests(unittest.TestCase):
    def setUp(self):
        self.info = dict(hwnd=91, pid=701, title='原神', minimized=True)
        self.user = MagicMock()
        self.user.IsWindow.return_value = True
        self.user.IsIconic.side_effect = [True, False, False]
        self.user.GetForegroundWindow.return_value = 91
        self.user.GetWindowThreadProcessId.side_effect = lambda hwnd, pid: setattr(pid._obj, 'value', 701)

    def window(self):
        with patch.object(wa, 'list_game_windows', return_value=[self.info.copy()]):
            return wa.GameWindow()

    def test_restore_and_verify_identity_and_foreground(self):
        window = self.window()
        with patch.object(wa, 'USER', self.user), patch.object(wa, 'list_game_windows', return_value=[self.info]):
            self.assertTrue(callable(getattr(window, 'activate', None)), 'startup activation is required')
            window.activate()
        self.user.ShowWindowAsync.assert_called_once_with(91, 9)
        self.user.ShowWindow.assert_not_called()
        self.user.SetForegroundWindow.assert_called_once_with(91)
        self.user.GetForegroundWindow.assert_called()
        self.assertGreaterEqual(self.user.GetWindowThreadProcessId.call_count, 2)
        self.assertFalse(window.info['minimized'], 'restored window metadata must not retain startup state')

    def test_wrong_reused_or_ambiguous_window_never_activated(self):
        window = self.window()
        candidates = [[], [self.info, self.info], [self.info | {'pid': 702}],
                      [self.info | {'title': 'Other'}], [self.info | {'hwnd': 92}]]
        for matches in candidates:
            with self.subTest(matches=matches), patch.object(wa, 'USER', self.user), \
                    patch.object(wa, 'list_game_windows', return_value=matches):
                self.assertTrue(callable(getattr(window, 'activate', None)), 'startup activation is required')
                with self.assertRaises(RuntimeError):
                    window.activate()
        self.user.ShowWindowAsync.assert_not_called()
        self.user.SetForegroundWindow.assert_not_called()

    def test_pid_changes_after_restore_aborts(self):
        window = self.window()
        def pid(hwnd, pointer):
            pointer._obj.value = 702 if self.user.ShowWindowAsync.called else 701
        self.user.GetWindowThreadProcessId.side_effect = pid
        with patch.object(wa, 'USER', self.user), patch.object(wa, 'list_game_windows', return_value=[self.info]):
            self.assertTrue(callable(getattr(window, 'activate', None)), 'startup activation is required')
            with self.assertRaises(RuntimeError):
                window.activate()

    def test_foreground_denial_times_out_without_retrying_activation(self):
        window = self.window()
        self.user.IsIconic.return_value = False
        self.user.IsIconic.side_effect = None
        self.user.GetForegroundWindow.return_value = 99
        with patch.object(wa, 'USER', self.user), patch.object(wa, 'list_game_windows', return_value=[self.info]):
            self.assertTrue(callable(getattr(window, 'activate', None)), 'startup activation is required')
            with self.assertRaises(RuntimeError):
                window.activate(timeout=.001)
        self.user.SetForegroundWindow.assert_called_once()

    def test_destroyed_window_never_activated(self):
        window = self.window()
        self.user.IsWindow.return_value = False
        with patch.object(wa, 'USER', self.user), patch.object(wa, 'list_game_windows', return_value=[self.info]):
            with self.assertRaises(RuntimeError):
                window.activate()
        self.user.ShowWindowAsync.assert_not_called()
        self.user.SetForegroundWindow.assert_not_called()

    def test_async_restore_waits_until_not_iconic_before_foreground_request(self):
        window = self.window()
        now = [0.0]
        self.user.IsIconic.side_effect = lambda hwnd: now[0] < .04
        def sleep(seconds):
            now[0] += seconds
        def foreground(hwnd):
            self.assertGreaterEqual(now[0], .04, 'foreground must wait for async restore')
            return True
        self.user.SetForegroundWindow.side_effect = foreground
        with patch.object(wa, 'USER', self.user), patch.object(wa, 'list_game_windows', return_value=[self.info]), \
                patch.object(wa.time, 'monotonic', side_effect=lambda: now[0]), patch.object(wa.time, 'sleep', side_effect=sleep):
            window.activate(timeout=.05)
        self.user.ShowWindowAsync.assert_called_once_with(91, 9)
        self.user.SetForegroundWindow.assert_called_once_with(91)
        self.assertLessEqual(now[0], .05)

    def test_async_restore_timeout_never_requests_foreground(self):
        window = self.window()
        now = [0.0]
        self.user.IsIconic.side_effect = None
        self.user.IsIconic.return_value = True
        def sleep(seconds):
            now[0] += seconds
        # Simulate time spent enqueueing restoration; it counts toward the same budget.
        self.user.ShowWindowAsync.side_effect = lambda hwnd, command: sleep(.03) or True
        with patch.object(wa, 'USER', self.user), patch.object(wa, 'list_game_windows', return_value=[self.info]), \
                patch.object(wa.time, 'monotonic', side_effect=lambda: now[0]), patch.object(wa.time, 'sleep', side_effect=sleep):
            with self.assertRaises(RuntimeError):
                window.activate(timeout=.05)
        self.user.ShowWindowAsync.assert_called_once_with(91, 9)
        self.user.SetForegroundWindow.assert_not_called()
        self.assertAlmostEqual(now[0], .05)


class StartupTests(unittest.TestCase):
    def test_capture_activation_is_optional_and_once(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
                window = MagicMock()
                window.info = dict(hwnd=91, pid=701, title='原神')
                window.capture.return_value = (MagicMock(size=(800, 600)), MagicMock(size=(640, 360)), (0, 0, 800, 600))
                stack.enter_context(patch.object(r, 'GameWindow', return_value=window))
                stack.enter_context(patch.object(r, 'list_game_windows', return_value=[]))
                stack.enter_context(patch.object(r, 'input_access_report', return_value={}))
                stack.enter_context(patch.object(r, 'EmergencyHotkey', return_value=MagicMock(return_value=False)))
                stack.enter_context(patch.object(r, 'GpuSampler'))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                code = r.main(['capture', '--countdown', '0', '--output', tmp] + (['--activate'] if enabled else []))
                self.assertEqual(code, 0)
                self.assertEqual(window.activate.call_count, int(enabled))

    def test_worker_cli_propagates_activation(self):
        with patch.object(b, 'worker', return_value=0) as worker:
            b.main(['worker', '--session', 'a' * 32, '--backend', 'cloud', '--activate'])
        worker.assert_called_once_with('a' * 32, backend='cloud', activate=True)

    def run_worker(self, root, **kwargs):
        with patch.object(b, 'session_path', return_value=root):
            return b.worker('a' * 32, **kwargs)

    def test_preexisting_stop_prevents_all_window_or_input_work(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            root = Path(tmp)
            (root / 'request').mkdir()
            (root / 'STOP').touch()
            window = stack.enter_context(patch.object(wa, 'GameWindow'))
            hotkey = stack.enter_context(patch.object(wa, 'EmergencyHotkey'))
            access = stack.enter_context(patch.object(wa, 'input_access_report'))
            sink = stack.enter_context(patch.object(wa, 'NativeInput'))
            self.assertNotEqual(self.run_worker(root, activate=True), 0)
            for mock in (window, hotkey, access, sink):
                mock.assert_not_called()
            self.assertIn('STOP', json.loads((root / 'closed.json').read_text())['reason'])

    def test_worker_checks_access_before_activation_and_freezes_geometry_after(self):
        for allowed in (False, True):
            with self.subTest(allowed=allowed), tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
                root = Path(tmp)
                (root / 'request').mkdir()
                events = []
                window = MagicMock()
                window.info = dict(hwnd=91, pid=701, title='原神')
                window.activate.side_effect = lambda: events.append('activate')
                window.rect.side_effect = lambda: events.append('rect') or (0, 0, 800, 600)
                stack.enter_context(patch.object(wa, 'GameWindow', return_value=window))
                stack.enter_context(patch.object(wa, 'EmergencyHotkey', return_value=MagicMock(return_value=False)))
                stack.enter_context(patch.object(wa, 'input_access_report', return_value={}))
                def access(report):
                    events.append('access')
                    if not allowed:
                        raise RuntimeError('Denied')
                stack.enter_context(patch.object(b, 'require_worker_access', side_effect=access))
                def job(*args, **kwargs):
                    (root / 'STOP').touch()
                    return {'status': 'passed'}
                stack.enter_context(patch.object(b, 'execute_job', side_effect=job))
                self.run_worker(root, activate=True)
                self.assertEqual(events, ['access', 'activate', 'rect'] if allowed else ['access'])

    def test_stop_arriving_during_access_check_prevents_activation(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            root = Path(tmp)
            (root / 'request').mkdir()
            window = stack.enter_context(patch.object(wa, 'GameWindow')).return_value
            hotkey = stack.enter_context(patch.object(wa, 'EmergencyHotkey'))
            bound = stack.enter_context(patch.object(b, 'BoundWindow'))
            stack.enter_context(patch.object(wa, 'input_access_report', return_value={}))
            stack.enter_context(patch.object(b, 'require_worker_access', side_effect=lambda report: (root / 'STOP').touch()))
            self.assertNotEqual(self.run_worker(root, activate=True), 0)
            window.activate.assert_not_called()
            bound.assert_not_called()
            hotkey.assert_not_called()

    def test_worker_activation_once_across_bootstrap_and_two_jobs(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            root = Path(tmp)
            (root / 'request').mkdir()
            for seq in (1, 2):
                (root / 'request' / f'req{seq:04d}.json').write_text('{"mode":"check"}')
            original = stack.enter_context(patch.object(wa, 'GameWindow')).return_value
            window = MagicMock()
            window.info = dict(hwnd=91, pid=701, title='原神')
            window.initial_rect = (0, 0, 800, 600)
            stack.enter_context(patch.object(b, 'BoundWindow', return_value=window))
            stack.enter_context(patch.object(wa, 'input_access_report', return_value={}))
            stack.enter_context(patch.object(b, 'require_worker_access'))
            stack.enter_context(patch.object(wa, 'EmergencyHotkey', return_value=MagicMock(return_value=False)))
            jobs = []
            def job(req, directory, window, stop, hotkey, **kwargs):
                jobs.append(directory.name)
                if len(jobs) == 3:
                    stop.trigger('Test complete')
                return {'status': 'passed'}
            stack.enter_context(patch.object(b, 'execute_job', side_effect=job))
            self.assertEqual(self.run_worker(root, activate=True), 0)
            original.activate.assert_called_once_with()
            self.assertEqual(jobs, ['bootstrap', 'req0001', 'req0002'])


class CaptureEvidenceTests(unittest.TestCase):
    def run_loop(self, mode):
        events, now = [], [0.0]
        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            window = MagicMock()
            window.info = dict(hwnd=91, pid=701, title='原神')
            rects = [(-800, 20, 0, 620), (-800, 20, 0, 620), (-1000, 20, 0, 720)]
            if mode == 'explore':
                rects = [rects[0]] * 3
            captures = []
            for index, rect in enumerate(rects):
                original = MagicMock(size=(rect[2] - rect[0], rect[3] - rect[1]))
                frame = MagicMock(size=(640, 360))
                def save(path, index=index):
                    events.append('save:' + path.name)
                    # A slow startup original save must finish before input time starts.
                    now[0] += .25
                original.save.side_effect = save
                captures.append((original, frame, rect))
            def capture():
                index = window.capture.call_count - 1
                events.append(f'capture:{index}')
                return captures[index]
            window.capture.side_effect = capture
            window.rect.return_value = rects[0]
            bridge = MagicMock()
            decision = dict(mouse=[0, 0, 0], steps=[['W']] * 6, parsed=True, truncated=False,
                            latency_ms=1, raw_text='W', pending_id='pending')
            def result(bridge, command, **kwargs):
                events.append(command)
                return decision.copy() if command == 'act' else {}
            guard = MagicMock()
            guard.reason = None
            guard.stop.is_set.return_value = False
            def start():
                events.append('guard_start')
                guard.deadline = now[0] + .01
            guard.start.side_effect = start
            def supervisor(*args, **kwargs):
                events.append('guard_create')
                return guard
            patches = {'GameWindow': MagicMock(return_value=window), 'list_game_windows': MagicMock(return_value=[]),
                       'input_access_report': MagicMock(return_value={'controller_integrity': 12288, 'game_integrity': 12288}),
                       'EmergencyHotkey': MagicMock(return_value=MagicMock(return_value=False)), 'GpuSampler': MagicMock(),
                       'create_bridge': MagicMock(return_value=bridge), 'result': result,
                       'validate_health': MagicMock(return_value={}), 'Supervisor': supervisor,
                       'NativeInput': MagicMock(), 'encode': MagicMock(return_value='frame'),
                       'execute_chunk': MagicMock(return_value=True)}
            for name, value in patches.items():
                stack.enter_context(patch.object(r, name, value))
            stack.enter_context(patch.object(r.time, 'monotonic', side_effect=lambda: now[0]))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            argv = [mode, '--backend', 'cloud', '--count', '2', '--seconds', '.01', '--countdown', '0', '--output', tmp]
            if mode == 'explore':
                argv += ['--execute']
            self.assertEqual(r.main(argv), 0)
            logs = [json.loads(line) for line in next(Path(tmp).glob('*/events.jsonl')).read_text().splitlines()]
            self.assertEqual([call.args[0] for call in patches['encode'].call_args_list],
                             [captures[1][1], captures[2][1]], 'setup frame must never become a decision input')
            return window, captures, events, logs, guard, now

    def test_setup_original_is_saved_before_guard_and_decisions_capture_fresh_frames(self):
        for mode in ('shadow', 'explore'):
            with self.subTest(mode=mode):
                window, captures, events, logs, guard, now = self.run_loop(mode)
                self.assertIn('save:original-start.png', events)
                self.assertLess(events.index('health'), events.index('capture:0'))
                self.assertLess(events.index('save:original-start.png'), events.index('guard_create'))
                self.assertLess(events.index('guard_start'), events.index('capture:1'))
                self.assertEqual(window.capture.call_count, 3)
                self.assertAlmostEqual(guard.deadline - now[0], .01)
                for original, frame, rect in captures[1:]:
                    original.save.assert_not_called()
                    frame.save.assert_called_once()
                self.assertEqual(sum(item == 'act' for item in events), 2)

    def test_setup_and_decisions_log_actual_capture_geometry_and_target(self):
        for mode in ('shadow', 'explore'):
            with self.subTest(mode=mode):
                window, captures, events, logs, guard, now = self.run_loop(mode)
                rows = [row for row in logs if row['event'] in ('setup_capture', 'decision')]
                self.assertEqual(len(rows), 3)
                for row, (original, frame, rect) in zip(rows, captures):
                    self.assertEqual(row['target'], r.backend_target('cloud'))
                    self.assertEqual(row['window']['hwnd'], window.info['hwnd'])
                    self.assertEqual(row['window']['pid'], window.info['pid'])
                    self.assertEqual(row['rect'], list(rect))
                    self.assertEqual(row['original_size'], list(original.size))
                    self.assertEqual(row['input_size'], list(frame.size))


if __name__ == '__main__':
    unittest.main()
