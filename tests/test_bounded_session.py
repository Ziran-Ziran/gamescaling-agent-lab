import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

try:
    from controller import bounded_session as b
except ImportError:
    b = None


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(b, 'bounded_session module is not implemented yet')

    def test_unknown_fields_and_unbounded_values_rejected(self):
        for request in [dict(mode='check', command='whoami'), dict(mode='check', output='x'),
                        dict(mode='capture', title='other'), dict(mode='explore', seconds=11),
                        dict(mode='shadow', count=31), dict(mode='explore', count=True),
                        dict(mode='explore', seconds=float('nan')), dict(mode='explore', history=2),
                        dict(mode='shadow', history=1), dict(mode='calibrate', allow_mouse='yes'),
                        dict(mode='check', seconds=1), dict(mode='explore', execute=True)]:
            with self.subTest(request=request), self.assertRaises(ValueError):
                b.validate_request(request)

    def test_defaults_and_camera_require_explicit_mouse(self):
        req = b.validate_request({'mode': 'explore'})
        self.assertEqual((req['seconds'], req['count'], req['history']), (10, 30, 0))
        with self.assertRaises(ValueError):
            b.validate_request({'mode': 'calibrate', 'action': 'left'})

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaises(ValueError):
            b.decode_request('{"mode":"check","mode":"explore"}')

    def test_sequential_consumption_and_duplicate_replay_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'request').mkdir()
            queue = b.RequestQueue(root)
            (root / 'request' / 'req0002.json').write_text('{"mode":"check"}')
            self.assertIsNone(queue.claim())
            (root / 'request' / 'req0001.json').write_text('{"mode":"capture"}')
            seq, raw = queue.claim()
            self.assertEqual(seq, 1)
            self.assertEqual(b.decode_request(raw), {'mode': 'capture'})
            (root / 'request' / 'req0001.json').write_text('{"mode":"explore"}')
            with self.assertRaises(RuntimeError):
                queue.claim()

    def test_job_limit_cannot_be_extended(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'request').mkdir()
            queue = b.RequestQueue(root)
            for i in range(1, 13):
                (root / 'request' / f'req{i:04}.json').write_text('{"mode":"check"}')
                self.assertEqual(queue.claim()[0], i)
            (root / 'request' / 'req0013.json').write_text('{"mode":"check"}')
            with self.assertRaises(RuntimeError):
                queue.claim()


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(b, 'bounded_session module is not implemented yet')

    def test_job_preserves_inner_stale_decision_reason_and_summary(self):
        class Window:
            info = {'pid': 123}
            def rect(self): return (0, 0, 800, 600)
            def is_foreground(self): return True
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stop = b.SessionStop(root, time.monotonic() + 20)
            summary = dict(status='stopped', stop_reason='Stale decision', decisions=3, sent_inputs=0)
            stop.window = Window()
            def stopped_main(argv):
                folder = Path(argv[argv.index('--output') + 1]) / 'test-explore'
                folder.mkdir(parents=True)
                (folder / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
                return 2
            with patch.object(b.wa, 'input_access_report', return_value={
                    'controller_integrity':12288, 'game_integrity':12288}), \
                    patch.object(b.run_agent, 'main', side_effect=stopped_main), \
                    patch.object(b, 'save_original', return_value=False), \
                    patch.object(b, 'synthetic_hotkey_selftest'):
                result = b.execute_job(b.validate_request({'mode':'explore'}), root/'job',
                                       Window(), stop, lambda: False)
            self.assertEqual(result['status'], 'stopped')
            self.assertEqual(result['stop_reason'], 'Stale decision')
            self.assertEqual(result['agent_summary'], summary)

    def test_stop_file_interrupts_active_supervision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stop = b.SessionStop(root, time.monotonic() + 20)
            released = threading.Event()
            guard = b.external_supervisor(stop.reason)(lambda: True, lambda: False,
                                                       released.set, 10)
            guard.start()
            (root / 'STOP').touch()
            self.assertTrue(released.wait(.5), 'STOP must release during a running job')
            self.assertIn('STOP', guard.reason)
            guard.close()

    def test_session_deadline_interrupts_long_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            stop = b.SessionStop(Path(tmp), time.monotonic() + .03)
            released = threading.Event()
            guard = b.external_supervisor(stop.reason)(lambda: True, lambda: False,
                                                       released.set, 10)
            guard.start()
            self.assertTrue(released.wait(.5))
            self.assertIn('deadline', guard.reason)
            guard.close()

    def test_original_window_identity_and_geometry_are_fixed(self):
        class Window:
            info = {'hwnd': 100, 'pid': 123, 'title': '原神'}
            def rect(self): return (0, 0, 800, 600)
            def is_foreground(self): return True
        matches = [Window.info.copy()]
        window = b.BoundWindow(Window(), lambda: matches)
        self.assertTrue(window.is_foreground())
        matches[0]['pid'] = 456
        self.assertFalse(window.is_foreground())
        with self.assertRaises(RuntimeError):
            window.rect()

    def selftest_objects(self):
        class LatchedEvent(threading.Event):
            def clear(self): raise AssertionError('Emergency events must never be cleared')
        class Hotkey:
            def __init__(self, vk=0x77):
                self.vk = vk
                self.pressed = LatchedEvent()
                self.error = None
                self.closed = False
            def __call__(self): return self.pressed.is_set()
            def close(self): self.closed = True
        return Hotkey(), Hotkey(0x87)

    def invoke_selftest(self, emergency, testkey, send, **kwargs):
        def factory(*, vk):
            self.assertEqual(vk, 0x87)
            return testkey
        return b.synthetic_hotkey_selftest(emergency, send, hotkey_factory=factory,
                                           timeout=.02, **kwargs)

    def test_selftest_real_f8_during_send_aborts_and_stays_latched(self):
        emergency, testkey = self.selftest_objects()
        edges = []
        def send(up):
            edges.append(up)
            if not up:
                testkey.pressed.set()
                emergency.pressed.set()
        with self.assertRaisesRegex(RuntimeError, 'F8'):
            self.invoke_selftest(emergency, testkey, send)
        self.assertTrue(emergency.pressed.is_set())
        self.assertEqual(edges, [False, True])
        self.assertTrue(testkey.closed)

    def test_selftest_separate_event_never_clears_either_latch(self):
        emergency, testkey = self.selftest_objects()
        def send(up):
            if not up: testkey.pressed.set()
        self.invoke_selftest(emergency, testkey, send)
        self.assertFalse(emergency.pressed.is_set())
        self.assertTrue(testkey.pressed.is_set())
        self.assertTrue(testkey.closed)

    def test_selftest_failed_f24_down_still_sends_up_and_closes(self):
        emergency, testkey = self.selftest_objects()
        edges = []
        def send(up):
            edges.append(up)
            if not up: raise RuntimeError('F24 down failed')
        with self.assertRaisesRegex(RuntimeError, 'F24 down failed'):
            self.invoke_selftest(emergency, testkey, send)
        self.assertEqual(edges, [False, True])
        self.assertTrue(testkey.closed)

    def test_selftest_requires_f24_event_and_honors_stop_after_send(self):
        emergency, testkey = self.selftest_objects()
        with self.assertRaises(RuntimeError):
            self.invoke_selftest(emergency, testkey, lambda up: None)
        emergency, testkey = self.selftest_objects()
        stopped = threading.Event()
        def send(up):
            if not up: testkey.pressed.set()
            else: stopped.set()
        with self.assertRaisesRegex(RuntimeError, 'STOP'):
            self.invoke_selftest(emergency, testkey, send,
                                 stop_reason=lambda: 'Global STOP' if stopped.is_set() else None)

    def test_plain_integrity_cannot_start_admin_worker(self):
        with self.assertRaises(RuntimeError):
            b.require_worker_access({'controller_integrity':8192, 'game_integrity':12288})

    def test_completed_time_limit_does_not_poison_following_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            stop = b.SessionStop(Path(tmp), time.monotonic() + 20)
            guard = b.external_supervisor(stop.reason)(lambda: True, lambda: False, lambda: None, .02)
            guard.start()
            self.assertTrue(guard.stop.wait(.5))
            self.assertEqual(guard.reason, 'Time limit')
            guard.close()
            self.assertIsNone(stop.reason())

    def test_all_invalid_json_shapes_are_rejected_as_value_errors(self):
        for value in [None, [], {}, {'mode': []}, {'mode': {}}, {'mode': True}]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                b.validate_request(value)

    def test_stop_reaches_reused_run_agent_guard_while_main_blocks(self):
        class Window:
            info = {'pid': 123}
            def rect(self): return (0, 0, 800, 600)
            def is_foreground(self): return False
        class Hotkey:
            def __call__(self): return False
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stop = b.SessionStop(root, time.monotonic() + 20)
            stop.window = Window()
            ready, released = threading.Event(), threading.Event()
            old_supervisor = b.run_agent.Supervisor
            def blocking_main(argv):
                guard = b.run_agent.Supervisor(lambda: True, lambda: False, released.set, 10)
                guard.start()
                ready.set()
                released.wait(1)
                guard.close()
                return 2
            with patch.object(b.wa, 'input_access_report', return_value={
                    'controller_integrity':12288, 'game_integrity':12288}), \
                    patch.object(b.run_agent, 'main', side_effect=blocking_main):
                thread = threading.Thread(target=b.execute_job, args=(
                    b.validate_request({'mode':'check'}), root/'job', Window(), stop, Hotkey()))
                thread.start()
                self.assertTrue(ready.wait(1))
                (root / 'STOP').touch()
                self.assertTrue(released.wait(.5))
                thread.join(1)
                self.assertFalse(thread.is_alive())
            self.assertIs(b.run_agent.Supervisor, old_supervisor)
            self.assertEqual(json.loads((root/'job'/'result.json').read_text())['status'], 'stopped')

    def test_foreground_can_change_after_main_cleanup_and_between_jobs(self):
        class Window:
            info = {'pid': 123}
            foreground = True
            def rect(self): return (0, 0, 800, 600)
            def is_foreground(self): return self.foreground
        hotkey, _ = self.selftest_objects()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            window = Window()
            stop = b.SessionStop(root, time.monotonic() + 20)
            stop.window = window
            def screenshot(window, directory, name):
                if name == 'after.png':
                    # main has returned, including its release/cleanup finally.
                    window.foreground = False
                    self.assertFalse(stop.armed)
                    return False
                return True
            with patch.object(b.wa, 'input_access_report', return_value={
                    'controller_integrity':12288, 'game_integrity':12288}), \
                    patch.object(b.run_agent, 'main', return_value=0), \
                    patch.object(b, 'save_original', side_effect=screenshot):
                outcome = b.execute_job(b.validate_request({'mode':'capture'}), root/'job',
                                        window, stop, hotkey)
            self.assertEqual(outcome['status'], 'passed')
            self.assertNotIn('after_error', outcome)
            self.assertIsNone(stop.reason())
            self.assertFalse(stop.armed)
            # The same idle permission never masks a real emergency event.
            hotkey.pressed.set()
            self.assertIn('F8', stop.reason())

    def test_cleanup_still_holding_inputs_remains_armed_and_focus_loss_latches(self):
        class Window:
            info = {'pid': 123}
            foreground = True
            def rect(self): return (0, 0, 800, 600)
            def is_foreground(self): return self.foreground
        hotkey, _ = self.selftest_objects()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            window = Window()
            stop = b.SessionStop(root, time.monotonic() + 20)
            stop.window = window
            cleanup_started, cleanup_allowed, released = (threading.Event() for _ in range(3))
            armed_during_cleanup = []
            def main_with_pending_cleanup(argv):
                guard = b.run_agent.Supervisor(window.is_foreground, lambda: False, released.set, 10)
                guard.start()
                try:
                    cleanup_started.set()
                    cleanup_allowed.wait(1)
                finally:
                    armed_during_cleanup.append(stop.armed)
                    guard.close()
                return 2
            with patch.object(b.wa, 'input_access_report', return_value={
                    'controller_integrity':12288, 'game_integrity':12288}), \
                    patch.object(b.run_agent, 'main', side_effect=main_with_pending_cleanup), \
                    patch.object(b, 'save_original', return_value=False):
                thread = threading.Thread(target=b.execute_job, args=(
                    b.validate_request({'mode':'capture'}), root/'job', window, stop, hotkey))
                thread.start()
                try:
                    self.assertTrue(cleanup_started.wait(1))
                    window.foreground = False
                    self.assertTrue(released.wait(.5))
                finally:
                    cleanup_allowed.set()
                    thread.join(1)
            self.assertEqual(armed_during_cleanup, [True])
            self.assertFalse(stop.armed)
            self.assertIn('foreground', stop.reason())


if __name__ == '__main__':
    unittest.main()
