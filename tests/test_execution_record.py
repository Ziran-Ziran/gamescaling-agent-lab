import threading
import unittest
from unittest.mock import patch

from controller.control_core import sanitize_action
from controller.run_agent import execute_and_record


class ExecutionRecordTests(unittest.TestCase):
    def test_calibration_records_input_without_history(self):
        from types import SimpleNamespace
        action = sanitize_action([0, 0, 0], [['W']] * 6)
        records, events = [], []
        guard = SimpleNamespace(stop=threading.Event(), trigger=lambda reason: None)
        sink = SimpleNamespace(key_down=lambda key: None, key_up=lambda key: None)
        clock = [1.0]
        def execute(action, tracked, stop):
            tracked.key_down('W')
            clock[0] += .2
            return True
        entry = {'index': 0, 'model_ms': None}
        with patch('controller.run_agent.execute_chunk', execute), patch('controller.run_agent.time.monotonic', lambda: clock[0]):
            execute_and_record(action, sink, guard, entry, records,
                lambda event, **data: events.append((event, data)), None, None, cycle_start=1)
        self.assertTrue(records[0]['executed'])
        self.assertEqual(records[0]['sent_inputs'], 1)
        self.assertEqual(records[0]['history_status'], 'not_applicable')
        self.assertAlmostEqual(events[0][1]['cycle_ms'], 200)

    def test_history_event_includes_commit_elapsed_time(self):
        from types import SimpleNamespace
        clock = [1.0]
        events, records = [], []
        guard = SimpleNamespace(stop=threading.Event(), trigger=lambda reason: None)
        def execute(*args):
            clock[0] += .2
            return True
        def commit(): clock[0] += .3
        with patch('controller.run_agent.execute_chunk', execute), patch('controller.run_agent.time.monotonic', lambda: clock[0]):
            execute_and_record(None, None, guard, {'index': 0}, records,
                lambda event, **data: events.append((event, data)), commit, lambda: None, cycle_start=1)
        self.assertAlmostEqual(events[0][1]['cycle_ms'], 200)
        self.assertAlmostEqual(events[-1][1]['cycle_ms'], 500)
        self.assertAlmostEqual(records[0]['cycle_ms'], 500)

    def run_case(self, execute, commit):
        class Sink:
            def key_down(self, key): pass
            def key_up(self, key): pass
            def move_relative(self, x, y): pass
        class Guard:
            stop = threading.Event()
            def trigger(self, reason): self.stop.set()
        guard = Guard()
        records, events = [], []
        entry = {'index': 0}
        action = sanitize_action([0, 0, 0], [['W']] * 6)
        with patch('controller.run_agent.execute_chunk', execute):
            try:
                execute_and_record(action, Sink(), guard, entry, records,
                    lambda event, **data: events.append((event, data.copy())),
                    lambda: commit(records, events), lambda: None)
            except TimeoutError:
                pass
        return guard, records, events

    def test_commit_failure_preserves_completed_input_and_stops(self):
        def execute(action, sink, stop):
            sink.key_down('W')
            return True
        def commit(records, events):
            self.assertTrue(records[0]['executed'])
            self.assertEqual(events[0][0], 'decision')
            raise TimeoutError('history transport lost')
        guard, records, events = self.run_case(execute, commit)
        self.assertTrue(guard.stop.is_set())
        self.assertTrue(records[0]['executed'])
        self.assertEqual(records[0]['sent_inputs'], 1)
        self.assertEqual(records[0]['history_status'], 'failed')
        self.assertEqual(events[-1][0], 'decision_failure')

    def test_interrupted_input_is_recorded_without_commit(self):
        def execute(action, sink, stop):
            sink.key_down('W')
            return False
        guard, records, events = self.run_case(execute,
            lambda *args: self.fail('Interrupted action must not be committed'))
        self.assertFalse(records[0]['executed'])
        self.assertTrue(records[0]['partial_execution'])
        self.assertEqual(records[0]['history_status'], 'reset')

    def test_execution_exception_preserves_partial_input(self):
        def execute(action, sink, stop):
            sink.key_down('W')
            raise TimeoutError('backend failed after sending input')
        guard, records, events = self.run_case(execute,
            lambda *args: self.fail('Failed execution must not be committed'))
        self.assertTrue(guard.stop.is_set())
        self.assertTrue(records[0]['partial_execution'])
        self.assertEqual(events[0][0], 'decision')
