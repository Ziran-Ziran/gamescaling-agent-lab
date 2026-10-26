import threading
import time
import unittest

from controller.supervisor import Supervisor


class SupervisorTests(unittest.TestCase):
    def test_stop_is_independent_of_a_blocking_request(self):
        released = threading.Event()
        emergency = threading.Event()
        guard = Supervisor(lambda: True, emergency.is_set, released.set, 1)
        guard.start()
        emergency.set()
        self.assertTrue(released.wait(.5))
        self.assertEqual(guard.reason, 'F8 emergency stop')
        guard.close()
        self.assertEqual(guard.reason, 'F8 emergency stop')

    def test_focus_loss_and_deadline_release(self):
        for foreground, seconds in [(False, 1), (True, .03)]:
            released = threading.Event()
            guard = Supervisor(lambda: foreground, lambda: False, released.set, seconds)
            guard.start()
            self.assertTrue(released.wait(.5))
            guard.close()

    def test_unbounded_runs_are_rejected(self):
        for limit in [0, -1, 61]:
            with self.assertRaises(ValueError):
                Supervisor(lambda: True, lambda: False, lambda: None, limit)
