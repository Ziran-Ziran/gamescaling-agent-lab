import unittest

from controller.run_agent import run_outcome


class RunStatusTests(unittest.TestCase):
    def test_expired_first_decision_is_not_success(self):
        self.assertEqual(run_outcome('explore', [{'executed': False}], 'Time limit'), 'stopped')

    def test_valid_bounded_execution_and_shadow_are_distinct(self):
        self.assertEqual(run_outcome('explore', [{'executed': True}], 'Time limit'), 'passed')
        self.assertEqual(run_outcome('shadow', [{'executed': False}], 'Completed'), 'passed')

    def test_emergency_or_stale_stop_cannot_pass(self):
        self.assertEqual(run_outcome('explore', [{'executed': True}], 'Stale decision'), 'stopped')
