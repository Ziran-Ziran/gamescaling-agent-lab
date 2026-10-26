import importlib.util
import os
import unittest

from scripts.training.hf_backend import configure_determinism


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional training runtime required (torch)')
class DeterminismTests(unittest.TestCase):
    def test_training_mode_sets_and_reports_process_local_determinism(self):
        import torch
        old_env = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
        old_enabled = torch.are_deterministic_algorithms_enabled()
        old_cudnn = torch.backends.cudnn.deterministic
        old_benchmark = torch.backends.cudnn.benchmark
        try:
            state = configure_determinism(True)
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            self.assertTrue(torch.backends.cudnn.deterministic)
            self.assertFalse(torch.backends.cudnn.benchmark)
            self.assertEqual(os.environ['CUBLAS_WORKSPACE_CONFIG'], ':4096:8')
            self.assertEqual(state['cublas_workspace_config'], ':4096:8')
            configure_determinism(False)
            self.assertFalse(torch.are_deterministic_algorithms_enabled())
        finally:
            torch.use_deterministic_algorithms(old_enabled)
            torch.backends.cudnn.deterministic = old_cudnn
            torch.backends.cudnn.benchmark = old_benchmark
            if old_env is None:
                os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
            else:
                os.environ['CUBLAS_WORKSPACE_CONFIG'] = old_env


if __name__ == '__main__':
    unittest.main()
