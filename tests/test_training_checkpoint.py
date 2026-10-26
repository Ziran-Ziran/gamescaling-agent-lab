import importlib.util
from pathlib import Path
import tempfile
import unittest

from scripts.training.checkpoint import save_checkpoint, restore_checkpoint


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional training runtime required (torch)')
class CheckpointTests(unittest.TestCase):
    def test_resume_matches_uninterrupted_optimizer_and_rng(self):
        import random
        import torch
        torch.manual_seed(7)
        random.seed(7)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        identity = {'dataset_sha256': 'abc', 'model_sha256': 'def', 'seed': 7}

        def step(net, opt):
            x = torch.randn(2, 3) * random.random()
            opt.zero_grad(set_to_none=True)
            loss = net(x).square().mean()
            loss.backward()
            opt.step()
            return loss.item()

        step(model, optimizer)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'checkpoint'
            save_checkpoint(path, model, optimizer, step=1, identity=identity)
            expected_loss = step(model, optimizer)
            expected = {k: v.clone() for k, v in model.state_dict().items()}
            other = torch.nn.Linear(3, 2)
            other_optimizer = torch.optim.AdamW(other.parameters(), lr=.01)
            restored = restore_checkpoint(path, other, other_optimizer, identity=identity)
            self.assertEqual(restored, 1)
            self.assertEqual(step(other, other_optimizer), expected_loss)
            for key, value in other.state_dict().items():
                self.assertTrue(torch.equal(value, expected[key]), key)
            with self.assertRaisesRegex(ValueError, 'identity'):
                restore_checkpoint(path, other, other_optimizer,
                                   identity={**identity, 'dataset_sha256': 'changed'})
            with self.assertRaises(FileExistsError):
                save_checkpoint(path, model, optimizer, step=2, identity=identity)


if __name__ == '__main__':
    unittest.main()
