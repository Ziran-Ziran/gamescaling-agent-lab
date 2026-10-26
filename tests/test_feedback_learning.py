import copy
from pathlib import Path
import random
import tempfile
import unittest

from scripts.training.feedback_learning import (bandit_loss, class_from_action, index_for_step,
                                                student_row, supervised_loss)
from scripts.training.action_head import LABELS, template_action
from controller.control_core import to_action_text

try:
    import torch
except ImportError:
    torch = None


class FeedbackBoundaryTests(unittest.TestCase):
    def test_reset_audit_binds_visible_horizon_seed_and_actual_expert_success(self):
        from scripts.training.bandit_train import checked_resets
        from scripts.training.headless import HeadlessGame
        game = HeadlessGame(seed=7, max_steps=1)
        row = {'game_id': 'headless-reach-v1', 'frame_id': 0, 'episode_id': game.episode_id,
               'sample_id': game.episode_id + ':0'}
        episode = {'episode_id': game.episode_id, 'frames': [
            {'terminal_reason': 'goal', 'reward': .99, 'feedback': {'state_before': game.snapshot()}}]}
        self.assertEqual(checked_resets([row], [episode]), {row['sample_id']: 7})
        for state_change in ({'max_steps': 8}, {'player': [1.5, 1.5]}):
            altered = copy.deepcopy(episode)
            altered['frames'][0]['feedback']['state_before'].update(state_change)
            with self.subTest(state=state_change), self.assertRaises(ValueError):
                checked_resets([row], [altered])
        altered = copy.deepcopy(episode)
        altered['frames'][0]['terminal_reason'] = 'horizon'
        with self.assertRaises(ValueError):
            checked_resets([row], [altered])

    def test_expert_label_requires_exact_restricted_executed_action(self):
        for index, label in enumerate(LABELS):
            self.assertEqual(class_from_action(to_action_text(template_action(label))), index)
        for invalid in ('bad', '<|action_start|>0 0 0 ; W ; A ; W ; W ; W ; W<|action_end|>',
                        '<|action_start|>1 0 0 ; W ; W ; W ; W ; W ; W<|action_end|>'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                class_from_action(invalid)

    def test_student_payload_excludes_teacher_and_reward_truth(self):
        original = {'system_prompt': 'visible rules', 'rgb': {'path': 'frame.png'},
                    'history': [{'action': 'past'}], 'action': 'secret teacher action',
                    'reward': 100., 'state_before': {'goal': [4, 4]}}
        row = student_row(original)
        self.assertEqual(set(row), {'system_prompt', 'rgb', 'history'})
        self.assertEqual(row['history'], [])
        row['rgb']['path'] = 'changed.png'
        self.assertEqual(original['rgb']['path'], 'frame.png')

    def test_data_order_restores_from_step_without_consuming_global_rng(self):
        state = random.getstate()
        full = [index_for_step(64, step, 20261028) for step in range(128)]
        self.assertEqual(set(full[:64]), set(range(64)))
        self.assertEqual(set(full[64:]), set(range(64)))
        self.assertEqual(full[4:], [index_for_step(64, step, 20261028) for step in range(4, 128)])
        self.assertEqual(state, random.getstate())


@unittest.skipIf(torch is None, 'optional torch CPU dependency unavailable')
class FeedbackLossTests(unittest.TestCase):
    def test_actual_reward_sign_reinforces_or_suppresses_sampled_class(self):
        for reward, sign in ((.99, -1), (-.01, 1)):
            logits = torch.zeros(5, requires_grad=True)
            loss = bandit_loss(logits, 0, reward, temperature=2., baseline=.19)
            loss.backward()
            self.assertEqual(-1 if logits.grad[0] < 0 else 1, sign)
            self.assertTrue(torch.isfinite(logits.grad).all())
        logits = torch.zeros(5, requires_grad=True)
        supervised_loss(logits, 2, temperature=2.).backward()
        self.assertLess(logits.grad[2], 0.)

    def test_feedback_is_detached_finite_scalar_and_cannot_smuggle_teacher_target(self):
        for reward in (float('nan'), float('inf'), torch.tensor(.99, requires_grad=True), True):
            with self.subTest(reward=reward), self.assertRaises(ValueError):
                bandit_loss(torch.zeros(5), 1, reward)
        for label in (-1, 5, True):
            with self.subTest(label=label), self.assertRaises(ValueError):
                supervised_loss(torch.zeros(5), label)

    def test_sampled_training_checkpoint_resume_matches_uninterrupted_cpu_run(self):
        from scripts.training.checkpoint import restore_checkpoint, save_checkpoint
        identity = {'fixture': 'bandit-five-class', 'temperature': 2., 'baseline': .19}

        def fresh():
            torch.manual_seed(20261028)
            model = torch.nn.Linear(2, 5)
            return model, torch.optim.AdamW(model.parameters(), lr=.01)

        def train(model, optimizer, start, end):
            log = []
            for step in range(start, end):
                index = index_for_step(2, step, 9)
                logits = model(torch.eye(2)[index])
                sampled = int(torch.distributions.Categorical(logits=logits / 2.).sample())
                reward = .99 if sampled == index else -.01  # Synthetic test environment only.
                loss = bandit_loss(logits, sampled, reward)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                log.append((sampled, reward, float(loss.detach())))
            return log

        model, optimizer = fresh()
        expected = train(model, optimizer, 0, 8)
        expected_parameters = copy.deepcopy(model.state_dict())
        expected_optimizer = copy.deepcopy(optimizer.state_dict())
        expected_rng = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as folder:
            model, optimizer = fresh()
            observed = train(model, optimizer, 0, 3)
            path = Path(folder) / 'checkpoint'
            save_checkpoint(path, model, optimizer, step=3, identity=identity)
            model, optimizer = fresh()
            step = restore_checkpoint(path, model, optimizer, identity=identity)
            observed.extend(train(model, optimizer, step, 8))
        self.assertEqual(expected, observed)
        self.assertTrue(all(torch.equal(value, model.state_dict()[key]) for key, value in expected_parameters.items()))
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        torch.testing.assert_close(expected_optimizer, optimizer.state_dict(), rtol=0., atol=0.)


if __name__ == '__main__':
    unittest.main()
