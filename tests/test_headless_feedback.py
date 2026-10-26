import unittest

from controller.control_core import SafeAction, to_action_text
from scripts.training.feedback import content_hash, reward_from_events, validate_feedback
from scripts.training.headless import HeadlessGame


def action(key=None):
    return SafeAction((0., 0., 0.), tuple((key,) if key else () for _ in range(6)))


class Clock:
    def __init__(self):
        self.now = 1_000_000_000

    def ns(self):
        return self.now

    def sleep(self, seconds):
        self.now += round(seconds * 1_000_000_000)


class HeadlessFeedbackTests(unittest.TestCase):
    def make(self, **kwargs):
        clock = Clock()
        game = HeadlessGame(clock_ns=clock.ns, sleep=clock.sleep, **kwargs)
        return game, clock

    def test_deterministic_rgb_contains_no_truth_payload(self):
        a, _ = self.make(seed=7)
        b, _ = self.make(seed=7)
        self.assertEqual(a.render().tobytes(), b.render().tobytes())
        self.assertEqual(a.snapshot(), b.snapshot())
        observation = a.observe()
        self.assertEqual(set(observation), {'image', 'captured_ns', 'episode_id', 'frame_id', 'clock_id'})
        self.assertNotIn('goal', str({k: v for k, v in observation.items() if k != 'image'}))

    def test_real_transition_reward_and_terminal_are_bound_to_actual_action(self):
        game, clock = self.make(variant='reach', layout={'player': (2, 2), 'goal': (3, 2)})
        observation = game.observe()
        outcome = game.step(action('D'), observed_ns=observation['captured_ns'])
        self.assertTrue(outcome['actual_action']['completed'])
        self.assertEqual(outcome['actual_action']['text'], to_action_text(action('D')))
        self.assertEqual(outcome['actual_action']['end_ns'] - outcome['actual_action']['start_ns'], 200_000_000)
        self.assertTrue(outcome['terminated'])
        self.assertFalse(outcome['truncated'])
        self.assertEqual(outcome['terminal_reason'], 'goal')
        self.assertAlmostEqual(outcome['reward'], .99)
        validate_feedback(outcome['feedback'])
        with self.assertRaisesRegex(RuntimeError, 'ended'):
            game.step(action(), observed_ns=clock.ns())

    def test_collision_and_closed_door_cannot_be_crossed(self):
        for obstacle in ('walls', 'door'):
            layout = {'player': (1, 1), 'goal': (4, 4)}
            layout[obstacle] = {(2, 1)} if obstacle == 'walls' else (2, 1)
            variant = 'reach' if obstacle == 'walls' else 'keydoor'
            if variant == 'keydoor':
                layout['key'] = (1, 3)
                layout['goal'] = layout['door']
            game, clock = self.make(variant=variant, layout=layout)
            result = game.step(action('D'), observed_ns=clock.ns())
            self.assertLess(game.snapshot()['player'][0], 2.)
            self.assertAlmostEqual(result['reward'], -.01)

    def test_hazard_and_horizon_have_distinct_end_semantics(self):
        game, clock = self.make(variant='hazard', layout={'player': (2, 2), 'goal': (4, 4), 'hazards': {(3, 2)}})
        result = game.step(action('D'), observed_ns=clock.ns())
        self.assertTrue(result['terminated'])
        self.assertEqual(result['terminal_reason'], 'hazard')
        self.assertAlmostEqual(result['reward'], -1.01)
        game, clock = self.make(max_steps=1)
        result = game.step(action(), observed_ns=clock.ns())
        self.assertTrue(result['truncated'])
        self.assertFalse(result['terminated'])
        self.assertEqual(result['terminal_reason'], 'horizon')
        self.assertAlmostEqual(result['reward'], -.01)

    def test_expert_solves_all_three_fully_rendered_rule_variants(self):
        for variant in ('reach', 'hazard', 'keydoor'):
            for seed in range(4):
                game, clock = self.make(variant=variant, seed=seed)
                while not game.done:
                    result = game.step(game.expert_action(), observed_ns=clock.ns())
                with self.subTest(variant=variant, seed=seed):
                    self.assertEqual(result['terminal_reason'], 'goal')
                    if variant == 'keydoor':
                        self.assertTrue(game.snapshot()['has_key'])

    def test_rewards_reject_fabrication_wrong_action_hash_and_bad_timestamps(self):
        game, clock = self.make()
        result = game.step(game.expert_action(), observed_ns=clock.ns())
        feedback = result['feedback']
        for field, value in (('reward', 1000.), ('action_sha256', 'bad'),
                             ('feedback_ns', 0), ('terminated', False)):
            altered = {**feedback, field: value}
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_feedback(altered)
        self.assertEqual(reward_from_events(['step', 'goal']), .99)
        with self.assertRaises(ValueError):
            reward_from_events(['step', 'goal', 'hazard'])

    def test_rehashed_state_cannot_claim_goal_at_the_wrong_position(self):
        game, clock = self.make(variant='reach', layout={'player': (2, 2), 'goal': (3, 2)})
        feedback = game.step(action('D'), observed_ns=clock.ns())['feedback']
        changed = {**feedback, 'state_after': {**feedback['state_after'], 'player': [1.5, 1.5]}}
        changed['state_after_sha256'] = content_hash(changed['state_after'])
        with self.assertRaises(ValueError):
            validate_feedback(changed)

    def test_invalid_or_hidden_object_layouts_are_rejected(self):
        for layout in ({'door': (0, 0)}, {'key': (3, 3)}, {'key': (3, 2), 'door': (3, 2)},
                       {'door': (5, 5), 'goal': (3, 2), 'key': (1, 1)}):
            with self.subTest(layout=layout), self.assertRaises(ValueError):
                self.make(variant='keydoor', layout=layout)

    def test_rehashed_state_must_follow_recorded_action_physics(self):
        game, clock = self.make(max_steps=2, layout={'goal': (5, 5)})
        feedback = game.step(action('D'), observed_ns=clock.ns())['feedback']
        changed = {**feedback, 'state_after': {**feedback['state_after'], 'player': [2.5, 3.5]}}
        changed['state_after_sha256'] = content_hash(changed['state_after'])
        with self.assertRaisesRegex(ValueError, 'physics'):
            validate_feedback(changed)

    def test_mouse_or_unsafe_direct_actions_never_reach_virtual_executor(self):
        game, clock = self.make()
        invalid = SafeAction((1., 0., 0.), ((),) * 6)
        with self.assertRaises(ValueError):
            game.step(invalid, observed_ns=clock.ns())


if __name__ == '__main__':
    unittest.main()
