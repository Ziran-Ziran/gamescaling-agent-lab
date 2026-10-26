import unittest

from scripts.training.formatting import assistant_labels, make_messages


class FormattingTests(unittest.TestCase):
    def test_only_action_completion_receives_loss(self):
        prompt, full = [10, 11, 12], [10, 11, 12, 200, 8, 201]
        self.assertEqual(assistant_labels(prompt, full), [-100, -100, -100, 200, 8, 201])
        with self.assertRaises(ValueError):
            assistant_labels(prompt, [10, 11, 0, 200])
        with self.assertRaises(ValueError):
            assistant_labels(prompt, prompt)

    def test_messages_keep_actual_history_and_current_target_out_of_prompt(self):
        row = {'system_prompt': 'system', 'rgb': {'path': 'now.png'},
               'action': 'CURRENT TARGET',
               'history': [{'rgb': {'path': 'old.png'}, 'action': 'ACTUAL PAST'}]}
        messages, paths = make_messages(row, history_len=1)
        self.assertEqual(paths, ['old.png', 'now.png'])
        self.assertEqual(messages[2]['content'], 'ACTUAL PAST')
        self.assertNotIn('CURRENT TARGET', str(messages))
        self.assertEqual(messages[-1]['content'], '<think>\n\n</think>')
        no_history, paths = make_messages(row, history_len=0)
        self.assertEqual(paths, ['now.png'])
        self.assertEqual(len(no_history), 3)


if __name__ == '__main__':
    unittest.main()
