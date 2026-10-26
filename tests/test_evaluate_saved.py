import unittest
import importlib.util
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from controller import evaluate_saved as ev

HAS_OFFICIAL = importlib.util.find_spec('gamescaling') is not None
NEEDS_OFFICIAL = 'Official gamescaling package required; run WSL tests with PYTHONPATH=/home/ziran/gamescaling-lab/GameScaling:.'


class SavedEvaluationTests(unittest.TestCase):
    @unittest.skipUnless(HAS_OFFICIAL, NEEDS_OFFICIAL)
    def test_controls_read_utf8_prompt_on_a_gbk_default_host(self):
        from PIL import Image
        native_read = Path.read_text
        def legacy_read(path, encoding=None, errors=None):
            return native_read(path, encoding=encoding or 'gbk', errors=errors)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'examples').mkdir(); (root/'prompts').mkdir()
            Image.new('RGB', (32, 18)).save(root/'examples/doom_battle1.jpg')
            (root/'prompts/example_doom_battle1.txt').write_text('Aim at the enemy’s head — fire.', encoding='utf-8')
            argv = ['evaluate_saved', '--controls', '--repeats', '1', '--image',
                    str(root/'examples/doom_battle1.jpg'), '--official-repo', str(root),
                    '--output', str(root/'out')]
            with patch('sys.argv', argv), patch('requests.Session') as session, \
                    patch.object(ev, 'control_plan', return_value=[]), \
                    patch.object(ev.subprocess, 'run', return_value=SimpleNamespace(stdout='', stderr='')), \
                    patch.object(Path, 'read_text', new=legacy_read), \
                    contextlib.redirect_stdout(io.StringIO()):
                session.return_value.get.return_value.json.return_value = {'data': []}
                session.return_value.get.return_value.text = ''
                ev.main()
            config = json.loads(next((root/'out').glob('*/config.json')).read_text(encoding='utf-8'))
            self.assertIn('doom_sha256', config)
            session.return_value.post.assert_not_called()

    @unittest.skipUnless(HAS_OFFICIAL, NEEDS_OFFICIAL)
    def test_strict_mouse_rejects_nonfinite_and_out_of_range_raw_numbers(self):
        for mouse in ('inf 0 0', 'nan 0 0', '-inf 0 0', '1001 0 0',
                      '0 -1001 0', '0 0 6', '0 0 -6', 'x 0 0'):
            with self.subTest(mouse=mouse):
                text = f'<think></think><|action_start|>{mouse} ; W ; W ; W ; W ; W ; W<|action_end|>'
                self.assertFalse(ev.inspect_action(text, 'W')['strict_valid'])
        text = '<think></think><|action_start|>-1000 1000 -5 ; W ; W ; W ; W ; W ; W<|action_end|>'
        self.assertTrue(ev.inspect_action(text, 'W')['strict_valid'])

    def test_doom_default_control_matches_official_sampling(self):
        cases = ev.control_plan(3, doom_defaults=True)
        self.assertEqual(len(cases), 3)
        self.assertEqual({c['seed'] for c in cases}, {3000, 3001, 3002})
        self.assertTrue(all(c['reasoning'] and c['temperature'] == 1 for c in cases))

    @unittest.skipUnless(HAS_OFFICIAL, NEEDS_OFFICIAL)
    def test_canonical_control_has_no_extra_prefix_and_nine_requests(self):
        cases = ev.control_plan(3)
        self.assertEqual(len(cases), 9)
        self.assertEqual({c['seed'] for c in cases}, {3000})
        body = ev.make_body(cases[0], 'data:image/jpeg;base64,TEST', None)
        self.assertTrue(body['messages'][0]['content'].startswith('You are a gaming expert'))
        self.assertNotIn('identifier', body['messages'][0]['content'])
        self.assertNotIn('synthetic', body['messages'][0]['content'])

    @unittest.skipUnless(HAS_OFFICIAL, NEEDS_OFFICIAL)
    def test_history_is_explicit_synthetic_noop_and_seed_is_recorded(self):
        case = {'history': 5, 'reasoning': False, 'temperature': 0, 'seed': 7,
                'instruction': ev.INSTRUCTIONS['W'], 'id': 'load-5'}
        body = ev.make_body(case, 'data:image/jpeg;base64,TEST', 'nonce')
        self.assertEqual(body['seed'], 7)
        self.assertEqual(len(body['messages']), 13)
        self.assertIn('synthetic', body['messages'][0]['content'])
        for msg in body['messages'][2:11:2]:
            self.assertEqual(msg['content'], ev.NOOP)
        self.assertEqual(body['messages'][-1]['content'], '<think>\n\n</think>')

    @unittest.skipUnless(HAS_OFFICIAL, NEEDS_OFFICIAL)
    def test_short_skeleton_is_not_strictly_valid_even_if_parser_recovers(self):
        text = '<think></think><|action_start|>0 0 0 ; W<|action_end|>'
        result = ev.inspect_action(text, 'W')
        self.assertTrue(result['parsed'])
        self.assertFalse(result['strict_valid'])
        self.assertFalse(result['instruction_match'])

    @unittest.skipUnless(HAS_OFFICIAL, NEEDS_OFFICIAL)
    def test_opposite_instruction_requires_exact_keys_and_zero_mouse(self):
        text = '<think></think><|action_start|>0 0 0 ; W ; W ; W ; W ; W ; W<|action_end|>'
        self.assertTrue(ev.inspect_action(text, 'W')['instruction_match'])
        self.assertFalse(ev.inspect_action(text, 'S')['instruction_match'])
        self.assertFalse(ev.inspect_action(text.replace('0 0 0', '1 0 0'), 'W')['instruction_match'])

    def test_plan_is_bounded_balanced_and_reproducible(self):
        cases = ev.plan(6, 20261023)
        self.assertEqual(cases, ev.plan(6, 20261023))
        self.assertEqual(len(cases), 60)
        self.assertEqual(len({c['id'] for c in cases}), 60)
        self.assertEqual(sum(c['experiment'] == 'load' for c in cases), 36)

    def test_summary_counts_http_failures_without_dropping_them(self):
        rows = [{'group': 'g', 'latency_ms': 10, 'error': None, 'parsed': True,
                 'strict_valid': True, 'instruction_match': True, 'usage': {'completion_tokens': 20}},
                {'group': 'g', 'latency_ms': 100, 'error': 'HTTP 400 context',
                 'parsed': False, 'strict_valid': False, 'instruction_match': False, 'usage': {}}]
        summary = ev.summarize(rows)['g']
        self.assertEqual(summary['requests'], 2)
        self.assertEqual(summary['errors'], 1)
        self.assertEqual(summary['latency_p95_ms'], 100)
        self.assertEqual(summary['instruction_match_rate'], .5)


if __name__ == '__main__':
    unittest.main()
