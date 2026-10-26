import math
from types import SimpleNamespace
import unittest

from controller.control_core import to_action_text
from scripts.training.action_head import (FORCED_PREFIX, LABELS, FirstKeyBackend, choose_class,
                                          first_key_contract, forward_action_logits, template_action)

try:
    import torch
except ImportError:
    torch = None


class Tokenizer:
    def __init__(self, *, unstable=False, multi=False, duplicate=False):
        self.unstable, self.multi, self.duplicate = unstable, multi, duplicate

    def decode(self, ids, **kwargs):
        return 'tail'

    def encode(self, text, **kwargs):
        if text == 'tail':
            return [1, 2]
        if text == 'tail' + FORCED_PREFIX:
            return [1, 9, 3] if self.unstable else [1, 2, 3]
        extensions = [' W', ' A', ' S', ' D', ' ;']
        extension = text[len('tail' + FORCED_PREFIX):]
        index = extensions.index(extension)
        return [1, 2, 3, 10 if self.duplicate else 10 + index] + ([99] if self.multi else [])


class FirstKeyTests(unittest.TestCase):
    @unittest.skipIf(torch is None, 'optional torch CPU dependency unavailable')
    def test_shared_forward_allows_training_gradients_without_forcing_inference_mode(self):
        parameter = torch.nn.Parameter(torch.tensor(1.))
        backend = SimpleNamespace(torch=torch, processor=SimpleNamespace(tokenizer=Tokenizer()))
        backend.model = lambda **inputs: SimpleNamespace(logits=(parameter * torch.arange(20.)).reshape(1, 1, 20))
        logits, candidates = forward_action_logits(backend, {'input_ids': torch.tensor([[1, 2]])})
        loss = -torch.nn.functional.log_softmax(logits[candidates].float(), dim=-1)[0]
        loss.backward()
        self.assertTrue(torch.isfinite(parameter.grad))
        self.assertNotEqual(float(parameter.grad), 0.)

    def test_contract_requires_exact_prefix_and_unique_single_token_classes(self):
        prefix, candidates = first_key_contract(Tokenizer(), [1, 2])
        self.assertEqual(prefix, [3])
        self.assertEqual(candidates, [10, 11, 12, 13, 14])
        for kwargs in ({'unstable': True}, {'multi': True}, {'duplicate': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                first_key_contract(Tokenizer(**kwargs), [1, 2])

    def test_selection_keeps_raw_scores_and_does_not_call_them_confidence(self):
        selected, record = choose_class([-.5, -1., 3., 0., -2.])
        self.assertEqual(selected, 'S')
        self.assertEqual(record['raw_class_logits']['S'], 3.)
        self.assertAlmostEqual(sum(record['restricted_softmax_not_calibrated'].values()), 1.)
        self.assertNotIn('confidence', record)
        for invalid in ([1.], [0., 0., math.nan, 0., 0.], [0., math.inf, 0., 0., 0.]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                choose_class(invalid)

    def test_every_template_is_safe_but_restricts_expressive_action_space(self):
        for label in LABELS:
            action = template_action(label)
            self.assertEqual(action.mouse, (0., 0., 0.))
            self.assertEqual(len(set(action.steps)), 1)
            self.assertIn('<|action_end|>', to_action_text(action))
        self.assertEqual(template_action('noop').steps, ((),) * 6)
        with self.assertRaises(ValueError):
            template_action('W D')

    @unittest.skipIf(torch is None, 'optional torch CPU dependency unavailable')
    def test_prefix_forward_preserves_original_batch_and_extends_multimodal_masks(self):
        backend = object.__new__(FirstKeyBackend)
        backend.torch, backend.device = torch, 'cpu'
        backend.processor = SimpleNamespace(tokenizer=Tokenizer())
        batch = {'input_ids': torch.tensor([[1, 2]]), 'attention_mask': torch.tensor([[1, 1]]),
                 'mm_token_type_ids': torch.tensor([[1, 0]]),
                 'pixel_values': torch.ones(1, 3), 'image_grid_thw': torch.tensor([[1, 1, 1]])}
        original = {key: value.clone() for key, value in batch.items()}

        def model(**inputs):
            self.assertFalse(inputs['use_cache'])
            self.assertEqual(inputs['logits_to_keep'], 1)
            self.assertEqual(inputs['input_ids'].tolist(), [[1, 2, 3]])
            self.assertEqual(inputs['attention_mask'].tolist(), [[1, 1, 1]])
            self.assertEqual(inputs['mm_token_type_ids'].tolist(), [[1, 0, 0]])
            self.assertTrue(torch.equal(inputs['pixel_values'], original['pixel_values']))
            logits = torch.zeros(1, 1, 20)
            logits[0, 0, 0], logits[0, 0, 12] = 9., 3.
            return SimpleNamespace(logits=logits)

        backend.model = model
        action, generated_tokens, elapsed = backend.generate(batch)
        self.assertIn('; S ; S ;', action)
        self.assertEqual(generated_tokens, 0)
        self.assertGreater(elapsed, 0.)
        self.assertFalse(backend.last_head_record['unrestricted_top_is_allowed'])
        self.assertEqual(backend.last_head_record['predicted_tokens'], 1)
        self.assertTrue(all(torch.equal(batch[key], value) for key, value in original.items()))


if __name__ == '__main__':
    unittest.main()
