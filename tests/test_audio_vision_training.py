import importlib.util
import unittest

from scripts.training.audio_vision_train import mask_for_step, training_source_paths


class AudioTrainingBoundaryTests(unittest.TestCase):
    def test_mask_dropout_is_step_recoverable_and_not_a_function_of_teacher(self):
        self.assertEqual([i for i in range(16) if mask_for_step(i)], [7, 15])
        with self.assertRaises(ValueError):
            mask_for_step(-1)
        names = {p.name for p in training_source_paths()}
        self.assertTrue({'audio_cue.py', 'audio_cue_data.py', 'audio_vision.py', 'audio.py',
                         'formatting.py', 'control_core.py', 'checkpoint.py'}.issubset(names))

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional torch CPU runtime required')
    def test_two_image_preparation_exposes_no_teacher_and_records_actual_mask(self):
        import tempfile
        from pathlib import Path
        from PIL import Image
        import torch
        from scripts.training.audio_vision import prepare_audio

        class Batch(dict):
            def to(self, device):
                return self

        class Processor:
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                return 'safe-prompt'

            def __call__(self, *, text, images, return_tensors):
                self.images = images
                return Batch(input_ids=torch.tensor([[1, 2, 3]]))

        class Backend:
            processor, device = Processor(), 'cpu'

            def synchronize(self):
                pass

        with tempfile.TemporaryDirectory() as folder:
            Image.new('RGB', (256, 256)).save(Path(folder)/'rgb.png')
            row = {'system_prompt': 'fixed rules', 'rgb': {'path': 'rgb.png'},
                   'audio': {'present': False, 'missing_reason': 'unavailable'},
                   'captured_ns': 100, 'clock_id': 'safe-clock', 'action': 'SECRET-TARGET'}
            backend = Backend()
            _, evidence = prepare_audio(backend, row, folder, folder)
            self.assertFalse(evidence['present'])
            self.assertEqual(evidence['image_count'], 2)
            self.assertEqual(len(backend.processor.images), 2)
            self.assertNotIn('SECRET', str(backend.processor.messages))


if __name__ == '__main__':
    unittest.main()
