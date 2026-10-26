"""Explicit audio-derived visual tokens through the existing two-image encoder."""
from pathlib import Path
import hashlib
import time

from scripts.training.audio import read_audio
from scripts.training.audio_cue import spectral_image


def observation_messages(row, spectral_path, *, present, rgb_only=False):
    # Only fixed player rules and sensor mask enter text. No target, frequency
    # measurement, source path, reset, teacher label or engine annotation.
    content = [{'type': 'text', 'text': 'current game screen'}, {'type': 'image'}]
    paths = [row['rgb']['path']]
    if not rgb_only:
        content += [{'type': 'text', 'text': f'current audio spectrogram; present={int(present)}'}, {'type': 'image'}]
        paths.append(str(spectral_path))
    else:
        content.append({'type': 'text', 'text': 'current audio unavailable; present=0'})
    return [{'role': 'system', 'content': row['system_prompt']},
            {'role': 'user', 'content': content},
            {'role': 'assistant', 'content': '<think>\n\n</think>'}], paths


def prepare_audio(backend, row, root, output, *, force_missing=False, rgb_only=False, rgb_blank=False):
    from PIL import Image
    started = time.perf_counter()
    audio = read_audio({'present': False, 'missing_reason': 'controlled_mask'} if force_missing or rgb_only else row['audio'],
                       root, captured_ns=row['captured_ns'], clock_id=row['clock_id'])
    spectral = spectral_image(audio)
    messages, _ = observation_messages(row, 'spectral-image', present=audio['present'], rgb_only=rgb_only)
    with Image.open(Path(root)/row['rgb']['path']) as source:
        rgb = source.convert('RGB')
    if rgb_blank:
        rgb = Image.new('RGB', rgb.size, (0, 0, 0))
    images = [rgb] if rgb_only else [rgb, spectral]
    def pixel_hash(image):
        return hashlib.sha256(f'RGB:{image.width}:{image.height}:'.encode()+image.tobytes()).hexdigest()
    prompt = backend.processor.apply_chat_template(messages, tokenize=False, continue_final_message=True,
                                                    add_generation_prompt=False)
    encoded = backend.processor(text=[prompt], images=images, return_tensors='pt')
    backend.synchronize()
    encoded = encoded.to(backend.device)
    backend.synchronize()
    return encoded, {'present': audio['present'], 'missing_reason': audio['missing_reason'],
                     'image_count': len(images), 'audio_sha256': audio.get('sha256'),
                     'input_rgb_pixel_sha256': pixel_hash(rgb),
                     'input_spectral_pixel_sha256': pixel_hash(spectral) if not rgb_only else None,
                     'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                     'input_tokens': encoded['input_ids'].shape[-1],
                     'prepare_ms': (time.perf_counter()-started)*1000}
