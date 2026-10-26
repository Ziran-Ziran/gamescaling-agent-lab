"""Versioned procedural audio sensor and dual-candidate RGB observation projection.

The underlying point-grid physics/reward v1 is unchanged. This wrapper hides which
of two colored candidates is active; its virtual tone is the player sensor cue.
"""
import hashlib
import io
import math
from pathlib import Path
import random
import struct
import time
import wave

from scripts.training.data import require, sha256_file
from scripts.training.headless import HeadlessGame

PROJECTION = 'audio-cue-dual-color-v1'
SENSOR = 'virtual-tone-pcm16-16khz-200ms-v1'
GAMES = {v: f'audio-cue-{v}-v1' for v in ('reach', 'hazard', 'slalom')}
COLORS = ((235, 200, 20), (20, 210, 210))


def waveform_bytes(context, tone, *, kind='tone'):
    """Deterministic full-scale-relative PCM; no playback or microphone access."""
    require(type(context) is int and context >= 0 and tone in (0, 1), 'invalid cue identity')
    require(kind in ('tone', 'noise', 'silence'), 'unsupported virtual sensor waveform')
    rng = random.Random(context)
    phase = rng.random() * 2 * math.pi
    frequency = (500, 2000)[tone]
    samples = [round(8192 * math.sin(2 * math.pi * frequency * i / 16000 + phase))
               if kind == 'tone' else rng.randrange(-10033, 10034) if kind == 'noise' else 0
               for i in range(3200)]
    target = io.BytesIO()
    with wave.open(target, 'wb') as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(struct.pack('<3200h', *samples))
    return target.getvalue()


def projected_rgb(state, spec):
    """Both candidate colors are observable; the privileged active goal is not."""
    from PIL import ImageDraw
    game = AudioCueGame(**spec)
    require(state['palette'] == game.base.palette, 'projection palette/identity mismatch')
    image = HeadlessGame.render_snapshot(state)
    draw = ImageDraw.Draw(image)
    # Both targets overwrite the base green goal completely. This must remain
    # identical for opposite active-goal states at the same player position.
    for side, (x, y) in enumerate(game.candidates):
        color = COLORS[side ^ spec['colors']]
        draw.rectangle((16 + x * 32 + 3, 16 + y * 32 + 3,
                        16 + (x + 1) * 32 - 3, 16 + (y + 1) * 32 - 3), fill=color)
    # Preserve avatar visibility if it stands on a candidate on later frames.
    x, y = [16 + v * 32 for v in state['player']]
    draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=(25, 60, 230))
    return image


class AudioCueGame:
    def __init__(self, *, context, colors, tone, variant='reach', clock_ns=time.perf_counter_ns,
                 sleep=time.sleep):
        require(type(context) is int and context >= 0 and type(colors) is int and colors in (0, 1)
                and type(tone) is int and tone in (0, 1) and variant in GAMES, 'invalid audio-cue reset')
        self.spec = {'context': context, 'colors': colors, 'tone': tone, 'variant': variant}
        self.clock_ns, self.sleep = clock_ns, sleep
        self.candidates = ((1, 3), (5, 3)) if variant == 'slalom' else ((2, 3), (4, 3))
        side = colors ^ tone
        layout = {'goal': self.candidates[side], 'hazards': {(3, 2), (3, 4)} if variant == 'hazard' else set()}
        if variant == 'slalom':
            layout['walls'] = {(2, 3), (4, 3)}
        self.base = HeadlessGame(variant='hazard' if variant == 'hazard' else 'reach', seed=context,
                                 max_steps=8 if variant == 'slalom' else 1, layout=layout,
                                 clock_ns=clock_ns, sleep=sleep)
        self.base.episode_id = f'{GAMES[variant]}-context-{context}-colors-{colors}-tone-{tone}'
        self.base.clock_id = self.base.episode_id + '-perf-counter-ns'

    def render(self):
        return projected_rgb(self.base.snapshot(), self.spec)

    def observe(self, root):
        root = Path(root)
        # Warm lazy renderer imports before the sensor window. The paused world
        # stays fixed, but the actual RGB snapshot is taken after availability.
        self.render()
        raw = waveform_bytes(self.spec['context'], self.spec['tone'])
        start = self.clock_ns()
        remaining = (start + 200_000_000 - self.clock_ns()) / 1e9
        if remaining > 0:
            self.sleep(remaining)
        end = start + 200_000_000
        available = self.clock_ns()  # Sensor samples are available in memory.
        image = self.render()
        captured = self.clock_ns()
        name = f'{self.base.episode_id}-{self.base.step_count:03}.wav'
        path = root / name
        require(not path.exists(), 'audio sensor output already exists')
        path.write_bytes(raw)
        return {'image': image, 'captured_ns': captured, 'frame_id': self.base.step_count,
                'audio': {'present': True, 'path': name, 'sha256': sha256_file(path),
                          'sample_rate': 16000, 'channels': 1, 'encoding': 'pcm_s16le',
                          'clock_id': self.base.clock_id, 'window_start_ns': start,
                          'window_end_ns': end, 'available_ns': available,
                          'sensor_version': SENSOR}}

    @staticmethod
    def system_prompt():
        return ('You control the blue point avatar in a paused grid. Two colored candidate targets are '
                'yellow and cyan. Only one is active. A current 200ms virtual sound identifies its color: '
                '500Hz means yellow, 2000Hz means cyan. The second image, when present=1, is its fixed '
                'spectrogram, time left-to-right and frequency increasing bottom-to-top, range 0..4000Hz. '
                'If present=0 the sound is unavailable; never infer a cue from a blank image. '
                'Choose the indicated color using its current RGB position. Dark cells block movement, '
                'red cells are lethal. W up, A left, S down, D right; holding for six substeps moves one '
                'cell in 200ms. Mouse/scroll zero. Output only the official six-substep action block.')


def spectral_image(observation):
    """Fixed STFT, no learned encoding or target/frequency metadata in the image."""
    import numpy as np
    from PIL import Image
    if not observation['present']:
        return Image.new('RGB', (256, 256), (0, 0, 0))
    samples = np.asarray(observation['waveform'], dtype=np.float64)
    require(observation['sample_rate'] == 16000 and len(samples) == 3200
            and np.isfinite(samples).all() and np.max(np.abs(samples)) <= 1.,
            'spectrogram requires finite 200ms16kHz normalized PCM')
    windows = np.stack([samples[i:i+256] for i in range(0, len(samples)-255, 128)])
    amplitude = np.abs(np.fft.rfft(windows * np.hanning(256), axis=1)) / 128
    db = 20 * np.log10(np.maximum(amplitude[:, :65], 1e-12))
    pixels = np.rint(np.clip((db + 80) / 80, 0, 1) * 255).astype('uint8').T[::-1]
    return Image.fromarray(pixels).resize((256, 256), Image.Resampling.NEAREST).convert('RGB')
