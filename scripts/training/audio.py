"""Player-observable audio windows: causal alignment, explicit masks and CPU features."""
from array import array
from pathlib import Path, PurePosixPath
import re
import sys
import wave

from scripts.training.data import require, sha256_file, text, timestamp


def missing(reason):
    return {'present': False, 'missing_reason': reason, 'waveform': [], 'sample_rate': None}


def read_audio(record, root, *, captured_ns, clock_id, max_age_ns=50_000_000):
    """Only return samples available before capture on the same episode clock.

    Structurally invalid or tampered data raises. Unavailable, stale or unaligned
    observations return an explicit mask without opening the audio file.
    """
    timestamp(captured_ns, 'RGB capture')
    timestamp(max_age_ns, 'audio max age')
    text(clock_id, 'RGB clock')
    require(isinstance(record, dict) and type(record.get('present')) is bool,
            'audio present must be a boolean')
    if not record['present']:
        return missing(text(record.get('missing_reason'), 'audio missing reason'))
    start = timestamp(record.get('window_start_ns'), 'audio start')
    end = timestamp(record.get('window_end_ns'), 'audio end')
    available = timestamp(record.get('available_ns'), 'audio availability')
    require(start < end <= available, 'audio window/availability order violated')
    if record.get('clock_id') != clock_id:
        return missing('clock_mismatch')
    if end > captured_ns:
        return missing('future_window')
    if available > captured_ns:
        return missing('not_yet_available')
    if captured_ns - end > max_age_ns:
        return missing('stale_window')
    name = text(record.get('path'), 'audio path')
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and '..' not in relative.parts and ':' not in name
            and '\\' not in name, 'audio path must stay under data root')
    root = Path(root).resolve()
    path = (root / name).resolve()
    require(path.is_relative_to(root), 'audio path escapes data root')
    digest = record.get('sha256')
    require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest), 'audio hash required')
    require(sha256_file(path) == digest, 'audio content hash mismatch')
    rate = record.get('sample_rate')
    require(type(rate) is int and 8000 <= rate <= 48000, 'unsupported sample rate')
    require(record.get('channels') == 1 and record.get('encoding') == 'pcm_s16le',
            'mono PCM16 WAV required')
    with wave.open(str(path), 'rb') as source:
        require(source.getnchannels() == 1 and source.getsampwidth() == 2
                and source.getcomptype() == 'NONE' and source.getframerate() == rate,
                'WAV format disagrees with metadata')
        frames = source.getnframes()
        require(0 < frames <= rate, 'audio window must be at most one second')
        require(abs((end - start) - frames * 1_000_000_000 / rate) <= 1_000_000_000 / rate,
                'audio window duration disagrees with sample count')
        raw = source.readframes(frames)
    require(len(raw) == frames * 2, 'truncated WAV sample data')
    samples = array('h')
    samples.frombytes(raw)
    if sys.byteorder != 'little':
        samples.byteswap()
    return {'present': True, 'missing_reason': None,
            'waveform': [value / 32768. for value in samples], 'sample_rate': rate,
            'window_start_ns': start, 'window_end_ns': end, 'available_ns': available,
            'sha256': digest}


def audio_features(observation):
    """Eight normalized spectral bands + RMS + zero crossings; mask remains separate."""
    import numpy as np
    if not observation['present']:
        return [0.] * 10
    samples = np.asarray(observation['waveform'], dtype=np.float32)
    require(samples.size > 0 and np.isfinite(samples).all(), 'finite nonempty waveform required')
    rms = float(np.sqrt(np.mean(samples ** 2)))
    if rms == 0:
        return [0.] * 10
    power = np.abs(np.fft.rfft(samples * np.hanning(len(samples)))) ** 2
    bands = np.asarray([float(chunk.sum()) for chunk in np.array_split(power, 8)])
    bands = bands / max(float(bands.sum()), 1e-12)
    crossing = float(np.mean((samples[:-1] >= 0) != (samples[1:] >= 0))) if len(samples) > 1 else 0.
    return bands.tolist() + [rms, crossing]
