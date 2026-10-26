"""Train a tiny CPU audio/RGB fusion model on generated tones, not GameScaling.

Synthetic classifier results validate modality plumbing only. No game is run,
no pretrained policy is modified, and no environment reward/RL is claimed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time
import wave

from scripts.training.audio import audio_features, read_audio
from scripts.training.data import require, sha256_file


def make_fusion(rgb):
    import torch

    class Fusion(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.rgb = base.requires_grad_(False)
            self.audio = torch.nn.Sequential(torch.nn.Linear(10, 32), torch.nn.Tanh(),
                                              torch.nn.Linear(32, 2))

        def forward(self, pixels, features, present):
            require(present.dtype == torch.bool and present.shape == pixels.shape[:1],
                    'one boolean audio mask per RGB example required')
            audio = self.audio(features)
            return self.rgb(pixels) + torch.where(present[:, None], audio, torch.zeros_like(audio))

    return Fusion(rgb)


def generate(root, seed):
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(seed)
    samples = []
    for split, count in (('train', 64), ('validation', 32), ('test', 64)):
        for i in range(count):
            key = i % 2
            sample_id = f'synthetic-audio-{split}-v1-{i:03}'
            # Grayscale distractors are independent of target; no RGB color cue here.
            shade = int(rng.integers(60, 190))
            image = np.full((32, 32, 3), shade, dtype=np.uint8)
            image[0, 0] = rng.integers(0, 256, size=3, dtype=np.uint8)
            rgb_path = root / f'{sample_id}.png'
            Image.fromarray(image).save(rgb_path)
            frequency = float(rng.uniform(400, 500) if key == 0 else rng.uniform(1700, 1900))
            phase = rng.uniform(0, 2 * np.pi)
            times = np.arange(3200) / 16000
            signal = .35 * np.sin(2 * np.pi * frequency * times + phase) + rng.normal(0, .008, len(times))
            pcm = np.clip(signal * 32768, -32768, 32767).astype('<i2')
            audio_path = root / f'{sample_id}.wav'
            with wave.open(str(audio_path), 'wb') as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(pcm.tobytes())
            record = {'present': True, 'path': audio_path.name, 'sha256': sha256_file(audio_path),
                      'sample_rate': 16000, 'channels': 1, 'encoding': 'pcm_s16le',
                      'clock_id': sample_id, 'window_start_ns': 800_000_000,
                      'window_end_ns': 1_000_000_000, 'available_ns': 1_000_000_000}
            observation = read_audio(record, root, captured_ns=1_000_000_000, clock_id=sample_id)
            samples.append({'sample_id': sample_id, 'game_id': f'synthetic-audio-{split}-v1',
                            'episode_id': sample_id, 'split': split, 'kind': 'synthetic',
                            'source': 'locally generated RGB distractor and sine tone',
                            'training_allowed': True, 'license': 'locally-generated-no-third-party-assets',
                            'rgb': {'path': rgb_path.name, 'sha256': sha256_file(rgb_path),
                                    'pixel_sha256': hashlib.sha256(image.tobytes()).hexdigest()},
                            'audio': record, 'captured_ns': 1_000_000_000, 'clock_id': sample_id,
                            'rgb_features': (image.mean(axis=(0, 1)) / 255).tolist(),
                            'audio_features': audio_features(observation),
                            'target_class': key, 'target_action_key': ('W', 'S')[key]})
    for field in ('episode_id',):
        require(len({s[field] for s in samples}) == len(samples), 'duplicate synthetic episodes')
    for field in ('pixel_sha256',):
        require(len({s['rgb'][field] for s in samples}) == len(samples), 'duplicate synthetic RGB')
    require(len({s['audio']['sha256'] for s in samples}) == len(samples), 'duplicate synthetic audio')
    path = root / 'samples.jsonl'
    path.write_text(''.join(json.dumps(s) + '\n' for s in samples), encoding='utf-8')
    return samples, sha256_file(path)


def run(output, *, seed=20261026, steps=200):
    import numpy as np
    import torch
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    random.seed(seed)
    root = output / 'generated'
    root.mkdir()
    samples, digest = generate(root, seed)
    rgb = torch.nn.Sequential(torch.nn.Linear(3, 16), torch.nn.Tanh(), torch.nn.Linear(16, 2))
    # Independent synthetic RGB color pretraining; this is a small baseline, not 0.8B.
    rng = np.random.default_rng(seed + 1)
    labels = torch.arange(128) % 2
    colors = rng.uniform(.02, .15, size=(128, 3)).astype('float32')
    colors[np.arange(128), np.where(labels.numpy() == 0, 0, 2)] += .7
    colors = torch.tensor(colors)
    optimizer = torch.optim.Adam(rgb.parameters(), lr=.01)
    for _ in range(150):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(rgb(colors), labels)
        loss.backward()
        optimizer.step()
    heldout_colors = colors.flip(0) * .96 + .01
    heldout_labels = labels.flip(0)
    with torch.inference_mode():
        retention_reference = rgb(heldout_colors).argmax(-1)
    model = make_fusion(rgb)
    optimizer = torch.optim.AdamW(model.audio.parameters(), lr=.01)
    train = [s for s in samples if s['split'] == 'train']
    pixels = torch.tensor([s['rgb_features'] for s in train], dtype=torch.float32)
    features = torch.tensor([s['audio_features'] for s in train], dtype=torch.float32)
    targets = torch.tensor([s['target_class'] for s in train])
    present = torch.ones(len(train), dtype=torch.bool)
    initial_loss = None
    for step in range(steps):
        if time.perf_counter() - started > 60:
            raise RuntimeError('CPU modality smoke exceeded its 60-second internal budget')
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(pixels, features, present), targets)
        if initial_loss is None:
            initial_loss = loss.item()
        require(torch.isfinite(loss).item(), 'nonfinite fusion loss')
        loss.backward()
        optimizer.step()
    model.eval()
    results = {}
    with torch.inference_mode():
        for split in ('train', 'validation', 'test'):
            rows = [s for s in samples if s['split'] == split]
            x = torch.tensor([s['rgb_features'] for s in rows], dtype=torch.float32)
            a = torch.tensor([s['audio_features'] for s in rows], dtype=torch.float32)
            y = torch.tensor([s['target_class'] for s in rows])
            mask = torch.ones(len(rows), dtype=torch.bool)
            absent = ~mask
            # Every adjacent pair has opposite tone labels. Swapping makes the
            # counterfactual deterministic, with no random permutation luck.
            swap = torch.arange(len(rows)).reshape(-1, 2).flip(1).reshape(-1)
            noise = torch.tensor(rng.normal(0, .3, size=a.shape), dtype=torch.float32)
            variants = {'rgb_only': (a, absent), 'aligned_audio': (a, mask),
                        'missing_audio': (torch.zeros_like(a), absent),
                        'swapped_audio': (a[swap], mask), 'noise_features': (noise, mask)}
            scores = {name: float((model(x, aa, mm).argmax(-1) == y).float().mean())
                      for name, (aa, mm) in variants.items()}
            # Shift audio timestamps into the future, run the real aligner, and mask.
            shifted = []
            for row in rows:
                metadata = {**row['audio'], 'window_start_ns': 900_000_000,
                            'window_end_ns': 1_100_000_000, 'available_ns': 1_100_000_000}
                shifted.append(read_audio(metadata, root, captured_ns=row['captured_ns'],
                                          clock_id=row['clock_id'])['present'])
            shifted_mask = torch.tensor(shifted, dtype=torch.bool)
            scores['future_audio_masked'] = float((model(x, torch.zeros_like(a), shifted_mask).argmax(-1) == y).float().mean())
            results[split] = {'samples': len(rows), 'class_accuracy': scores,
                              'future_audio_present_count': sum(shifted)}
        zeros = torch.zeros(len(heldout_colors), 10)
        retained = model(heldout_colors, zeros, torch.zeros(len(zeros), dtype=torch.bool)).argmax(-1)
    torch.save({'model': model.state_dict(), 'seed': seed, 'steps': steps}, output / 'fusion.pt')
    result = {'schema_version': 1, 'seed': seed, 'device': 'cpu', 'torch': torch.__version__,
              'started_utc': started_utc, 'finished_utc': datetime.now(timezone.utc).isoformat(), 'steps': steps,
              'initial_train_loss': initial_loss, 'final_train_loss': loss.item(),
              'trainable_audio_parameters': sum(p.numel() for p in model.audio.parameters()),
              'dataset_sha256': digest, 'checkpoint_sha256': sha256_file(output / 'fusion.pt'),
              'results': results, 'rgb_synthetic_retention': {
                  'samples': len(retained), 'output_agreement': float((retained == retention_reference).float().mean()),
                  'class_accuracy': float((retained == heldout_labels).float().mean())},
              'elapsed_seconds': time.perf_counter() - started,
              'evidence_level': 'tiny CPU synthetic audio fusion smoke; no GameScaling audio or game ability claim'}
    (output / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261026)
    parser.add_argument('--steps', type=int, choices=range(1, 501), default=200)
    args = parser.parse_args()
    run(args.output, seed=args.seed, steps=args.steps)


if __name__ == '__main__':
    main()
