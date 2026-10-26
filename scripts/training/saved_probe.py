"""Generate from one saved official/example image; no trajectory labels are invented.

Reference comparison is output agreement only, not game success or ability retention.
"""
import argparse
import json
from pathlib import Path

from scripts.training.data import sha256_file
from scripts.training.hf_backend import RGBBackend, environment
from scripts.training.metrics import action_signature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--prompt', type=Path, required=True)
    parser.add_argument('--source-description', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--max-new-tokens', type=int, choices=range(1, 257), default=96)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text(encoding='utf-8')) if args.reference else None
    image = args.image.resolve()
    prompt = args.prompt.read_text(encoding='utf-8')
    identity = {'image_sha256': sha256_file(image), 'prompt_sha256': sha256_file(args.prompt),
                'reasoning': False, 'history_len': 0, 'max_new_tokens': args.max_new_tokens}
    if reference and reference['probe_identity'] != identity:
        parser.error('reference image/prompt/generation identity mismatch')
    args.output.mkdir(parents=True, exist_ok=False)
    backend = RGBBackend(args.model, adapter=args.adapter)
    if reference and reference['model']['fingerprint'] != backend.identity['fingerprint']:
        parser.error('reference base model identity mismatch')
    backend.model.eval()
    row = {'system_prompt': prompt, 'rgb': {'path': image.name}, 'history': []}
    batch, _, timing = backend.prepare(row, image.parent)
    completion, count, inference_ms = backend.generate(batch, max_new_tokens=args.max_new_tokens)
    signature = action_signature(completion)
    agreement = (signature == action_signature(reference['completion'])
                 if reference and signature is not None and action_signature(reference['completion']) is not None
                 else None)
    result = {'schema_version': 1, 'probe_identity': identity, 'model': backend.identity,
              'adapter': backend.adapter_identity,
              'environment': environment(), 'source_description': args.source_description,
              'completion': completion, 'generated_tokens': count, 'strict_valid': signature is not None,
              'reference_action_agreement': agreement, 'ground_truth': None,
              'timing_ms': {**timing, 'inference': inference_ms},
              'evidence_level': 'single saved-image output probe; agreement is not gameplay retention',
              'peak_cuda_allocated_mib': backend.torch.cuda.max_memory_allocated() / 1024 ** 2}
    (args.output / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
