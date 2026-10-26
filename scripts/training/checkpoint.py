"""Local checkpoints including trainable parameters, optimizer and exact RNG state."""
import json
from pathlib import Path
import random

from scripts.training.data import sha256_file


def save_checkpoint(path, model, optimizer, *, step, identity):
    import torch
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    trainables = {name: value.detach().cpu().clone() for name, value in model.named_parameters()
                  if value.requires_grad}
    state = {'schema_version': 1, 'step': step, 'identity': identity,
             'trainables': trainables, 'optimizer': optimizer.state_dict(),
             'python_rng': random.getstate(), 'torch_rng': torch.get_rng_state(),
             'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}
    torch.save(state, path / 'state.pt')
    # Written last: absent manifest means an interrupted/incomplete checkpoint.
    metadata = {'schema_version': 1, 'step': step, 'identity': identity,
                'state_sha256': sha256_file(path / 'state.pt'),
                'trainable_names': sorted(trainables)}
    (path / 'manifest.json').write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding='utf-8')


def restore_checkpoint(path, model, optimizer, *, identity):
    import torch
    path = Path(path)
    metadata = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
    if metadata['identity'] != identity:
        raise ValueError('checkpoint identity mismatch: model/data/optimizer configuration changed')
    if sha256_file(path / 'state.pt') != metadata['state_sha256']:
        raise ValueError('checkpoint content hash mismatch')
    state = torch.load(path / 'state.pt', map_location='cpu', weights_only=True)
    if state['identity'] != identity or state['step'] != metadata['step']:
        raise ValueError('checkpoint state/manifest identity mismatch')
    parameters = dict(model.named_parameters())
    names = {name for name, value in parameters.items() if value.requires_grad}
    if names != set(state['trainables']) or names != set(metadata['trainable_names']):
        raise ValueError('checkpoint trainable parameter identity mismatch')
    with torch.no_grad():
        for name in names:
            if parameters[name].shape != state['trainables'][name].shape:
                raise ValueError('checkpoint parameter shape mismatch')
            parameters[name].copy_(state['trainables'][name])
    optimizer.load_state_dict(state['optimizer'])
    random.setstate(state['python_rng'])
    torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng'] is not None:
        if not torch.cuda.is_available():
            raise ValueError('CUDA checkpoint requires original CUDA device topology')
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state['step']
