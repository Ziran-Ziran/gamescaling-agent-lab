"""Fixed inference targets. Neither jobs nor command-line flags accept arbitrary URLs."""
from pathlib import Path
import sys

from controller.wsl_bridge import WSLBridge

BACKENDS = ('local', 'cloud')
_TARGETS = {
    'local': {'backend': 'local', 'base_url': 'http://127.0.0.1:8000/v1',
              'model_id': 'gamescaling', 'model_root': 'THUSI-Lab/GameScaling-0.8B-S3'},
    'cloud': {'backend': 'cloud', 'base_url': 'http://127.0.0.1:18080/v1',
              'model_id': 'gamescaling', 'model_root': 'THUSI-Lab/GameScaling-9B-S3'},
}
_WINDOWS_REPO = r'\\wsl.localhost\Ubuntu\home\ziran\gamescaling-lab\GameScaling'


def backend_target(backend='local'):
    if type(backend) is not str or backend not in BACKENDS:
        raise ValueError('backend must be local or cloud')
    return _TARGETS[backend].copy()


def create_bridge(backend='local', *, timeout=15):
    target = backend_target(backend)
    if backend == 'local':
        return WSLBridge(timeout=timeout)
    return WSLBridge(command=[sys.executable, '-u',
        str(Path(__file__).with_name('policy_worker.py')), '--repo', _WINDOWS_REPO,
        '--base-url', target['base_url']], timeout=timeout)


def validate_health(health, backend='local'):
    target = backend_target(backend)
    try:
        models = health['models']['data']
        valid = isinstance(models, list) and any(isinstance(row, dict) and
            row.get('id') == target['model_id'] and row.get('root') == target['model_root']
            for row in models)
    except (KeyError, TypeError):
        valid = False
    if not valid:
        raise RuntimeError(f"Expected {target['backend']} model id={target['model_id']} "
                           f"root={target['model_root']} at {target['base_url']}")
    return health
