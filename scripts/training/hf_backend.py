"""Local-files-only Hugging Face RGB backend with explicit action-only loss.

The backend uses the checkpoint's chat template and actual executed history.
It imports no desktop control, UE, cloud, or model download code.
"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import time

from scripts.training.data import (SPLITS, audit_episodes, read_jsonl, require,
                                   sha256_file)
from scripts.training.formatting import assistant_labels, make_messages


def require_guard():
    path, token = os.environ.get('GAMESCALING_HEAVY_LOCK'), os.environ.get('GAMESCALING_HEAVY_TOKEN')
    require(path and token, 'run GPU work through scripts/resource_guard.py with the shared lock')
    metadata = json.loads(Path(path).read_text(encoding='utf-8'))
    require(metadata.get('token') == token, 'heavy-task lock owner mismatch')


def load_dataset(directory):
    """Reaudit original episodes and compare every normalized row, not just file hashes."""
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    require(sha256_file(manifest['source_path']) == manifest['source_sha256'], 'source hash changed')
    require(sha256_file(manifest['splits_path']) == manifest['splits_sha256'], 'split assignment hash changed')
    expected, _ = audit_episodes(read_jsonl(manifest['source_path']),
                                 json.loads(Path(manifest['splits_path']).read_text(encoding='utf-8')),
                                 manifest['data_root'], history_len=manifest['history_len'])
    result = {}
    for split in SPLITS:
        info = manifest['files'][split]
        require(info['path'] == f'{split}.jsonl', 'unexpected split file path')
        path = directory / info['path']
        require(sha256_file(path) == info['sha256'], 'converted data hash changed')
        rows = list(read_jsonl(path))
        require(rows == [row for row in expected if row['split'] == split],
                'converted rows differ from audited source actions/history')
        result[split] = rows
    return result, manifest, sha256_file(directory / 'manifest.json')


def model_identity(path):
    path = Path(path).resolve()
    require(path.is_dir(), 'model must be an existing local snapshot directory')
    required = ('config.json', 'model.safetensors', 'tokenizer.json',
                'tokenizer_config.json', 'chat_template.jinja', 'processor_config.json')
    for name in required:
        require((path / name).is_file(), f'local model missing {name}; downloads are disabled')
    config = json.loads((path / 'config.json').read_text(encoding='utf-8'))
    require(config['model_type'] == 'qwen3_5', 'this validated backend requires qwen3_5')
    optional = ('generation_config.json', 'preprocessor_config.json', 'video_preprocessor_config.json')
    files = {name: sha256_file(path / name) for name in (*required, *optional) if (path / name).is_file()}
    identity = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {'snapshot': path.name, 'model_type': config['model_type'], 'files': files,
            'fingerprint': identity}


def environment():
    import torch
    packages = {}
    for name in ('torch', 'transformers', 'peft', 'accelerate', 'tokenizers', 'safetensors', 'Pillow'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {'host': socket.gethostname(), 'platform': platform.platform(),
            'python': platform.python_version(), 'packages': packages,
            'torch_cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None}


def adapter_identity(path):
    if path is None:
        return None
    root = Path(path)
    files = {name: sha256_file(root / name)
             for name in ('adapter_config.json', 'adapter_model.safetensors')}
    return {'files': files, 'fingerprint': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}


class RGBBackend:
    def __init__(self, model_path, *, device='cuda', adapter=None, rank=None,
                 learning_seed=20261026):
        import random
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        require_guard()
        self.torch, self.device = torch, device
        self.identity = model_identity(model_path)
        self.adapter_identity = adapter_identity(adapter)
        self.model_path = Path(model_path).resolve()
        random.seed(learning_seed)
        torch.manual_seed(learning_seed)
        torch.set_num_threads(4)
        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True,
                                                        trust_remote_code=False)
        dtype = torch.bfloat16 if device == 'cuda' else torch.float32
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path, local_files_only=True, trust_remote_code=False,
            dtype=dtype, attn_implementation='sdpa').to(device)
        if adapter is not None:
            from peft import PeftModel
            require(Path(adapter).is_dir(), 'adapter must be a local directory')
            self.model = PeftModel.from_pretrained(self.model, str(adapter),
                                                   local_files_only=True, is_trainable=False)
        elif rank is not None:
            from peft import LoraConfig, get_peft_model
            config = LoraConfig(r=rank, lora_alpha=rank * 2, lora_dropout=0.,
                                target_modules=['q_proj', 'v_proj'], bias='none',
                                task_type='CAUSAL_LM')
            self.model = get_peft_model(self.model, config)
        self.model.eval()
        ending = self.processor.tokenizer.encode('<|action_end|>', add_special_tokens=False)
        require(ending and self.processor.tokenizer.decode(ending, skip_special_tokens=False)
                == '<|action_end|>', 'tokenizer must roundtrip the action ending')

    def synchronize(self):
        if self.device == 'cuda':
            self.torch.cuda.synchronize()

    def prepare(self, row, root, *, history_len=0, supervised=False):
        from PIL import Image
        started = time.perf_counter()
        messages, paths = make_messages(row, history_len=history_len)
        images = []
        for path in paths:
            with Image.open(Path(root) / path) as image:
                images.append(image.convert('RGB'))
        decoded = time.perf_counter()
        prompt = self.processor.apply_chat_template(messages, tokenize=False,
                                                     continue_final_message=True,
                                                     add_generation_prompt=False)
        encoded = self.processor(text=[prompt], images=images, return_tensors='pt')
        prompt_ids = encoded['input_ids'][0].tolist()
        target_count = 0
        if supervised:
            completion = '\n\n' + row['action']
            full = self.processor(text=[prompt + completion], images=images, return_tensors='pt')
            labels = assistant_labels(prompt_ids, full['input_ids'][0].tolist())
            target_count = len(labels) - len(prompt_ids)
            full['labels'] = self.torch.tensor([labels], dtype=self.torch.long)
            encoded = full
        self.synchronize()
        encoded_at = time.perf_counter()
        encoded = encoded.to(self.device)
        self.synchronize()
        transferred = time.perf_counter()
        return encoded, target_count, {'decode': (decoded - started) * 1000,
                                       'encode': (encoded_at - decoded) * 1000,
                                       'host_to_device': (transferred - encoded_at) * 1000}

    def loss(self, batch, target_count):
        # Keeping only target logits avoids allocating prompt_length * 248k vocabulary
        # logits. The preceding prompt token predicts the first completion token.
        labels = batch['labels'][:, -target_count:]
        inputs = {k: v for k, v in batch.items() if k != 'labels'}
        output = self.model(**inputs, use_cache=False, logits_to_keep=target_count + 1)
        logits = output.logits[:, :-1, :].float()
        require(logits.shape[1] == target_count, 'model did not honor logits_to_keep')
        return self.torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                                      labels.reshape(-1))

    def generate(self, batch, *, max_new_tokens=64):
        self.synchronize()
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(**batch, do_sample=False, max_new_tokens=max_new_tokens,
                                         stop_strings=['<|action_end|>'],
                                         tokenizer=self.processor.tokenizer, use_cache=True,
                                         pad_token_id=self.processor.tokenizer.pad_token_id)
        self.synchronize()
        inference_at = time.perf_counter()
        ids = output[0, batch['input_ids'].shape[-1]:]
        completion = self.processor.tokenizer.decode(ids, skip_special_tokens=False)
        return completion, len(ids), (inference_at - started) * 1000
