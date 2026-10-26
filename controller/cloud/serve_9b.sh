#!/usr/bin/env bash
# Run only after the checkpoint downloader has verified the pinned snapshot.
set -euo pipefail

LAB_ROOT=/root/autodl-tmp/gamescaling-lab
export PATH="$LAB_ROOT/.venv/bin:$PATH"
export HF_HOME="$LAB_ROOT/cache/huggingface"
export HF_HUB_OFFLINE=1
export XDG_CACHE_HOME="$LAB_ROOT/cache"
export TORCHINDUCTOR_CACHE_DIR="$LAB_ROOT/cache/torchinductor"
export TRITON_CACHE_DIR="$LAB_ROOT/cache/triton"
export TMPDIR="$LAB_ROOT/tmp"
mkdir -p "$TMPDIR"
REVISION=$(python - "$LAB_ROOT/download-manifest.json" <<'PY'
import json, sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text())
assert manifest['repo'] == 'THUSI-Lab/GameScaling-9B-S3'
assert manifest['status'] == 'verified', 'Run download_9b.py and wait for verification first'
assert len(manifest['revision']) == 40
assert Path(manifest['snapshot']).is_dir()
print(manifest['revision'])
PY
)

if ! command -v vllm >/dev/null 2>&1; then
  echo 'No vllm command in the selected environment; inspect/install the approved environment first.' >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo 'No nvidia-smi; inspect the remote GPU environment before serving.' >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv
vllm --version

# Keep the API on remote loopback. Reach it through an SSH tunnel.
# Match the local image/context settings first; larger-image tests are separate.
exec vllm serve THUSI-Lab/GameScaling-9B-S3 \
  --revision "$REVISION" \
  --host 127.0.0.1 \
  --port 18000 \
  --served-model-name gamescaling \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.80 \
  --limit-mm-per-prompt '{"image": 6, "video": 0}' \
  --mm-processor-kwargs '{"max_pixels": 230400}' \
  --trust-remote-code
