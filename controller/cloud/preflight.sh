#!/usr/bin/env bash
# Read-only host, runtime, disk and GPU inventory. Do not print credentials/env.
set -u
hostname
uname -srmo
id -un
pwd
if [ -r /etc/os-release ]; then
  cat /etc/os-release
fi
nproc
free -h
df -h / /root/autodl-tmp
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
fi
for gs_tool in python python3 pip uv vllm conda git curl; do
  command -v "$gs_tool" || true
done
if command -v python >/dev/null 2>&1; then
  python - <<'PY'
import sys, platform, importlib.metadata
print('Python:', sys.version)
print('Executable:', sys.executable)
print('Machine:', platform.machine())
for package in ('torch', 'vllm', 'transformers', 'huggingface-hub', 'uv', 'Pillow', 'requests'):
    try:
        print(package, importlib.metadata.version(package))
    except importlib.metadata.PackageNotFoundError:
        print(package, 'not installed in this interpreter')
PY
fi
