#!/usr/bin/env bash
# Print GPU / disk / python info (for run notes).

set -euo pipefail

echo "=== machine ==="
echo "date: $(date -Is 2>/dev/null || date)"
echo "host: $(hostname)"
echo "user: $(whoami)"
echo "pwd:  $(pwd)"
echo

echo "=== OS ==="
uname -a || true
if [[ -f /etc/os-release ]]; then
  . /etc/os-release
  echo "PRETTY_NAME=${PRETTY_NAME:-}"
fi
echo

echo "=== CPU / RAM / Disk ==="
command -v nproc >/dev/null && echo "cpus: $(nproc)" || true
free -h 2>/dev/null || true
df -h / "$HOME" 2>/dev/null || df -h
echo

echo "=== GPU ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi NOT FOUND"
fi
echo

echo "=== Python / Conda ==="
command -v conda >/dev/null && conda --version || echo "conda: none"
command -v python >/dev/null && python --version || true
command -v python3 >/dev/null && python3 --version || true
if command -v python >/dev/null 2>&1; then
  python - <<'PY' 2>/dev/null || true
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda_device", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch: not importable:", e)
PY
fi
echo

echo "=== Disk warning ==="
# Portable-ish: use df on /
avail_kb=$(df -k / | awk 'NR==2{print $4}')
if [[ -n "${avail_kb:-}" ]] && [[ "$avail_kb" -lt 20000000 ]]; then
  echo "WARNING: less than ~20GB free on / — Isaac may fail. Resize disk."
else
  echo "root free blocks(k): ${avail_kb:-unknown}"
fi

echo
echo "=== done ==="
echo "Next: follow docs/cloud_session_commands.md"
