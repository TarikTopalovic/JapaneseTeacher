#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -x ".venv/bin/python" ]]; then
  PYTHON=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
else
  PYTHON="python"
fi

if ! "$PYTHON" -c "import uvicorn" >/dev/null 2>&1; then
  echo "uvicorn not found for $PYTHON. Installing project dependencies..."
  "$PYTHON" -m pip install -r requirements.txt
fi

# System CUDA path first (matches shell-level fix).
if [[ -d "/opt/cuda/lib64" ]]; then
  export LD_LIBRARY_PATH="/opt/cuda/lib64:${LD_LIBRARY_PATH:-}"
fi

# Make pip CUDA wheels discoverable at runtime (fixes missing libcublas/cudnn loads).
CUDA_LIB_DIRS="$("$PYTHON" - <<'PY'
import site
from pathlib import Path

dirs = []
for base in site.getsitepackages():
    nvidia = Path(base) / "nvidia"
    if not nvidia.exists():
        continue
    for lib_dir in nvidia.glob("*/lib"):
        if lib_dir.is_dir():
            dirs.append(str(lib_dir))
print(":".join(dirs))
PY
)"

if [[ -n "$CUDA_LIB_DIRS" ]]; then
  export LD_LIBRARY_PATH="${CUDA_LIB_DIRS}:${LD_LIBRARY_PATH:-}"
fi

# If runtime still missing cu12 libs required by faster-whisper/ctranslate2, install them.
if ! "$PYTHON" - <<'PY'
import ctypes
import os
from ctypes.util import find_library

if find_library("cublas"):
    raise SystemExit(0)

for name in ("libcublas.so.12", "libcublas.so"):
    try:
        ctypes.CDLL(name)
        raise SystemExit(0)
    except OSError:
        pass

raise SystemExit(1)
PY
then
  echo "CUDA runtime libs missing (libcublas). Installing cu12 runtime wheels..."
  "$PYTHON" -m pip install "nvidia-cublas-cu12>=12,<13" "nvidia-cudnn-cu12>=9,<10" "nvidia-cuda-runtime-cu12>=12,<13"

  CUDA_LIB_DIRS="$("$PYTHON" - <<'PY'
import site
from pathlib import Path

dirs = []
for base in site.getsitepackages():
    nvidia = Path(base) / "nvidia"
    if not nvidia.exists():
        continue
    for lib_dir in nvidia.glob("*/lib"):
        if lib_dir.is_dir():
            dirs.append(str(lib_dir))
print(":".join(dirs))
PY
)"
  if [[ -n "$CUDA_LIB_DIRS" ]]; then
    export LD_LIBRARY_PATH="${CUDA_LIB_DIRS}:${LD_LIBRARY_PATH:-}"
  fi
fi

"$PYTHON" -m uvicorn server:app --host 127.0.0.1 --port "${PORT:-8000}" --reload
