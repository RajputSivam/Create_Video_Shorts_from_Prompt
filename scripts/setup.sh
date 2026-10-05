#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -x .venv/bin/python ]]; then
  python3.12 -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r backend/requirements.txt
if [[ "${INSTALL_ML:-0}" == "1" ]]; then
  .venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
fi

mkdir -p outputs jobs cache/huggingface cache/whisper models/voices assets/music assets/sfx assets/fonts
if ! command -v ollama >/dev/null; then
  echo "Install Ollama from https://ollama.com/download/linux before pulling the local LLM."
elif [[ "${DOWNLOAD_MODELS:-0}" == "1" ]]; then
  ollama pull qwen2.5:7b
fi

python - <<'PY'
from pathlib import Path
from urllib.request import urlretrieve

files = {
    "models/voices/en_US-amy-medium.onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx",
    "models/voices/en_US-amy-medium.onnx.json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json",
    "models/voices/en_US-joe-medium.onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/joe/medium/en_US-joe-medium.onnx",
    "models/voices/en_US-joe-medium.onnx.json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/joe/medium/en_US-joe-medium.onnx.json",
    "models/voices/hi_IN-priyamvada-medium.onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/priyamvada/medium/hi_IN-priyamvada-medium.onnx",
    "models/voices/hi_IN-priyamvada-medium.onnx.json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/priyamvada/medium/hi_IN-priyamvada-medium.onnx.json",
    "models/voices/hi_IN-rohan-medium.onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/rohan/medium/hi_IN-rohan-medium.onnx",
    "models/voices/hi_IN-rohan-medium.onnx.json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/rohan/medium/hi_IN-rohan-medium.onnx.json",
}
for filename, url in files.items():
    target = Path(filename)
    if not target.exists():
        print(f"Downloading {target}")
        urlretrieve(url, target)
PY

if [[ "${DOWNLOAD_MODELS:-0}" == "1" ]]; then
  .venv/bin/python -m workers.download_models
fi

echo "Install CUDA-enabled PyTorch using INSTALL_ML=1 scripts/setup.sh on NVIDIA systems."
echo "Run: npm run doctor"
