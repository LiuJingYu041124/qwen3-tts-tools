#!/bin/zsh
set -e

cd "$(dirname "$0")"

APP_VENV="$HOME/.venvs/chinese-dub-studio"
TTS_VENV="$HOME/.venvs/mlx-qwen3-tts"
WHISPER_MODEL="$PWD/models/ggml-small.bin"
VOICE_DESIGN_MODEL="mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
VOICE_CLONE_MODEL="mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"

if [ -x /opt/homebrew/bin/brew ]; then
  eval "$(/opt/homebrew/bin/brew shellenv)"
elif [ -x /usr/local/bin/brew ]; then
  eval "$(/usr/local/bin/brew shellenv)"
fi
if ! command -v brew >/dev/null 2>&1; then
  echo "缺少 Homebrew，请先从 https://brew.sh 安装。"
  read "?按回车键关闭窗口..."
  exit 1
fi

BREW_PREFIX="$(brew --prefix)"
FULL_FFMPEG="$BREW_PREFIX/opt/ffmpeg-full/bin/ffmpeg"
FULL_FFPROBE="$BREW_PREFIX/opt/ffmpeg-full/bin/ffprobe"

if ! command -v python3.12 >/dev/null 2>&1; then
  echo "正在安装 Python 3.12..."
  brew install python@3.12
fi
PYTHON="$(command -v python3.12)"

if ! command -v whisper-cli >/dev/null 2>&1; then
  echo "正在安装 Whisper CLI..."
  brew install whisper-cpp
fi
if ! command -v ollama >/dev/null 2>&1; then
  echo "正在安装 Ollama..."
  brew install ollama
fi
if ! command -v curl >/dev/null 2>&1; then
  echo "缺少命令：curl"
  read "?按回车键关闭窗口..."
  exit 1
fi

if [ ! -x "$FULL_FFMPEG" ] || ! "$FULL_FFMPEG" -hide_banner -filters 2>&1 | grep -q ' ass '; then
  echo "正在安装带中文字幕支持的 ffmpeg-full..."
  brew install ffmpeg-full
fi
if [ ! -x "$FULL_FFMPEG" ] || [ ! -x "$FULL_FFPROBE" ]; then
  echo "ffmpeg-full 安装失败。"
  read "?按回车键关闭窗口..."
  exit 1
fi

if [ ! -x "$APP_VENV/bin/python" ]; then
  "$PYTHON" -m venv "$APP_VENV"
fi
"$APP_VENV/bin/python" -m pip install -q -r requirements.txt

if [ ! -x "$TTS_VENV/bin/python" ]; then
  echo "首次启动：正在创建 Qwen3-TTS 运行环境..."
  "$PYTHON" -m venv "$TTS_VENV"
fi

if ! "$TTS_VENV/bin/python" -c "import mlx_audio, scipy, huggingface_hub" >/dev/null 2>&1; then
  echo "正在安装 Qwen3-TTS 依赖..."
  "$TTS_VENV/bin/python" -m pip install -U pip setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple
  "$TTS_VENV/bin/python" -m pip install -U mlx-audio scipy huggingface_hub -i https://pypi.tuna.tsinghua.edu.cn/simple
fi

mkdir -p "$PWD/models"
WHISPER_SHA1="55356645c2b361a969dfd0ef2c5a50d530afd8d5"
CURRENT_WHISPER_SHA1=""
if [ -s "$WHISPER_MODEL" ]; then
  CURRENT_WHISPER_SHA1="$(shasum "$WHISPER_MODEL" | cut -d' ' -f1)"
fi
if [ "$CURRENT_WHISPER_SHA1" != "$WHISPER_SHA1" ]; then
  echo "首次启动：正在下载 Whisper small 模型（约 466 MiB）..."
  rm -f "$WHISPER_MODEL.part"
  curl -L --fail --retry 5 --retry-delay 5 --retry-all-errors \
    --output "$WHISPER_MODEL.part" \
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin"
  if [ "$(shasum "$WHISPER_MODEL.part" | cut -d' ' -f1)" != "$WHISPER_SHA1" ]; then
    echo "Whisper 模型校验失败，请检查网络后重试。"
    rm -f "$WHISPER_MODEL.part"
    read "?按回车键关闭窗口..."
    exit 1
  fi
  mv "$WHISPER_MODEL.part" "$WHISPER_MODEL"
fi

OLLAMA_LOG="${TMPDIR:-/tmp}/qwen3-tts-ollama.log"
if ! ollama list >/dev/null 2>&1; then
  echo "正在启动 Ollama 服务..."
  ollama serve >"$OLLAMA_LOG" 2>&1 &
  for _ in {1..30}; do
    ollama list >/dev/null 2>&1 && break
    sleep 1
  done
fi
if ! ollama list >/dev/null 2>&1; then
  echo "Ollama 服务启动失败，日志：$OLLAMA_LOG"
  read "?按回车键关闭窗口..."
  exit 1
fi

if ! ollama show qwen3.5:9b >/dev/null 2>&1; then
  echo "未找到 Ollama 模型 qwen3.5:9b，正在下载..."
  ollama pull qwen3.5:9b
fi

if ! HF_HUB_OFFLINE=1 "$TTS_VENV/bin/python" -c "from mlx_audio.tts.utils import load_model; load_model('$VOICE_DESIGN_MODEL')" >/dev/null 2>&1; then
  echo "首次启动：正在下载 Qwen3-TTS 1.7B VoiceDesign BF16 场景造声模型（约 4.5 GB）..."
  "$TTS_VENV/bin/hf" download "$VOICE_DESIGN_MODEL" --max-workers 4
  HF_HUB_OFFLINE=1 "$TTS_VENV/bin/python" -c "from mlx_audio.tts.utils import load_model; load_model('$VOICE_DESIGN_MODEL')" >/dev/null
fi

if ! HF_HUB_OFFLINE=1 "$TTS_VENV/bin/python" -c "from mlx_audio.tts.utils import load_model; load_model('$VOICE_CLONE_MODEL')" >/dev/null 2>&1; then
  echo "首次启动：正在下载 Qwen3-TTS 1.7B Base BF16 声音克隆模型（约 4.6 GB）..."
  "$TTS_VENV/bin/hf" download "$VOICE_CLONE_MODEL" --max-workers 4
  HF_HUB_OFFLINE=1 "$TTS_VENV/bin/python" -c "from mlx_audio.tts.utils import load_model; load_model('$VOICE_CLONE_MODEL')" >/dev/null
fi

export WHISPER_MODEL
export QWEN3_TTS_PYTHON="$TTS_VENV/bin/python"
export QWEN3_TTS_MODEL="$VOICE_DESIGN_MODEL"
export QWEN3_TTS_VOICE_DESIGN_MODEL="$VOICE_DESIGN_MODEL"
export QWEN3_TTS_VOICE_CLONE_MODEL="$VOICE_CLONE_MODEL"
export OLLAMA_MODEL="qwen3.5:9b"
export FFMPEG="$FULL_FFMPEG"
export FFPROBE="$FULL_FFPROBE"
export HF_HUB_OFFLINE=1

exec "$APP_VENV/bin/python" dubstudio.py serve
