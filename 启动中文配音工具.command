#!/bin/zsh
set -e
cd "$(dirname "$0")"
VENV="$HOME/.venvs/chinese-dub-studio"
if [ -x /opt/homebrew/bin/python3 ]; then
  PYTHON=/opt/homebrew/bin/python3
else
  PYTHON=/usr/bin/python3
fi
if [ ! -x "$VENV/bin/python" ]; then
  "$PYTHON" -m venv "$VENV"
fi
"$VENV/bin/pip" install -q -r requirements.txt
exec "$VENV/bin/python" dubstudio.py serve
