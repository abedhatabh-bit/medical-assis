#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env (fill in OPENAI_API_KEY)"
fi

echo "Setup complete. Activate with: source .venv/bin/activate"