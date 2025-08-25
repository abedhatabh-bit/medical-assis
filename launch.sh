#!/bin/bash
# Medical Assistant CLI Launch Script

set -euo pipefail

echo "🏥 Medical Assistant CLI Launcher"
echo "================================"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Running setup..."
    ./setup.sh
else
    echo "✅ Virtual environment found"
fi

# Activate virtual environment
source .venv/bin/activate

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found. Please copy .env.example to .env and add your OpenAI API key."
    exit 1
fi

# Check if OpenAI API key is set
if ! grep -q "OPENAI_API_KEY=your_openai_api_key_here" .env; then
    echo "✅ OpenAI API key appears to be configured"
else
    echo "⚠️  Please edit .env and set your OpenAI API key"
    echo "   Current setting: OPENAI_API_KEY=your_openai_api_key_here"
    exit 1
fi

echo ""
echo "🚀 Launching Medical Assistant CLI..."
echo "   Run with: python -m app --help"
echo ""

# If arguments provided, run the command
if [ $# -gt 0 ]; then
    python -m app "$@"
else
    python -m app --help
fi