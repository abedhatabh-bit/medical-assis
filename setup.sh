#!/usr/bin/env bash
set -euo pipefail

echo "🏥 Medical Assistant CLI Setup"
echo "=============================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "Python version: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Setup environment file
if [ ! -f .env ]; then
  cp .env.example .env
  echo "📝 Created .env file (please edit and add your OPENAI_API_KEY)"
else
  echo "✅ .env file already exists"
fi

# Create store directory
mkdir -p store/outputs

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your OpenAI API key"
echo "2. Run: ./launch.sh --help"
echo "3. Or activate with: source .venv/bin/activate"