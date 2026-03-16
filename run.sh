#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Launch the Analytics Assistant Streamlit app.
# Usage:  ./run.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure .env exists
if [[ ! -f ".env" ]]; then
  echo "ERROR: .env not found. Copy .env.example to .env and fill in your credentials."
  exit 1
fi

# Create log directory
mkdir -p logs

# Ensure dependencies (including plotly for charts)
pip install -q -r requirements.txt 2>/dev/null || true

echo "Starting Analytics Assistant..."
# Run from app/ so Streamlit finds app/pages/ (Admin Context Manager, etc.)
cd app && streamlit run streamlit_app.py \
  --server.headless false \
  --server.port 8501 \
  --browser.gatherUsageStats false
