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

echo "Starting Analytics Assistant..."
streamlit run app/streamlit_app.py \
  --server.headless false \
  --server.port 8501 \
  --browser.gatherUsageStats false
