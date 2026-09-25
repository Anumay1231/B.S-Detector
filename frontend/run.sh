#!/usr/bin/env bash
# ============================================================
# B.S. Detector — Localhost Launcher
# Runs the Streamlit cyber-forensic frontend on http://localhost:8501
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "============================================================"
echo "🛡️  B.S. Detector — Localhost Frontend Launcher"
echo "============================================================"

# Detect Python Environment
PYTHON_BIN=""
if [ -f "${REPO_DIR}/../.venv/bin/streamlit" ]; then
    PYTHON_BIN="${REPO_DIR}/../.venv/bin/python"
    STREAMLIT_BIN="${REPO_DIR}/../.venv/bin/streamlit"
elif [ -f "${REPO_DIR}/.venv/bin/streamlit" ]; then
    PYTHON_BIN="${REPO_DIR}/.venv/bin/python"
    STREAMLIT_BIN="${REPO_DIR}/.venv/bin/streamlit"
elif command -v streamlit &>/dev/null; then
    PYTHON_BIN="$(which python3)"
    STREAMLIT_BIN="$(which streamlit)"
else
    echo "⚠️  Streamlit not found in current environment."
    echo "   Please run: pip install -r ${SCRIPT_DIR}/requirements.txt"
    exit 1
fi

echo "✓ Using Python: ${PYTHON_BIN}"
echo "✓ Using Streamlit: ${STREAMLIT_BIN}"
echo ""
echo "🚀 Launching Cyber-Forensic Studio on localhost..."
echo "   URL: http://localhost:8501"
echo "============================================================"

cd "${REPO_DIR}"
"${STREAMLIT_BIN}" run frontend/app.py \
    --server.address localhost \
    --server.port 8501 \
    --theme.base dark \
    --theme.primaryColor "#00f0ff" \
    --theme.backgroundColor "#070a12" \
    --theme.secondaryBackgroundColor "#101827" \
    --theme.textColor "#f1f5f9"
