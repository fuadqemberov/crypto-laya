#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ -x .native/freqtrade/bin/python ]; then export FREQTRADE_PYTHON="$PWD/.native/freqtrade/bin/python"; fi
if [ -x .native/jesse/bin/python ]; then export JESSE_PYTHON="$PWD/.native/jesse/bin/python"; fi
exec .venv/bin/python app.py "$@"
