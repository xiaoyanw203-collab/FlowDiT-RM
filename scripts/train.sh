#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

python -m flowdit_rm.training.train "$@"
