#!/usr/bin/env bash
# Run inside an activated Python 3.10 environment.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"
python -m pip install --upgrade pip 'setuptools>=68,<81' wheel
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
DS_BUILD_OPS=0 python -m pip install --no-build-isolation deepspeed==0.18.5
python -m pip install -e .
python -m pip check
