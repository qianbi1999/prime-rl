#!/bin/bash
# One-click apply all NPU migration patches
# Usage: bash patches/apply.sh
set -e

echo "=== Layer 1/3: Device abstraction layer ==="
git apply patches/0001-device-abstraction.patch
echo "  + src/prime_rl/_device.py"

echo "=== Layer 2/3: Source code adaptation ==="
git apply patches/0002-source-adaptation.patch
echo "  ~ 12 source files + pyproject.toml (cuda -> npu)"

echo "=== Layer 3/3: NPU training config ==="
git apply patches/0003-npu-config.patch
echo "  + examples/wordle/rl_npu.toml"

echo ""
echo "NPU migration patches applied successfully."
echo "Next steps: pip install -r requirements-npu.txt"
echo "See patches/README.md for full training guide."
