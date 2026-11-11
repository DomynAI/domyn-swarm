#!/bin/bash
# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0
#
# Build a Python virtual environment with NVIDIA Dynamo over a vLLM image.
# Point `backend.endpoint.router.venv` of a `type: dynamo` router at the result.
#
# Usage: install_dynamo_venv.sh VERSION IMAGE VENV
# Example:
#   install_dynamo_venv.sh 1.6.0 /shared/images/vllm_0.31.0-cu129.sif /shared/venvs/dynamo-1.6.0
#
# Run it on a login node: pip needs the internet, and the --help checks at the
# end run without GPUs. Put VENV on a shared filesystem that every compute node
# mounts at the same path, and use the same IMAGE as the swarm's `image`.
#
# The venv is created with the image's own python3 and --system-site-packages,
# so it reuses the image's vllm and torch. The image's package versions become
# pip constraints, and pip adds only what ai-dynamo needs beyond the image. The
# vllm extra of ai-dynamo is left out for the same reason. Pick the ai-dynamo
# VERSION whose vllm extra pins the image's vLLM version (1.6.0 pins 0.31.0).
#
# Environment:
#   UNPIN        packages dropped from the constraints because ai-dynamo needs
#                another version (default: pydantic pydantic_core
#                googleapis-common-protos, as ai-dynamo 1.6.0 requires).
#                Set UNPIN= to keep every constraint.
#   EXTRA_BINDS  comma-separated Singularity bind specifications added to the
#                parent directory of VENV, which is always bound
#
# Rerunning the script skips the venv and the install when they are in place.
set -euo pipefail

if [ $# -ne 3 ]; then
  echo "usage: $0 VERSION IMAGE VENV" >&2
  exit 2
fi

VERSION=$1
IMAGE=$2
VENV=$(realpath -m "$3")
UNPIN=${UNPIN-pydantic pydantic_core googleapis-common-protos}

[ -f "$IMAGE" ] || [ -d "$IMAGE" ] || { echo "[install] ERROR: image $IMAGE not found" >&2; exit 1; }
mkdir -p "$(dirname "$VENV")"
BINDS=$(dirname "$VENV")${EXTRA_BINDS:+,$EXTRA_BINDS}
run() { singularity exec --bind "$BINDS" "$IMAGE" "$@"; }

if [ ! -x "$VENV/bin/python" ]; then
  echo "[install] creating $VENV over the image's python3"
  run python3 -m venv --system-site-packages "$VENV"
fi

run python3 -m pip list --format=freeze --disable-pip-version-check > "$VENV/container-constraints.txt"
unpin_pattern="^($(tr ' ' '\n' <<< "$UNPIN" | sed '/^$/d' | tr 'A-Z_.' 'a-z--' | paste -sd'|'))=="
tr 'A-Z_.' 'a-z--' < "$VENV/container-constraints.txt" | paste -d' ' - "$VENV/container-constraints.txt" |
  grep -v -E "$unpin_pattern" | cut -d' ' -f2 > "$VENV/constraints.txt" || true
echo "[install] $(wc -l < "$VENV/constraints.txt") constraints from the image, unpinned: ${UNPIN:-none}"

installed=$(run "$VENV/bin/python" -m pip show ai-dynamo 2>/dev/null | sed -n 's/^Version: //p' || true)
if [ "$installed" = "$VERSION" ]; then
  echo "[install] ai-dynamo $VERSION already installed"
else
  run "$VENV/bin/python" -m pip install --disable-pip-version-check \
    --constraint "$VENV/constraints.txt" "ai-dynamo==$VERSION"
fi

run "$VENV/bin/python" - <<'EOF'
import importlib.metadata as metadata
import re

pins = [r for r in metadata.requires("ai-dynamo") or [] if re.match(r"vllm\b", r) and "extra" in r]
wanted = re.search(r"==\s*([\w.]+)", pins[0]).group(1) if pins else None
have = metadata.version("vllm")
print(f"[install] ai-dynamo {metadata.version('ai-dynamo')}, "
      f"ai-dynamo-runtime {metadata.version('ai-dynamo-runtime')}, vllm {have}")
if wanted and not have.startswith(wanted):
    print(f"[install] WARNING: ai-dynamo[vllm] pins vllm {wanted}, the image has {have}")
EOF

run "$VENV/bin/python" -m dynamo.frontend --help > "$VENV/frontend-help.txt"
echo "[install] dynamo.frontend --help: $(wc -l < "$VENV/frontend-help.txt") lines in $VENV/frontend-help.txt"

# Without a GPU, vLLM cannot pick a platform or read the device capability while
# it builds its argument parser. Pin the CUDA platform, report an A100, and load
# the image's CUDA compat libcuda in place of the missing driver.
singularity exec --bind "$BINDS" \
  --env LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/nvidia/lib64:/usr/local/cuda/lib64 \
  "$IMAGE" "$VENV/bin/python" - dynamo.vllm > "$VENV/vllm-help.txt" <<'EOF'
import runpy
import sys

import vllm.platforms
from vllm.platforms.cuda import CudaPlatform
from vllm.platforms.interface import DeviceCapability

CudaPlatform.get_device_capability = classmethod(lambda cls, device_id=0: DeviceCapability(8, 0))
CudaPlatform.get_device_name = classmethod(lambda cls, device_id=0: "NVIDIA A100-SXM-64GB")
vllm.platforms._current_platform = CudaPlatform()
module = sys.argv[1]
sys.argv = [module, "--help"]
runpy.run_module(module, run_name="__main__", alter_sys=True)
EOF
echo "[install] dynamo.vllm --help: $(wc -l < "$VENV/vllm-help.txt") lines in $VENV/vllm-help.txt"
for flag in --kv-events-config --discovery-backend --event-plane --request-plane; do
  grep -q -- "$flag" "$VENV/vllm-help.txt" || echo "[install] WARNING: dynamo.vllm has no $flag"
done
