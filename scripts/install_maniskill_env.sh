#!/bin/bash
# One-click idempotent installer for the `maniskill` conda env.
# ManiSkill 3 = GPU-parallelized robotics sim built on SAPIEN.
#
# What it does:
#   1. conda create -n maniskill python=3.11   (skip if exists)
#   2. pip install torch                        (CUDA wheel)
#   3. pip install -e dependencies/ManiSkill    (the pinned submodule, editable)
#   4. sanity check: import mani_skill + sapien, vulkan ICD presence
#
# Re-running is safe: every step is guarded.
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV=maniskill
PY=3.11

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV"; then
    echo "✅ conda env '$ENV' already exists, skipping create."
else
    echo "📦 creating conda env '$ENV' (python=$PY) ..."
    conda create -n "$ENV" python=$PY -y
fi

conda activate "$ENV"

# torch — install once. ManiSkill leaves torch to the user so CUDA matches the box.
if python -c "import torch" 2>/dev/null; then
    echo "✅ torch already installed ($(python -c 'import torch;print(torch.__version__)'))."
else
    echo "📦 installing torch ..."
    pip install -q torch
fi

# mani_skill — editable from the submodule so the notebook tracks the pinned commit.
if python -c "import mani_skill" 2>/dev/null; then
    echo "✅ mani_skill already importable ($(python -c 'import mani_skill;print(mani_skill.__version__)' 2>/dev/null))."
else
    echo "📦 installing mani_skill (editable, from dependencies/ManiSkill) ..."
    pip install -q -e "$REPO_ROOT/dependencies/ManiSkill"
fi

# numpy<2 pin — REQUIRED. mplib==0.1.1 (the motion-planning backend, §3) is a native
# extension built against numpy 1.x; numpy 2.x segfaults inside ArticulatedModel().
# `pip install torch` pulls numpy 2.x by default, so we force it back down and pin a
# numpy-1-compatible opencv. torch/sapien/mani_skill all work fine on numpy 1.26.
if python -c "import numpy,sys; sys.exit(0 if numpy.__version__.startswith('1.') else 1)" 2>/dev/null; then
    echo "✅ numpy already <2 ($(python -c 'import numpy;print(numpy.__version__)'))."
else
    echo "📦 pinning numpy<2 + opencv<4.12 (mplib 0.1.1 needs numpy 1.x — else §3 segfaults) ..."
    pip install -q "numpy<2" "opencv-python<4.12"
fi

echo
echo "🔎 sanity check ..."
python - <<'PY'
import mani_skill, sapien, torch
print(f"  mani_skill : {mani_skill.__version__}")
print(f"  sapien     : {sapien.__version__}")
print(f"  torch      : {torch.__version__}  (cuda={torch.cuda.is_available()})")
PY

# Vulkan ICD — SAPIEN renders through Vulkan. Missing ICD = black screen / crash.
if ls /usr/share/vulkan/icd.d/*.json >/dev/null 2>&1 || [ -n "$VK_ICD_FILENAMES" ]; then
    echo "  vulkan ICD : found ✅"
else
    echo "  vulkan ICD : ⚠️  none found under /usr/share/vulkan/icd.d/"
    echo "             GUI/offscreen rendering may fail. See:"
    echo "             https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#vulkan"
fi

echo
echo "🎉 done. Notebook cells run via:  conda run -n $ENV --no-capture-output python -m mani_skill.examples.<demo> ..."
