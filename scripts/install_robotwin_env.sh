#!/bin/bash
# One-click idempotent installer for the `robotwin` conda env.
# RoboTwin 2.0 = SAPIEN-based bimanual data generator + benchmark (CVPR'25 Highlight).
#
# Mirrors dependencies/RoboTwin/script/_install.sh but: (a) into a dedicated conda env,
# (b) every heavy step guarded so re-running is cheap, (c) downloads the ~GB assets.
#
# Steps:
#   1. conda create -n robotwin python=3.10
#   2. pip install -r script/requirements.txt   (torch 2.4.1, sapien 3.0.0b1, mplib, ...)
#   3. pip install pytorch3d (from source, no-build-isolation)
#   4. patch sapien/urdf_loader.py + mplib/planner.py   (utf-8 + collision-check fix)
#   5. clone + pip install curobo v0.7.8                (GPU motion planning)
#   6. download + unzip assets (background_texture / embodiments / objects) from HF
#
# First run: 30-60 min (curobo build + ~GB asset download). Re-run: seconds.
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RT="$REPO_ROOT/dependencies/RoboTwin"
# Dedicated env name — do NOT reuse a pre-existing `robotwin` env that may belong to
# another project (e.g. lerobot-experience), whose activate hook / PYTHONPATH would
# shadow this submodule's envs/ and pull in a different RoboTwin checkout.
ENV="${ROBOTWIN_ENV:-robotwin_sx}"
PY=3.10

source "$(conda info --base)/etc/profile.d/conda.sh"

# ---- 1. env ----
if conda env list | awk '{print $1}' | grep -qx "$ENV"; then
    echo "✅ conda env '$ENV' already exists, skipping create."
else
    echo "📦 creating conda env '$ENV' (python=$PY) ..."
    conda create -n "$ENV" python=$PY -y
fi
conda activate "$ENV"

# ---- 2. base requirements ----
if python -c "import sapien, mplib, torch" 2>/dev/null; then
    echo "✅ base deps (sapien/mplib/torch) already installed."
else
    echo "📦 pip install -r script/requirements.txt (this pulls torch 2.4.1, sapien 3.0.0b1, ...) ..."
    pip install -r "$RT/script/requirements.txt"
fi

# ---- 3. pytorch3d ----
if python -c "import pytorch3d" 2>/dev/null; then
    echo "✅ pytorch3d already installed."
else
    echo "📦 installing pytorch3d (from source, no-build-isolation) ..."
    pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation
fi

# ---- 4. patches (idempotent: only patch if the old pattern still present) ----
SAPIEN_LOC="$(pip show sapien 2>/dev/null | awk '/Location/{print $2}')/sapien"
URDF_LOADER="$SAPIEN_LOC/wrapper/urdf_loader.py"
if [ -f "$URDF_LOADER" ] && grep -qE '"r"\) as' "$URDF_LOADER"; then
    echo "🔧 patching sapien urdf_loader.py (utf-8 encoding) ..."
    sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8")\3/g' "$URDF_LOADER"
else
    echo "✅ sapien urdf_loader.py already patched (or pattern absent)."
fi

MPLIB_LOC="$(pip show mplib 2>/dev/null | awk '/Location/{print $2}')/mplib"
PLANNER="$MPLIB_LOC/planner.py"
if [ -f "$PLANNER" ] && grep -qE 'or collide or not within_joint_limit:' "$PLANNER"; then
    echo "🔧 patching mplib planner.py (drop collide short-circuit) ..."
    sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' "$PLANNER"
else
    echo "✅ mplib planner.py already patched (or pattern absent)."
fi

# ---- 5. curobo ----
if python -c "import curobo" 2>/dev/null; then
    echo "✅ curobo already installed."
else
    echo "📦 installing curobo v0.7.8 (GPU motion planning, builds CUDA kernels — slow) ..."
    if [ ! -d "$RT/envs/curobo" ]; then
        git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git "$RT/envs/curobo"
    fi
    # _FORTIFY_SOURCE=0 — REQUIRED on this box. System nvcc is CUDA 12.0, too old for
    # Ubuntu 24's glibc 2.39 fortify headers (uses __builtin_dynamic_object_size, which
    # CUDA <12.4 nvcc can't parse). conda activation defaults _FORTIFY_SOURCE=2, so the
    # curobo CUDA kernel build dies with "identifier __builtin_dynamic_object_size is
    # undefined". Forcing it off makes the .cu files compile. (Alt fix: install cuda-nvcc>=12.4.)
    NVCC_APPEND_FLAGS="-U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0" \
    CFLAGS="-U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0" \
    CXXFLAGS="-U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0" \
    CPPFLAGS="-U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0" \
        pip install -e "$RT/envs/curobo" --no-build-isolation
    pip install warp-lang==1.12.0 setuptools==69.5.1
fi

# ---- 6. assets ----
if [ -d "$RT/assets/embodiments" ] && [ -d "$RT/assets/objects" ]; then
    echo "✅ assets already downloaded (assets/embodiments + assets/objects present)."
else
    echo "📦 downloading + unzipping assets (~GB, from HF TianxingChen/RoboTwin2.0) ..."
    ( cd "$RT/assets" && python _download.py \
        && for z in background_texture embodiments objects; do
               [ -f "$z.zip" ] && { unzip -q -o "$z.zip" && rm -f "$z.zip"; }
           done )
    echo "🔧 configuring embodiment asset paths ..."
    ( cd "$RT" && python ./script/update_embodiment_config_path.py )
fi

echo
echo "🎉 done. Run a task preview with:"
echo "    python scripts/robotwin_demo.py preview <task_name> --gpu 0"
