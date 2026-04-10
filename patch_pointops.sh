#!/bin/bash
# Patch pointops for PyTorch 2.x compatibility.
#
# THC/THC.h was removed in PyTorch 2.0. This script comments out the
# include directives so pointops compiles with modern PyTorch.
#
# Additionally, the pointops submodule (YodaEmbedding/pointops) is missing
# functions required by this codebase: queryandgroup and sectorized_fps.
# These exist in SymPointV2's pointops (https://github.com/nicehuster/SymPointV2).
# See the note in the README for instructions.
#
# Usage:
#   git submodule update --init --recursive
#   bash patch_pointops.sh
#   cd modules/pointops && python setup.py install

set -e

POINTOPS_SRC="modules/pointops/src"

if [ ! -d "$POINTOPS_SRC" ]; then
    echo "Error: $POINTOPS_SRC not found."
    echo "Run 'git submodule update --init --recursive' first."
    exit 1
fi

echo "Patching THC includes for PyTorch 2.x compatibility..."

find "$POINTOPS_SRC" -name '*.cpp' -exec \
    sed -i 's|^#include <THC/THC.h>|// #include <THC/THC.h>  // removed for PyTorch 2.x|' {} +

find "$POINTOPS_SRC" -name '*.cpp' -exec \
    sed -i 's|^extern THCState|// extern THCState|' {} +

echo "Done. Now run:"
echo "  cd modules/pointops && python setup.py install"
