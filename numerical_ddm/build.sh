#!/bin/bash
set -eo pipefail
# Execute from workspace root. Uses the same conda environment as the original build.
source "$HOME/miniforge/etc/profile.d/conda.sh"
conda activate moose
set -u
make -C moose_env/moose/modules/combined -j"${MOOSE_BUILD_JOBS:-8}"
