#!/usr/bin/env bash
# Chain RUN_SCRIPT for the AICR ALL-9B SIMPLE baseline (ARM=17): the control for
# ARM=15's all-self adversary — untuned templates, same retrieval+websearch
# stack, every model role on the frozen 9B. See run_sg_adv_noret.sh for why the
# arm number is baked into the shim.
#
#   RUN_SCRIPT=scripts/self_evolving/aicr/run_sg_selfsimp.sh \
#     sbatch -J sg-selfsimp scripts/self_evolving/aicr/train_chain.sbatch
set -euo pipefail
ARM=17 EXP_SUFFIX="${EXP_SUFFIX:-_aicr}" \
    exec bash "$(dirname "$0")/run_specgap_aicr.sh"
