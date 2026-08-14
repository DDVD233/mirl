#!/usr/bin/env bash
# Chain RUN_SCRIPT for the AICR adversary-v2 no-retrieval arm (ARM=11).
#
# WHY A SHIM. train_chain.sbatch runs "$RUN_SCRIPT" in whatever environment the
# resubmit machinery reconstructed; baking ARM here (instead of exporting it at
# sbatch time) means a chain successor can never come up as the wrong arm.
#
#   RUN_SCRIPT=scripts/self_evolving/aicr/run_sg_adv_noret.sh \
#     sbatch -J sg-adv scripts/self_evolving/aicr/train_chain.sbatch
set -euo pipefail
ARM=11 EXP_SUFFIX="${EXP_SUFFIX:-_aicr}" \
    exec bash "$(dirname "$0")/run_specgap_aicr.sh"
