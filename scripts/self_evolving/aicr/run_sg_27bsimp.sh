#!/usr/bin/env bash
# Chain RUN_SCRIPT for the AICR 27B SIMPLE baseline (ARM=18): the untuned floor
# at 27B scale, the control for ARM=16's 27B adversary. Submit with the larger
# memory the 27B's FSDP offload needs:
#   RUN_SCRIPT=scripts/self_evolving/aicr/run_sg_27bsimp.sh \
#     sbatch -J sg-27bsimp -c 48 --mem=800G scripts/self_evolving/aicr/train_chain.sbatch
set -euo pipefail
ARM=18 EXP_SUFFIX="${EXP_SUFFIX:-_aicr}" \
    exec bash "$(dirname "$0")/run_specgap_aicr.sh"
