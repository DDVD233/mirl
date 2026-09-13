#!/usr/bin/env bash
# Chain RUN_SCRIPT for the general-description adversary arm (ARM=23) on AICR.
#
# Same shim pattern as run_sg_adv_noret.sh: ARM is baked here so a chain successor
# can never come up as the wrong arm. The arm serves its own summarizer on GPU 3 of
# the allocation (SUMM_BASE local), shared with the policy, which trains on all four.
#
#   RUN_SCRIPT=/scratch/dvdai_mit/self_evolving/verl_specgap/scripts/self_evolving/aicr/run_sg_general.sh \
#     FRP_PORT=2339 sbatch -J sg-gen -p rtx-batch --gres=gpu:rtx_pro_6000:4 -c 48 --mem=600G \
#     /scratch/dvdai_mit/self_evolving/verl_specgap/scripts/self_evolving/aicr/train_chain.sbatch
set -euo pipefail
ARM=23 EXP_SUFFIX="${EXP_SUFFIX:-_aicr}" N_GPUS="${N_GPUS:-4}" FROZEN_GPU="${FROZEN_GPU:-3}" \
    exec bash "$(dirname "$0")/run_specgap_aicr.sh"
