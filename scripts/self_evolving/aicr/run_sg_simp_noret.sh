#!/usr/bin/env bash
# Chain RUN_SCRIPT for the AICR SIMPLE-PROMPT fixed baseline (ARM=14): the
# untuned 1-2 sentence generator templates, no retrieval, pure self-judge.
# Replaces sg-fix as the fair baseline for sg-adv — see run_sg_adv_noret.sh
# for why the arm number is baked into the shim.
#
#   RUN_SCRIPT=scripts/self_evolving/aicr/run_sg_simp_noret.sh \
#     sbatch -J sg-simp scripts/self_evolving/aicr/train_chain.sbatch
set -euo pipefail
ARM=14 EXP_SUFFIX="${EXP_SUFFIX:-_aicr}" \
    exec bash "$(dirname "$0")/run_specgap_aicr.sh"
