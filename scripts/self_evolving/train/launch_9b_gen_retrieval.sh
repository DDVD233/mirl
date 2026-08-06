#!/usr/bin/env bash
# Thin launcher for the RETRIEVAL arm of the 9B generation experiment.
#
# Exists so queue_when_gpus_free.sh has a single command to run: that helper execs
# its argument directly, with no shell to carry env assignments, so RETRIEVAL=1 and
# the wandb key have to live inside a script.
#
# GEN_PORT defaults to 8042, not the control's 8041: if this is queued behind the
# control on the same box, the control's gen server may still be shutting down when
# this starts, and run_9b_hb_gen.sh treats an occupied port as a FATAL orphan.
set -xeuo pipefail

S=/scratch/sheng/self_evolving
cd "$S/verl_healthbench"

export WANDB_API_KEY="$(cat $S/.wandb_key_dvd)"
export RETRIEVAL=1
export GEN_PORT="${GEN_PORT:-8042}"
export EXP="${EXP:-hb9b_gen_retrieval}"

exec bash scripts/self_evolving/train/run_9b_hb_gen.sh "$@"
