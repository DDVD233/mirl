#!/usr/bin/env bash
# Run the gemma-3 vllm smoke test inside the apptainer container.
#
# Usage (on the cluster):
#   bash examples/tactile/run_apptainer.sh examples/tactile/smoke_vllm_gemma3.sh
set -x
python3 examples/tactile/smoke_vllm_gemma3.py "$@"
