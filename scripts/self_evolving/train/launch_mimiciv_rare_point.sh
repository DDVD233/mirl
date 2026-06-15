#!/bin/bash
# Launch wrapper for mimiciv_rare training on point.dd.works.
#
# Bypasses the conda activate / PATH propagation problem we hit when
# launching via ssh + nohup: a disowned child bash inherits the
# default PATH (base env first), so `python3` resolves to the wrong
# interpreter (no tensordict). This wrapper prepends the verl env's
# bin to PATH explicitly before exec-ing the underlying script.
#
# vLLM servers run separately in the cu128 env (vllm 0.14.1).
# Training runs in the verl env (vllm 0.11.0 + tensordict).

set -e
export PATH="/home/dvdai/miniconda3/envs/verl/bin:$PATH"
export PYTHON_BIN="/home/dvdai/miniconda3/envs/verl/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export RAY_ADDRESS=local
# Surface real tracebacks if Hydra/Ray swallows them.
export HYDRA_FULL_ERROR=1
export API_BASE="${API_BASE:-http://localhost:8002/v1}"
export EMBED_API_BASE="${EMBED_API_BASE:-http://localhost:8001/v1}"
export MILVUS_URI="${MILVUS_URI:-http://mib.media.mit.edu:19531}"
export DATA_DIR="${DATA_DIR:-/home/dvdai/scratch/dvdai/self_evolving_datasets/mimiciv_rare}"
LOG_DIR="${LOG_DIR:-/home/dvdai/scratch/dvdai/self_evolving_datasets/logs}"

cd /home/dvdai/verl
echo "python3 -> $(which python3)" >&2
python3 -c "import tensordict, sys; sys.stderr.write(f'tensordict OK {tensordict.__version__}\n')"

bash scripts/self_evolving/train/run_multi_agent_mimiciv_rare.sh \
    ++data.self_evolving.log_dir="$LOG_DIR" \
    "$@"
