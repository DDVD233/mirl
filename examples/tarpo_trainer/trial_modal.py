# .
# ├── modal_verl_rl.py
# └── train.sh

# TERMINAL

# pip install modal
# modal setup

# # Volumes
# modal volume create verl-checkpoints
# modal volume create hf-cache
# modal volume create hb-data

# # Secrets (create hf-secret only if dataset/model is gated/private)
# modal secret create hf-secret HUGGINGFACE_HUB_TOKEN="hf_..."
# # optional
# modal secret create wandb-secret WANDB_API_KEY="..."


### In the train.sh
# #!/usr/bin/env bash
# set -euo pipefail

# # Required env vars set by Modal function:
# #   DATA_ROOT   (e.g. /data/hb)
# #   OUT_ROOT    (e.g. /checkpoints/naive_density)
# # Optional:
# #   WANDB_PROJECT, WANDB_RUN_NAME, HF_HOME

# export PYTHONPATH="/opt/verl:${PYTHONPATH:-}"
# export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"

# echo "DATA_ROOT=${DATA_ROOT}"
# echo "OUT_ROOT=${OUT_ROOT}"

# echo "Sanity checks:"
# which ffmpeg || true
# ffmpeg -version || true
# python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
# python -c "import sys; sys.path.insert(0,'/opt/verl'); import verl.trainer.main_ppo; print('OK: verl.trainer.main_ppo import')"

# TRAIN_JSONL="${DATA_ROOT}/final_v8_train_cleaned_2.jsonl"
# VAL_JSONL="${DATA_ROOT}/final_v8_val_cleaned.jsonl"

# python3 -m verl.trainer.main_ppo \
#   algorithm.adv_estimator=tarpo \
#   data.train_files="${TRAIN_JSONL}" \
#   data.val_files="${VAL_JSONL}" \
#   data.train_batch_size=256 \
#   data.val_batch_size=64 \
#   data.max_prompt_length=4096 \
#   data.max_response_length=2048 \
#   data.filter_overlong_prompts=False \
#   data.truncation='right' \
#   data.image_key=images \
#   data.video_key=videos \
#   data.prompt_key=problem \
#   data.dataloader_num_workers=8 \
#   data.modalities='audio,videos' \
#   data.train_modality_batching.enabled=True \
#   data.train_modality_batching.drop_last=True \
#   data.val_modality_batching.enabled=True \
#   data.val_modality_batching.drop_last=False \
#   data.format_prompt=/opt/verl/examples/format_prompt/default.jinja \
#   actor_rollout_ref.model.path=Qwen/Qwen2.5-Omni-7B \
#   actor_rollout_ref.actor.optim.lr=5e-7 \
#   actor_rollout_ref.model.use_remove_padding=True \
#   actor_rollout_ref.actor.ppo_mini_batch_size=128 \
#   actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
#   actor_rollout_ref.actor.use_kl_loss=False \
#   actor_rollout_ref.actor.kl_loss_coef=0 \
#   actor_rollout_ref.actor.kl_loss_type=low_var_kl \
#   actor_rollout_ref.actor.entropy_coeff=0 \
#   actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
#   actor_rollout_ref.model.enable_gradient_checkpointing=True \
#   actor_rollout_ref.actor.fsdp_config.param_offload=False \
#   actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
#   actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
#   actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
#   actor_rollout_ref.rollout.name=vllm \
#   actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=False \
#   actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
#   actor_rollout_ref.rollout.enable_chunked_prefill=False \
#   actor_rollout_ref.rollout.enforce_eager=False \
#   actor_rollout_ref.rollout.free_cache_engine=True \
#   actor_rollout_ref.rollout.n=5 \
#   actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
#   actor_rollout_ref.ref.fsdp_config.param_offload=True \
#   actor_rollout_ref.rollout.max_model_len=6192 \
#   actor_rollout_ref.rollout.max_num_batched_tokens=6192 \
#   algorithm.use_kl_in_reward=False \
#   custom_reward_function.path=/opt/verl/examples/reward_function/human_behaviour_tarpo.py \
#   custom_reward_function.name=human_behaviour_compute_score_batch \
#   reward_model.reward_manager=batch \
#   trainer.critic_warmup=0 \
#   trainer.logger='["console","wandb"]' \
#   trainer.project_name="${WANDB_PROJECT:-rl_omni_heldout}" \
#   trainer.experiment_name="${WANDB_RUN_NAME:-naive_density}" \
#   trainer.n_gpus_per_node=4 \
#   trainer.nnodes=1 \
#   trainer.save_freq=25 \
#   trainer.val_before_train=False \
#   trainer.val_only=False \
#   trainer.validation_data_dir="${OUT_ROOT}" \
#   trainer.test_freq=50 \
#   trainer.total_epochs=5 \
#   trainer.advantage_save_dir="${OUT_ROOT}/advantages" \
#   trainer.advantage_plot_freq=15 \
#   trainer.default_local_dir="${OUT_ROOT}"



## And then we run the following :

import os
import subprocess
import modal

app = modal.App("keane-verl-rl")

# Persistent volumes
ckpt_vol = modal.Volume.from_name("verl-checkpoints", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("hb-data", create_if_missing=True)

# -----------------------------
# Config: edit these
# -----------------------------
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "YOUR_ORG/YOUR_HF_DATASET_REPO")
HF_DATASET_SUBDIR = os.environ.get("HF_DATASET_SUBDIR", "")  # if your jsonl is in a subfolder inside the HF dataset repo
DATA_ROOT = "/data/hb"  # where we store the dataset snapshot in the mounted data_vol

VERL_REPO = "https://github.com/DDVD233/mirl.git"
VERL_BRANCH = "keane_tarpo"

TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
TORCH_PKGS = [
    "torch==2.8.0",
    "torchvision==0.23.0",
    "torchaudio==2.8.0",
]

VLLM_VERSION = "0.10.2"
FLASH_ATTN_REF = "main"

# GPU config: use the recommended string form (supports "H100:4" etc.).  [oai_citation:2‡Modal](https://modal.com/docs/reference/modal.gpu)
GPU = "H100:4"

# Note: ephemeral_disk is in MiB; 600*1024 ≈ 600 GiB.  [oai_citation:3‡Modal](https://modal.com/docs/guide/resources?utm_source=chatgpt.com)
PREP_EPHEMERAL_DISK_MIB = 600 * 1024

# -----------------------------
def run(cmd: str):
    print(f"\n>>> {cmd}\n", flush=True)
    subprocess.run(cmd, shell=True, check=True)


# -----------------------------
# Image build
# -----------------------------
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install(
        "git",
        "git-lfs",
        "build-essential",
        "cmake",
        "ninja-build",
        "pkg-config",
        "curl",
        "ca-certificates",
        "bzip2",
    )
    .run_commands(
        # ---------- ffmpeg 6.* via micromamba ----------
        "mkdir -p /opt/micromamba",
        "curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /opt/micromamba --strip-components=1 bin/micromamba",
        "ln -sf /opt/micromamba/bin/micromamba /usr/local/bin/micromamba",
        "micromamba create -y -p /opt/conda -c conda-forge 'ffmpeg=6.*' 'libgcc-ng>=13' 'libstdcxx-ng>=13'",
        "ln -sf /opt/conda/bin/ffmpeg /usr/local/bin/ffmpeg",
        "ffmpeg -version",

        # ---------- Python deps ----------
        "python -m pip install -U pip",
        f"pip install --index-url {TORCH_INDEX} " + " ".join(TORCH_PKGS),
        "pip install -U 'setuptools>=78,<80'",
        "pip install huggingface_hub hf_transfer",
        "pip install ujson scikit-learn qwen-vl-utils",

        # ---------- VERL ----------
        f"rm -rf /opt/verl && git clone -b {VERL_BRANCH} {VERL_REPO} /opt/verl",
        "pip install -r /opt/verl/requirements.txt",

        # ---------- vLLM pinned ----------
        f"pip install 'vllm[audio]=={VLLM_VERSION}'",

        # ---------- MathRuler ----------
        "rm -rf /opt/MathRuler && git clone https://github.com/hiyouga/MathRuler.git /opt/MathRuler",
        "pip install /opt/MathRuler",

        # ---------- flash-attn from source ----------
        f"rm -rf /opt/flash-attention && git clone --branch {FLASH_ATTN_REF} https://github.com/Dao-AILab/flash-attention /opt/flash-attention",
        "rm -rf /root/.cache/torch_extensions/* /opt/flash-attention/build /opt/flash-attention/dist",
        "export CUDA_HOME=/usr/local/cuda && export PATH=$CUDA_HOME/bin:$PATH && export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH && "
        "cd /opt/flash-attention && pip install -v --no-build-isolation .",

        # ---------- torchcodec last (no-deps to avoid torch drift) ----------
        "pip install --no-deps --index-url https://download.pytorch.org/whl/cu128 torchcodec==0.6.0",

        # ---------- sanity ----------
        "python -c 'import torch; print(torch.__version__, torch.version.cuda)'",
        "python -c 'import vllm; print(vllm.__version__)'",
        "python -c 'import sys; sys.path.insert(0, \"/opt/verl\"); import verl.trainer.main_ppo; print(\"OK: verl.trainer.main_ppo\")'",
        "pip check || true",
    )
    .add_local_file("./train.sh", remote_path="/opt/train.sh")
    .run_commands("chmod +x /opt/train.sh")
    .env(
        {
            "PYTHONPATH": "/opt/verl",
            "HF_HOME": "/root/.cache/huggingface",
            # Hugging Face downloader can use faster transfers when available
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
)


# -----------------------------
# Step 1: Download HF dataset into persistent /data volume
# -----------------------------
@app.function(
    image=image,
    timeout=24 * 60 * 60,
    ephemeral_disk=PREP_EPHEMERAL_DISK_MIB,
    volumes={
        "/data": data_vol,
        "/root/.cache/huggingface": hf_cache_vol,
    },
    secrets=[
        # include hf-secret if dataset is gated/private
        modal.Secret.from_name("hf-secret"),
    ],
)
def prepare_dataset():
    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")

    target_dir = DATA_ROOT
    marker = os.path.join(target_dir, ".READY")

    if os.path.exists(marker):
        print(f"Dataset already present at {target_dir}")
        return

    # Download entire dataset repo snapshot into /data/hb
    # local_dir_use_symlinks=False ensures files are physically present in the Volume
    # (no fragile symlink dependency on cache-only paths).
    py = f"""
from huggingface_hub import snapshot_download
import os

repo_id = {HF_DATASET_REPO!r}
local_dir = {target_dir!r}

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    local_dir_use_symlinks=False,
)
open(os.path.join(local_dir, ".READY"), "w").write("ok")
print("Downloaded dataset to", local_dir)
"""
    run("python - << 'PY'\n" + py + "\nPY")

    data_vol.commit()
    hf_cache_vol.commit()


# -----------------------------
# Step 2: Train (runs your shell script)
# -----------------------------
@app.function(
    image=image,
    gpu=GPU,
    timeout=24 * 60 * 60,
    volumes={
        "/checkpoints": ckpt_vol,
        "/root/.cache/huggingface": hf_cache_vol,
        "/data": data_vol,
    },
    secrets=[
        modal.Secret.from_name("hf-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
)
def train(run_name: str = "naive_density"):
    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
    os.environ["PYTHONPATH"] = "/opt/verl:" + os.environ.get("PYTHONPATH", "")

    dataset_dir = DATA_ROOT
    if HF_DATASET_SUBDIR:
        dataset_dir = os.path.join(dataset_dir, HF_DATASET_SUBDIR.strip("/"))

    out_root = f"/checkpoints/{run_name}"

    # Expose paths to the bash script
    env = os.environ.copy()
    env["DATA_ROOT"] = dataset_dir
    env["OUT_ROOT"] = out_root
    env["WANDB_RUN_NAME"] = run_name

    print(f"Running training with DATA_ROOT={dataset_dir}, OUT_ROOT={out_root}")
    subprocess.run("bash /opt/train.sh", shell=True, check=True, env=env)

    ckpt_vol.commit()
    hf_cache_vol.commit()


# -----------------------------
# Local entrypoint
# -----------------------------
@app.local_entrypoint()
def main():
    # 1) Download dataset to /data/hb (CPU)
    prepare_dataset.remote()

    # 2) Run training (GPU)
    train.remote("naive_density")


### running commands:

# export HF_DATASET_REPO="YOUR_ORG/YOUR_DATASET_REPO"
# # optionally if your jsonl lives under a subfolder inside the dataset repo:
# # export HF_DATASET_SUBDIR="human_behaviour_data"

# modal run modal_verl_rl.py