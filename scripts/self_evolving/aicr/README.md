# AICR cluster setup (self_evolving / healthbench)

Migration target now that MSR GPU allocation is shrinking. Login:
`ssh dvdai_mit@login.aicr.ai` — Slurm, account `ppliang_mit`. Docs: https://docs.aicr.ai/

## Partitions

| partition | GPUs/node | max time | notes |
|---|---|---|---|
| `b200-batch` | 8× B200 (183 GB) | 24 h | main training partition, ~28 nodes |
| `b200-devel` | 8× B200 | 4 h | interactive/debug, 3 nodes |
| `rtx-batch` | 8× RTX Pro 6000 | 24 h | 17 nodes |
| `rtx-devel` | 8× RTX Pro 6000 | 4 h | 2 nodes |
| `cpu` | — | 24 h | orchestration, transfers |

Nodes: 128 CPU, ~2.3 TB RAM. Request GPUs with `--gres=gpu:b200:4`.

## Storage

| path | quota | persistence | use for |
|---|---|---|---|
| `~` (home) | 100 GiB | 7-day snapshots | frp binary, dotfiles |
| `/work/mit/ppliang_mit/dvdai` (= `~/scratch`!) | 5 TB **group-shared**, watch usage | 7-day snapshots | code (`mirl/`), SIF (`sif/`), small data (`self_evolving/`) |
| `/scratch/dvdai_mit` | 10 TiB per user | **30-DAY PURGE**, no snapshots | HF cache, checkpoints, logs (`self_evolving/{hf_cache,checkpoints,logs}`) |

Note the confusing symlink: `~/scratch` points at the *work* dir, not `/scratch`.
The `/work` group quota is shared by the whole lab — keep only code + small
data there; everything big lives on `/scratch/dvdai_mit`.

**Anti-purge:** the 30-day `/scratch` purge is mtime-based. `touch_scratch.sh`
re-touches every path under `/scratch/dvdai_mit` whose mtime is >7 days old,
and runs as a Slurm-native **scrontab** job (biweekly: 1st + 15th, 04:00, cpu
partition — `scrontab -l` to inspect, `scrontab -e` to edit; lives in slurm so
no login-node process to babysit). Still back up milestone checkpoints to HF
hub (`backup_checkpoints_to_hf_v2.py`) — /scratch has no snapshots and the
toucher is a safety net, not a backup.

## Environment

No conda build needed — everything runs inside the project image via apptainer
(1.4.5 on all nodes):

```
SIF=/scratch/dvdai_mit/self_evolving/sif/verl-selfevolving-cu130-vllm0.22.1.sif
apptainer exec --nv --writable-tmpfs --bind /work:/work --bind /scratch:/scratch $SIF ...
```

(weianxie runs the same image for the pubmedqa self-evolving job — see
`/work/mit/ppliang_mit/weianxie/mirl/scripts/self_evolving/train/slurm_b200_selfimprove.sbatch`
for a working reference, including the B200 flash-attn v2 pin.)

## Connectivity (verified from compute nodes)

- Outbound is open: HF, `point.dd.works` (frps :7000, TRAPI proxy :18890), `mib.media.mit.edu:22`.
- Milvus (medvecdb on mib) is **not** directly reachable — tunnel it:
  `ssh -N -L 19531:localhost:19531 mib` (mib maps host 19531 → Milvus 19530, so
  the kb scripts' default `http://localhost:19531` works unchanged).
- ssh aliases `msr` (point.dd.works:2333), `msr2` (:2334), `mib` are in
  `~/.ssh/config.dd` on AICR, keyed by `~/.ssh/mib_transfer_ed25519` (pubkey
  installed on all three hosts). Direct `rsync msr:/scratch/sheng/... ` works —
  see `sync_from_msr.sh`.

## Long training on a 24 h partition: `train_chain.sbatch`

`sbatch` once; it re-queues itself with `--dependency=afterany:<self>` at
startup, so the chain lives inside slurm (nothing to babysit on the login
node). Each link:

1. queues its successor (skipped if one is already pending),
2. starts frpc → node reachable at `ssh -p $FRP_PORT dvdai_mit@point.dd.works`
   (default 2335) for debugging. ⚠️ point.dd.works has a firewall: only
   pre-opened ports work (2335 is reserved for AICR; 2333/2334/2336/2337/2338
   belong to the MSR/lab servers' own tunnels — don't take them). An arbitrary
   port (e.g. 2399) registers with frps but is unreachable from outside,
3. opens the Milvus tunnel,
4. runs `$RUN_SCRIPT`, which must resume from its checkpoint dir
   (verl `trainer.resume_mode=auto`).

```bash
# launch
RUN_SCRIPT=/work/mit/ppliang_mit/dvdai/mirl/scripts/self_evolving/train/<run>.sh \
    sbatch scripts/self_evolving/aicr/train_chain.sbatch

# stop the chain (current job keeps running; successors no-op)
touch /scratch/dvdai_mit/self_evolving/chain/STOP.hb-train
# ... or also kill the current link:
scancel --name=hb-train

# resume a stopped chain
rm /scratch/dvdai_mit/self_evolving/chain/STOP.hb-train
```

A clean (rc=0) `RUN_SCRIPT` exit writes the STOP file itself, so a finished
run doesn't loop. Env (`RUN_SCRIPT`, `FRP_PORT`, `CHAIN`) propagates to
successors via `--export=ALL`.

## What's on AICR already

- code: `/work/mit/ppliang_mit/dvdai/mirl` (branch `feat/healthbench-retrieval-augmented`,
  incl. uncommitted work — synced from mib 2026-07-30)
- data: `/work/mit/ppliang_mit/dvdai/self_evolving/` — all SFT trace files +
  `healthbench_pro_val.parquet`
- init checkpoint: `/scratch/dvdai_mit/self_evolving/checkpoints/retrieval_sft_qwen36_27b_selective_step15_hf_merged/`
  (the v7-selective SFT init, 51 GB)
- base model: `Qwen/Qwen3.6-27B` fully cached in `/scratch/dvdai_mit/self_evolving/hf_cache`
  (HF token there too; login-node downloads need `HF_HUB_DISABLE_XET=1` and the
  IPv4 wrapper `/work/mit/ppliang_mit/dvdai/hf_v4.py` — or just run them as cpu jobs)
- SIF: `/scratch/dvdai_mit/self_evolving/sif/` (weianxie has a fallback copy on /work)

## Queued experiment chains (2026-07-30)

| chain job | recipe | init | judge | frp | gen srv |
|---|---|---|---|---|---|
| `hb-evolve` | v13 latest (rubric co-gen + retrieval + frontier /evolve), EXP `healthbench_rubric_qwen36_27b_v14_aicr` | gpt56-SFT step 28 | self (server5) train, gpt-chat-latest val | 2335 | :8006 |
| `p1-mimic` | phase-1 (`run_phase1_mimic.sh`): data evolution only, fixed composite reward, code @ `de576393` in `mirl-phase1/` | ddvd233/mimiciv_rare_qwen36_27b_sft_distill (HF backup) | gpt-chat-latest (train+val) | 2337 | :8007 |

p1-mimic runs "infinite" steps (1e6) — stop it with the STOP file. Both write
checkpoints to /scratch via the compat bind; host-side convenience symlink:
`/work/mit/ppliang_mit/dvdai/checkpoints/scratch_checkpoints`.
