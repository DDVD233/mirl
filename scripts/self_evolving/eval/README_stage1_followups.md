# Stage-1 follow-ups (2026-09-08)

All MSR access uses the existing SSH/FRP endpoints. No Kubernetes commands are
needed. `S=/scratch/sheng/self_evolving` below. Credentials stay in `$S/.trapi_key`.

## Decisions

- Static-pool training is the requested control for fixed training material; no
  additional clean-seed or matched-teacher reruns are being launched.
- The additional curated-data control uses 9B and a frozen, non-SFT 9B self-judge.
- The paper compares the best full SFT+RL checkpoint (51.59%) with the best static
  pool checkpoint (44.13%), not the unfinished curated-data run as its main control.
  Teacher arms without SFT remain explicitly labeled as such.
- Lack of a saturated curated-data run is not presented as a limitation.

## 9B curated-data control

Node: `root@point.dd.works:2335`. tmux windows in session `main`:
`stage1-control`, `stage1-selfjudge`, `stage1-regrade`.

The isolated checkout `$S/verl_stage1_9b_control` is pinned to
`01c43aa6bff049f01088ff474079e40da7f612a1`. Its reward implementation is copied
from the existing stage-1 checkout, including its local changes. The padded-path
entropy patch is recorded in `train/stage1_9b_entropy.patch`.

Initialization is the archived actor from
`ddvd233/mimiciv_rare_qwen35_9b_sft_distill_global_step_90`, restored under
`$S/checkpoints/restored/qwen35_9b_sft_distill_step90/actor/huggingface`.
The downloaded HF revision is `77d537002efe3540c035273b4511f742d49f0d42`;
the weights' recorded LFS SHA-256 is
`5c64f0a3c11690d7266befca1a8f25ecacb4a477b77a1ced60b5448c41bb6b7a`.
GPU 0/1 train; GPU 3 serves frozen `Qwen/Qwen3.5-9B` on localhost:8188.
This is a new 9B control, not a continuation of the old 27B optimizer state.

`train/run_stage1_9b_control.sh` records the configuration: all 5719 training
admissions, GRPO batch 64, 8 samples, learning rate 2e-7, no KL, 1000 steps,
validation/checkpoint every 20 steps. Validation is greedy on all 2452 admissions.
The pinned training dataset loader retains its original image handling, including
truncated-image loading and placeholder fallback. This differs from the explicitly
logged image-omission policy in the frozen-backbone inference baselines.

Console logs and offline W&B files are under `$S/logs_stage1_9b_control`.
The independent regrader writes `fixed_judge/{step}.summary.json` and `curve.json`
there, with input hashes and full coverage checks. It uses the fixed strict paper
judge, `gpt-chat-latest_2026-05-28`; training rewards remain self-judged.
The first step-0 score was 36.05%. A subsequent complete path audit found 2238
unique TRAINING images missing at historical paths, but zero missing validation
images and byte-identical ECG copies. Thus the initial validation score was not
affected; the initial concern that it was invalid was disproved. Its artifacts
are retained under `*_before_path_fix` names, separately from the clean rerun.
`train/prepare_stage1_control_data.py` now makes isolated path-corrected inputs,
retaining all cases and checking that every referenced file exists. It resolves
3359 training and 1554 evaluation ECG references, with zero missing image files.
Hashes are recorded in `control_data_manifest.json`; original datasets are unchanged.
An isolated system-site-packages environment at `$S/venvs/stage1_9b_control`
adds `flash-linear-attention==0.5.2` and `fla-core==0.5.2`; the kernel passed a
GPU forward/backward test. The first slow torch-fallback attempt was stopped
before any update and its log retained as `train_torch_fallback.log`. The corrected
run repeats initial validation with the resolved image paths. Convolution still uses its torch
fallback; the expensive gated-delta operation uses FLA.
The clean run completed its first training update on 2026-09-08 at about 04:42 UTC.
Its fresh full initial validation scores 36.70% under the fixed paper judge; this
is a starting-checkpoint measurement, not an RL improvement.

`main:stage1-supervisor` checks owned panes every five minutes and restarts only
dead panes after their assigned GPUs are released. It never kills a live process.
The local `sync_stage1_followups.py` collector publishes only full component
summaries and fully regraded post-initialization control checkpoints, then rebuilds
the PDF. Step 0 is excluded from the automatic trained-control table row.
At the user's request, the local collector now runs with `--push`: after a
successful build it commits and pushes completed table/PDF updates to the paper
repository's `origin/main`. It refuses pre-existing edits or staging, never stages
raw traces, and retries failed pushes without force-pushing. Private run archives
are required for full figure regeneration and are not published with the tooling.

## Component audit

Node: `root@point.dd.works:2333`, tmux `main:stage1-components`.
Code, manifests, traces and resumable outputs: `$S/stage1_component_audit`.
The launcher retries missing items after API failures; malformed generated tasks
are recorded as outcomes, not regenerated until they pass.

The historical validator audit samples 200 accepted and 200 rejected tasks from
the full-pipeline archive. Independent model assessments are blinded to historical
decisions and use freshly retrieved references. These are not clinician labels,
and the references do not reconstruct the original validator's retrieved context.
Population-weighted recall accounts for the unequal stratum sizes; uncertain labels
are excluded from decisive precision/recall but remain in the reported counts.
The complete summary is archived at `paper_data/stage1/component_audit` and in
the paper's Judging appendix.

The paired generation experiment uses 64 text-only training seeds, balanced between
MCQ and free response, with 8 solver samples per formatted generated task. Arms:

- 16 passages, 50% recent solver accuracy feedback (reference).
- 3 passages or no passages, keeping 50% feedback.
- 16 passages with difficulty feedback removed.
- 16 passages with recent solver accuracy feedback set to 20% or 80%.

These are generation-level grounding/difficulty checks, not downstream RL outcome
ablations or a complete reproduction of every original retrieval setting. They
report task validity, validator acceptance, solver accuracy and mixed-reward groups.
The 12-condition reduced smoke test passed after restoring the original boxed
verdict protocol for free-answer grading. Only complete full-run summaries should
enter paper tables; smoke outputs are not scientific results.

## Paper regeneration

Run `.venv/bin/python visualizations/paper/refresh_stage1_comparisons.py`, then
`latexmk -pdf -interaction=nonstopmode -halt-on-error iclr2026_conference.tex`
inside `paper/`. The script validates all 19 archived official HealthBench runs
against their 525 example-level grades and rejects duplicate IDs or mixed graders.
Input hashes are in `paper_data/stage1/healthbench_official_refresh.json`.

No official trained-9B HealthBench artifact or completed 397B base-model run was
found in the inspected archive. The previous AICR 397B job failed during vLLM
startup while flashinfer created cubin symlinks. Missing results remain unreported;
old-grader scores must not be substituted into the refreshed official figures.
The 397B rerun is AICR job `716803`, label `qwen35_397b_fp8_v2`, with eight B200s
requested and FlashInfer's writable package directory bound to node-local storage.
This is a submitted rerun, not a completed result.
Two pre-script launches failed on b0006. Only this job was updated to exclude
b0006 and released; it subsequently returned to ordinary priority-based pending.
