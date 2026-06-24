# MIMIC-Rare per-category eval — gpt-5.3-chat judge

28 retained checkpoints across 15 runs, each evaluated on the full 2452-case
MIMIC-IV rare-disease test set (greedy, thinking-off), scored with the training
reward pipeline, judged by gpt-5.3-chat_2026-03-03. 8 ICD-chapter categories
(data_source). 0 eval errors; all images passed (multimodal intact).
Metric below = judge_acc_lenient (overall headline; acc==jLen here).

## Best checkpoint per run (overall)
| run | best step | jLen |
|---|---|---|
| qwen36_27b_full_opd_qwen397b_v2 | 15 | 0.250 |
| qwen36_27b_selfimprove | 20 | 0.242 |
| qwen36_27b_full_gpt55 | 65 | 0.236 |
| gemma4_31b_selfimprove_split | 340 | 0.234 |
| qwen36_27b_deepseekv4pro | 20 | 0.233 |
| gemma4_31b_rl_gpt51judge | 240 | 0.226 |
| qwen36_27b_selfimprove_sft | 220 | 0.212 |
| qwen36_27b_sft_long | 300 | 0.197 |
| qwen36_27b_full_kimi | 90 | 0.197 (fmt 0.84) |
| qwen35_9b_selfimprove | 200 | 0.153 |
| qwen35_9b_sft_long | 700 | 0.121 |
| qwen35_9b_full_gpt55 | 75 | 0.099 |
| gemma4_e4b_rl_gpt51judge | 350 | 0.082 |
| gemma4_e4b_sft.gpt51_archived | 5 | 0.080 |
| gemma4_e4b_sft | 350 | 0.072 |

## Per-category winner
| category (test n) | best run | jLen | range across runs |
|---|---|---|---|
| neoplasms (695) | qwen36_27b_full_opd | 0.459 | 0.112–0.459 |
| digestive (88) | qwen36_27b_full_opd | 0.330 | 0.000–0.330 |
| nervous_system (388) | gemma4_31b_selfimprove_split | 0.309 | 0.095–0.309 |
| blood_immune (223) | gemma4_31b_rl_gpt51judge | 0.233 | 0.022–0.233 |
| circulatory (141) | gemma4_31b_rl_gpt51judge | 0.142 | 0.014–0.142 |
| infectious (102) | qwen36_27b_selfimprove | 0.137 | 0.000–0.137 |
| endocrine_metabolic (494) | qwen35_9b_selfimprove | 0.132 | 0.002–0.132 |
| other (321) | qwen36_27b_deepseekv4pro | 0.087 | 0.022–0.087 |

## Takeaways
- Scale dominates: 27B/31B ~0.20–0.25, 9B ~0.10–0.15, e4b ~0.07–0.08.
- Category difficulty is very uneven: neoplasms is by far the most solvable
  (top 0.46); endocrine/metabolic is hardest (top 0.13) despite being the 2nd
  largest category; the heterogeneous "other" bucket is also hard (0.09).
- Model-specific strengths: gemma4-31B leads blood_immune (0.233) and
  circulatory (0.142), beating the larger qwen-27B there, while qwen-27B leads
  neoplasms/digestive. So the best *overall* model is not best in every chapter.
- Training method: self-improvement clearly helps the small model
  (qwen35-9B selfimprove 0.153 vs full 0.099); OPD distillation from the 397B
  teacher gives the strongest 27B (full_opd 0.250).
- full_kimi's score is suppressed by format adherence (boxes only 84% vs ~1.0).

results.json holds full per-checkpoint × per-category metrics
(acc, judge_acc_lenient/strict, format_ok, score, embed_sim, ...).
