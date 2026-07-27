# HealthBench-Pro Retrieval Investigation — Overnight 2026-07-24

## Bottom line
**Retrieval (knowledge injection) is net-negative on the HealthBench-Professional val set
in every configuration we tried.** The best model remains **no-retrieval stock Qwen3.6-27B
at 0.559**. The mandatory ">0.52 with retrieval at step-0" gate is not achievable with the
current knowledge base and tool design, so the best-achievable overnight run is a
**no-retrieval RL from stock** (honestly pursuing the ~0.56 ceiling).

## The decisive step-0 val matrix (gpt-5.1 judge, 525 tasks, same harness)

| Init / config | val acc | retrieval rate | reasoning (think chars) |
|---|---|---|---|
| **stock, NO tool (single-turn)** | **0.559** | 0% | full |
| wsfix — SFT on 100%-retrieval traces | 0.496 | 100% | ~7700 |
| stock + optional retrieval loop | 0.416 | 76% | full |
| selective SFT (280 direct + 200 retrieval) | 0.488 | ~0% | **killed (110)** |
| stock + RARE-retrieval (fixed tool desc) | 0.490 | **7%** | full (6931) |

Every configuration in which the retrieval tool is *present* lands at 0.42–0.50 — below both
the 0.52 gate and the 0.559 no-tool baseline.

## What we tried and what each taught us
1. **Warm-start SFT on 100%-retrieval traces (wsfix, 0.496).** Baked in an always-retrieve
   habit — retrieval rate 1.00 on all 525 val tasks. Over-retrieval on writing/ethics/direct
   tasks (where lookup only distracts) = the whole gap vs 0.559. The "retrieval is optional"
   prompt could not override the SFT habit.
2. **Selective SFT (0.488).** Rebuilt the SFT data as a 60/40 mix of *direct* (no-search) and
   *retrieval* traces to teach the model *when* to retrieve. It did drop retrieval to ~0% — but
   the short synthetic `<think>` blocks (the teacher endpoint returns no reasoning_content, so
   think blocks are always synthesized) trained the model to **stop reasoning** (think chars
   7700 → 110), and answer quality collapsed. Net worse.
3. **Root-cause on the tool itself.** The tool-schema description literally said *"Call this
   FIRST, before answering."* That is an always-retrieve bias. Rewrote it to a "rare
   last-resort, answer directly by default" description + strengthened the injected instruction.
   Retrieval rate fell **76% → 7%** and reasoning stayed intact — but acc was still **0.490**.
   Conclusion: the mere *presence* of the retrieval apparatus (tool schema + instruction in the
   prompt) degrades even the 93% of answers that stay direct, on top of the 7% retrievals
   scoring low.

## Why retrieval hurts here (mechanism)
- **KB passage quality.** `medical_knowledge_v2` returns title-heavy / off-target rows for many
  of these tasks; wikidoc enrichment halved useless-title hits (26%→12%) but did not fix it.
- **Rubric mismatch.** HealthBench-Pro rubrics reward *comprehensive, broad* answers. Grounding
  in a handful of passages makes the model *less* comprehensive (it anchors to the passages),
  which costs rubric points.
- **Tool-prompt overhead.** Injecting the tool schema + retrieval instruction changes the
  model's framing and makes even direct answers slightly worse (~0.07).
- **Can't inject reasoning via SFT.** The teacher endpoint (server5:18184) returns no
  `reasoning_content`; `enable_thinking=True` yields empty/runaway output. So SFT traces can't
  carry real reasoning, and any SFT that changes behavior tends to degrade the model's own
  strong reasoning.

## Overnight run (best achievable)
`healthbench_rubric_qwen36_27b_v8_noret_stock` — the v6 self-evolving rubric RL recipe,
**single-turn, no retrieval, from stock** (not the mimic-distill init, which was itself a ~1pt
handicap). tmux `train:v8noret`. Expect val_before_train ≈ 0.559 and a v6-style plateau near
0.55–0.56. **The 0.70 target is not reachable via this retrieval approach.**

## Recommended next steps (if pursuing knowledge injection further)
1. **Fix retrieval quality first, offline**: add a reranker, retrieve more + dedup, and expand
   the KB with full-text (not titles). Verify on the ~123 known knowledge-failure val items that
   the *top* passage actually contains the needed fact before wiring it into RL.
2. **Inject retrieval only on verified knowledge-gap tasks** (a gate/classifier), never on
   writing/ethics/translation — so it never touches the tasks it hurts.
3. **Give the model passages as optional reference, not as "evidence to ground in"** — preserve
   its comprehensiveness; measure per-category (writing must not regress).
4. **Accept the ceiling**: if the goal is a shippable model now, the no-retrieval RL (~0.56) is
   the best current option; 0.70 needs either a stronger base model or materially better retrieval.
