export const meta = {
  name: 'classify-retrieval-regressions',
  description: 'Classify each retrieval-introduced regression (data/prompt/other) and propose fixes',
  phases: [
    { title: 'Classify', detail: 'per-case root-cause classification in batches' },
    { title: 'Synthesize', detail: 'aggregate breakdown + prioritized fixes' },
  ],
}

const PATH = '/tmp/claude-1001/-home-dvd-mirl/ff781709-9841-4dff-a82a-bdfedf8f2645/scratchpad/regress.json'
const N = 134
const BATCH = 8

const CASE_SCHEMA = {
  type: 'object',
  properties: {
    results: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          idx: { type: 'integer', description: 'the case "idx" field (val question index)' },
          category: { enum: ['data', 'prompt', 'other'] },
          primary_cause: {
            enum: [
              'bad_passages',              // data: retrieved passages off-target/wrong/title-only, misled the answer
              'missing_relevant_passages', // data: KB lacks the needed fact (post-cutoff, exact code); retrieval could not help
              'over_grounding',            // prompt: model suppressed correct known facts / deferred to passages / "evidence does not contain"
              'unnecessary_retrieval',     // prompt: task needed no lookup (writing/ethics/translation); retrieving changed style/added noise
              'comprehensiveness_loss',    // other: answer narrower/shorter (breadth rubric) even w/ ok passages, not clearly over-grounding
              'format_leakage',            // other: tool/planning text leaked into answer, or answer truncated by tool overhead
              'tool_prompt_overhead',      // other: direct answer (no retrieval used) worse due to tool schema/instruction in prompt
              'judge_noise',               // other: flip looks like judge variance; answers comparable quality
              'other',
            ],
          },
          evidence: { type: 'string', description: 'specific 1-2 sentence evidence citing the passages/answers' },
          fix: { type: 'string', description: 'concrete actionable fix (prompt edit / data change / gate)' },
        },
        required: ['idx', 'category', 'primary_cause', 'evidence', 'fix'],
      },
    },
  },
  required: ['results'],
}

const batches = []
for (let s = 0; s < N; s += BATCH) batches.push([s, Math.min(s + BATCH, N)])

phase('Classify')
const classified = await parallel(batches.map(([s, e]) => () =>
  agent(
    `You are diagnosing why adding a medical-retrieval tool to a strong model (Qwen3.6-27B) made specific HealthBench answers WORSE. ` +
    `Same base model with vs without retrieval; these are REGRESSIONS (no-retrieval got it right, retrieval got it wrong).\n\n` +
    `Read your assigned cases (list positions ${s}..${e - 1}) by running this exact Bash command:\n` +
    `  python3 -c "import json; d=json.load(open('${PATH}')); import sys; print(json.dumps(d[${s}:${e}]))"\n\n` +
    `Each case has: question, nr_answer (no-retrieval, CORRECT), re_answer (retrieval, WRONG), re_query, re_passages, ` +
    `criteria_lost (rubric points that flipped from met->unmet), retrieval_used (bool), re_think_chars.\n\n` +
    `For EACH case, decide the SINGLE primary_cause of the regression and category (data / prompt / other). ` +
    `Judge honestly: is it the PASSAGES being bad/missing (data), the model MISUSING retrieval or retrieving when it should not have (prompt), ` +
    `a comprehensiveness/format/overhead effect (other), or just judge_noise (answers really are comparable)? ` +
    `Look at what criteria_lost says and compare nr_answer vs re_answer concretely. If retrieval_used is false, the regression came from the ` +
    `tool prompt overhead on a direct answer (tool_prompt_overhead) unless it is clearly judge_noise. ` +
    `Give specific evidence and a concrete fix. Return every case in your batch.`,
    { label: `classify:${s}-${e - 1}`, phase: 'Classify', schema: CASE_SCHEMA }
  ).then(r => (r && r.results) ? r.results : [])
))

const all = classified.flat().filter(Boolean)

// tally for the synthesis prompt
const byCause = {}, byCat = {}
for (const r of all) { byCause[r.primary_cause] = (byCause[r.primary_cause] || 0) + 1; byCat[r.category] = (byCat[r.category] || 0) + 1 }

phase('Synthesize')
const SYNTH_SCHEMA = {
  type: 'object',
  properties: {
    breakdown: { type: 'string', description: 'counts by category and primary_cause' },
    top_causes: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          cause: { type: 'string' },
          count: { type: 'integer' },
          category: { type: 'string' },
          representative_idxs: { type: 'array', items: { type: 'integer' } },
          fix: { type: 'string', description: 'the concrete fix that would close the most cases' },
          expected_gain: { type: 'string', description: 'rough #cases recoverable if fixed' },
        },
        required: ['cause', 'count', 'category', 'fix', 'expected_gain'],
      },
    },
    round1_recommendation: { type: 'string', description: 'the single highest-ROI fix to implement first and why' },
  },
  required: ['breakdown', 'top_causes', 'round1_recommendation'],
}

const synth = await agent(
  `You are synthesizing a root-cause analysis of ${all.length} regressions where adding a retrieval tool made a strong model WORSE on HealthBench ` +
  `(0.559 no-retrieval -> 0.416 with retrieval, same base). Goal: pick the fixes that close the gap ITERATIVELY, biggest lever first.\n\n` +
  `Tally so far — by category: ${JSON.stringify(byCat)} ; by primary_cause: ${JSON.stringify(byCause)}.\n\n` +
  `Full per-case classifications (JSON): ${JSON.stringify(all).slice(0, 90000)}\n\n` +
  `Produce: (1) a clear breakdown; (2) top_causes ranked by count with representative idxs and the concrete fix for each ` +
  `(distinguish DATA fixes = better passages/rerank/KB from PROMPT fixes = grounding/gating instruction from OTHER); ` +
  `(3) round1_recommendation = the single highest-ROI fix to implement first, with reasoning about expected gain.`,
  { label: 'synthesize', phase: 'Synthesize', schema: SYNTH_SCHEMA }
)

return { total: all.length, byCategory: byCat, byCause, synthesis: synth, all }
