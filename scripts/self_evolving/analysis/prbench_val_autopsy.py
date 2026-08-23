#!/usr/bin/env python3
"""Autopsy of PRBench-Hard val generations: why does the arm improve slowly?

Read-only, deterministic. Run on the pod:
    python3 /scratch/sheng/self_evolving/prbench_val_autopsy.py
"""
import json, os, sys
import pandas as pd
import numpy as np

DUMP = "/scratch/sheng/self_evolving/logs_hb9b/val_generations/prbench9b_specgap_ship_websearch"
PARQ = "/scratch/sheng/self_evolving/prbench_hard_val.parquet"
STEPS = [0, 15, 125, 130]

def load_step(s):
    rows = []
    with open(os.path.join(DUMP, f"{s}.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            r["rubric_met"] = json.loads(r["rubric_met"]) if isinstance(r["rubric_met"], str) else r["rubric_met"]
            rows.append(r)
    return rows

steps = {s: load_step(s) for s in STEPS}
for s in STEPS:
    assert len(steps[s]) == 550, (s, len(steps[s]))

pq = pd.read_parquet(PARQ)
ei = [pq.iloc[i]["extra_info"] for i in range(len(pq))]
def aslist(x):
    if isinstance(x, str):
        import ast
        return ast.literal_eval(x)
    return list(x)
rubrics = [aslist(e["rubric_items"]) for e in ei]     # [{criterion_text, points}]
cats    = [list(e["categories"]) for e in ei]
convs   = [aslist(e["conversation"]) for e in ei]
use_case = [e["use_case"] for e in ei]

print("== sanity ==")
mm = sum(1 for i in range(550) if len(steps[0][i]["rubric_met"]) != len(rubrics[i]))
cm = sum(1 for i in range(550) if len(cats[i]) != len(rubrics[i]))
print(f"tasks where len(rubric_met)!=len(rubric_items): {mm}; len(categories)!=len(rubric_items): {cm}")
# spot check criterion text alignment
al = 0
for i in range(0, 550, 50):
    a = steps[0][i]["rubric_met"][0]["criterion"][:60]
    b = rubrics[i][0]["criterion_text"][:60]
    al += (a == b)
print(f"first-criterion text match on 11 spot checks: {al}/11")
kf = {s: np.mean([r["judge_fail"] for r in steps[s]]) for s in STEPS}
print("judge_fail rate by step:", {s: round(v,3) for s,v in kf.items()})

def acc_mean(s_list):
    # average acc over the given steps, per task then overall
    per = np.mean([[r["acc"] for r in steps[s]] for s in s_list], axis=0)
    return per
acc0, acc15 = acc_mean([0]), acc_mean([15])
accL = acc_mean([125, 130])
print(f"\nmean acc: step0={acc0.mean():.4f}  step15={acc15.mean():.4f}  late(125+130)={accL.mean():.4f}  step125={acc_mean([125]).mean():.4f} step130={acc_mean([130]).mean():.4f}")

def tier(p):
    p = abs(p)
    return "crit(9-10)" if p >= 9 else ("imp(5-8)" if p >= 5 else "slight(1-4)")

# ---------- 1. score decomposition ----------
print("\n== 1. positive-criteria met rate by |points| tier, negative trigger rate ==")
def decomp(s_list):
    met = {}; tot = {}
    neg_trig = 0; neg_tot = 0
    for s in s_list:
        for i, r in enumerate(steps[s]):
            for j, c in enumerate(r["rubric_met"]):
                pts = rubrics[i][j]["points"] if j < len(rubrics[i]) else c.get("points", 0)
                if pts is None: pts = c.get("points", 0)
                m = bool(c["met"])
                if pts >= 0:
                    t = tier(pts)
                    tot[t] = tot.get(t, 0) + 1
                    met[t] = met.get(t, 0) + m
                else:
                    neg_tot += 1
                    neg_trig += m
    out = {t: met[t] / tot[t] for t in tot}
    out["NEG-trigger"] = neg_trig / max(neg_tot, 1)
    out["_neg_n"] = neg_tot / len(s_list)
    out["_pos_counts"] = {t: tot[t] // len(s_list) for t in tot}
    return out
d0, d15, dL = decomp([0]), decomp([15]), decomp([125, 130])
print(f"{'tier':<12}{'n/step':>8}{'step0':>9}{'step15':>9}{'late':>9}{'delta':>9}")
for t in ["crit(9-10)", "imp(5-8)", "slight(1-4)"]:
    print(f"{t:<12}{d0['_pos_counts'][t]:>8}{d0[t]:>9.3f}{d15[t]:>9.3f}{dL[t]:>9.3f}{dL[t]-d0[t]:>+9.3f}")
print(f"{'NEG-trigger':<12}{int(d0['_neg_n']):>8}{d0['NEG-trigger']:>9.3f}{d15['NEG-trigger']:>9.3f}{dL['NEG-trigger']:>9.3f}{dL['NEG-trigger']-d0['NEG-trigger']:>+9.3f}")
# points-weighted contribution to the remaining gap at late steps
gap = {}
for i, r in enumerate(steps[130][0:0] or steps[130]):
    pass
lost_pts = {}
tot_pts = 0.0
for i in range(550):
    denom = sum(x["points"] for x in rubrics[i] if x["points"] > 0)
    for j, c in enumerate(steps[130][i]["rubric_met"]):
        pts = rubrics[i][j]["points"]
        if pts > 0 and not c["met"]:
            lost_pts[tier(pts)] = lost_pts.get(tier(pts), 0) + pts / denom
    tot_pts += 1
print("share of remaining (1-acc) gap at step130 by tier (points-weighted, per-task-normalized):",
      {t: round(v / tot_pts, 3) for t, v in sorted(lost_pts.items())})

# ---------- 2. per-category met rate ----------
print("\n== 2. per-category positive-criteria met rate (points-weighted), step0 vs late ==")
def cat_table(s_list):
    w_met = {}; w_tot = {}
    for s in s_list:
        for i, r in enumerate(steps[s]):
            for j, c in enumerate(r["rubric_met"]):
                if j >= len(cats[i]): continue
                pts = rubrics[i][j]["points"]
                if pts <= 0: continue
                cat = cats[i][j]
                w_tot[cat] = w_tot.get(cat, 0) + pts
                w_met[cat] = w_met.get(cat, 0) + pts * bool(c["met"])
    return {c: (w_met.get(c, 0) / w_tot[c], w_tot[c] / len(s_list)) for c in w_tot}
c0, cL = cat_table([0]), cat_table([125, 130])
print(f"{'category':<42}{'w/step':>8}{'step0':>8}{'late':>8}{'delta':>8}")
for c in sorted(cL, key=lambda c: cL[c][0]):
    print(f"{c:<42}{c0[c][1]:>8.0f}{c0[c][0]:>8.3f}{cL[c][0]:>8.3f}{cL[c][0]-c0[c][0]:>+8.3f}")

# ---------- 3. multi-turn & long prompts ----------
print("\n== 3. scores by #user turns and prompt length ==")
n_user = np.array([sum(1 for t in convs[i] if t["role"] == "user") for i in range(550)])
def bucket(n):
    return "5+" if n >= 5 else ("2-4" if n >= 2 else "1")
buck = np.array([bucket(n) for n in n_user])
print(f"{'user turns':<10}{'n':>5}{'step0':>8}{'late':>8}{'delta':>8}")
for b in ["1", "2-4", "5+"]:
    m = buck == b
    print(f"{b:<10}{m.sum():>5}{acc0[m].mean():>8.3f}{accL[m].mean():>8.3f}{accL[m].mean()-acc0[m].mean():>+8.3f}")
inlen = np.array([len(steps[130][i]["input"]) for i in range(550)])
long_m = inlen > 45000
print(f"long prompts (input>45k chars): n={long_m.sum()}  step0={acc0[long_m].mean():.3f}  late={accL[long_m].mean():.3f}  (rest late={accL[~long_m].mean():.3f})")
print(f"  of long prompts, multi-turn(2+): {sum(1 for i in range(550) if long_m[i] and n_user[i]>=2)}")
# use_case split for context
for uc in ["finance", "legal"]:
    m = np.array([u == uc for u in use_case])
    print(f"use_case={uc:<8} n={m.sum():>3}  step0={acc0[m].mean():.3f}  late={accL[m].mean():.3f}")

# ---------- 4. answer characteristics vs score at 130 ----------
print("\n== 4. answer characteristics vs acc at step 130 ==")
s130 = steps[130]
def ans_from_output(r):
    o = r["output"] or ""
    if "</think>" in o:
        o = o.rsplit("</think>", 1)[1]
    return o.strip()
ans_len = np.array([len(ans_from_output(r)) for r in s130])
out_len = np.array([len(r["output"] or "") for r in s130])
print(f"NOTE: extracted_answer is truncated at 512 chars in the dump; using output-after-</think> instead.")
print(f"output chars: mean={out_len.mean():.0f} max={out_len.max()}")
thk = np.array([r["think_chars"] for r in s130])
nsr = np.array([r["n_search"] for r in s130])
tc0 = sum(1 for r in s130 if r["think_closed"] == 0)
a130 = np.array([r["acc"] for r in s130])
ncrit = np.array([len(rubrics[i]) for i in range(550)])
print(f"answer chars: mean={ans_len.mean():.0f} median={np.median(ans_len):.0f} p90={np.percentile(ans_len,90):.0f}")
print(f"think chars: mean={thk.mean():.0f}; n_search: mean={nsr.mean():.2f}; think_closed=0 count: {tc0}")
print(f"(step0 answer chars mean={np.mean([len(ans_from_output(r)) for r in steps[0]]):.0f}, n_search mean={np.mean([r['n_search'] for r in steps[0]]):.2f})")
dfc = pd.DataFrame({"acc": a130, "ans_len": ans_len, "think": thk, "n_search": nsr, "n_crit": ncrit, "in_len": inlen})
sp = dfc.corr(method="spearman")["acc"]
print("spearman vs acc@130:", {k: round(v, 3) for k, v in sp.items() if k != "acc"})
q = pd.qcut(ans_len, 5, duplicates="drop")
print("acc by answer-length quintile:")
g = pd.DataFrame({"q": q, "acc": a130, "len": ans_len}).groupby("q", observed=True)
for name, gr in g:
    print(f"  len {gr['len'].min():>5.0f}-{gr['len'].max():>5.0f}: acc={gr['acc'].mean():.3f} (n={len(gr)})")
# met rate vs rubric size
q2 = pd.qcut(ncrit, 4, duplicates="drop")
print("acc@130 by rubric size quartile:")
for name, gr in pd.DataFrame({"q": q2, "acc": a130, "n": ncrit}).groupby("q", observed=True):
    print(f"  n_crit {gr['n'].min():>2}-{gr['n'].max():>2}: acc={gr['acc'].mean():.3f} (n={len(gr)})")

# ---------- 5. failure texture ----------
print("\n== 5. thirty highest-weight unmet POSITIVE criteria at step130 (max 1/task, points>=9) ==")
cand = []
for i in range(550):
    best = None
    for j, c in enumerate(s130[i]["rubric_met"]):
        pts = rubrics[i][j]["points"]
        if pts >= 9 and not c["met"]:
            if best is None or pts > best[0]:
                best = (pts, i, j)
    if best: cand.append(best)
cand.sort(key=lambda x: (-x[0], x[1]))
for pts, i, j in cand[:30]:
    txt = rubrics[i][j]["criterion_text"].replace("\n", " ")[:120]
    print(f"  [task{i:>3} p{pts} {cats[i][j][:24]:<24}] {txt}")
print(f"(tasks with >=1 unmet crit>=9 at step130: {len(cand)}/550)")

lost = won = stay_met = stay_un = 0
lost_by_tier = {}; won_by_tier = {}
for i in range(550):
    for j in range(min(len(rubrics[i]), len(s130[i]["rubric_met"]), len(steps[0][i]["rubric_met"]))):
        pts = rubrics[i][j]["points"]
        if pts <= 0: continue
        m0, m1 = bool(steps[0][i]["rubric_met"][j]["met"]), bool(s130[i]["rubric_met"][j]["met"])
        if m0 and not m1: lost += 1; lost_by_tier[tier(pts)] = lost_by_tier.get(tier(pts), 0) + 1
        elif m1 and not m0: won += 1; won_by_tier[tier(pts)] = won_by_tier.get(tier(pts), 0) + 1
        elif m0: stay_met += 1
        else: stay_un += 1
print(f"criterion churn step0->130 (positive only): newly-won={won}, LOST={lost}, stayed-met={stay_met}, stayed-unmet={stay_un}")
print(f"  won by tier: {won_by_tier}; lost by tier: {lost_by_tier}")

# ---------- 6. negative criteria ----------
print("\n== 6. negative criteria triggered in >5% of their tasks at late steps ==")
negs = {}
for i in range(550):
    for j, x in enumerate(rubrics[i]):
        if x["points"] < 0:
            key = x["criterion_text"]
            t0 = bool(steps[0][i]["rubric_met"][j]["met"])
            tl = (bool(steps[125][i]["rubric_met"][j]["met"]) + bool(s130[i]["rubric_met"][j]["met"])) / 2
            e = negs.setdefault(key, [0, 0.0, 0, x["points"]])
            e[0] += t0; e[1] += tl; e[2] += 1
# aggregate identical texts across tasks; also overall
tot_n = sum(e[2] for e in negs.values())
trig0 = sum(e[0] for e in negs.values()); trigL = sum(e[1] for e in negs.values())
print(f"total negative criteria: {tot_n} across tasks; overall trigger rate step0={trig0/max(tot_n,1):.3f} late={trigL/max(tot_n,1):.3f}")
shown = 0
for key, e in sorted(negs.items(), key=lambda kv: -(kv[1][1] / kv[1][2])):
    rate_l = e[1] / e[2]
    if rate_l > 0.05 and e[2] >= 1:
        print(f"  [n={e[2]} p{e[3]}] step0={e[0]/e[2]:.2f} late={rate_l:.2f} :: {key[:110]}")
        shown += 1
        if shown >= 15: break
if shown == 0:
    print("  (none above 5%)")

# ---------- 7. examples ----------
print("\n== 7. examples ==")
def show(i, label):
    last_u = [t for t in convs[i] if t["role"] == "user"][-1]["content"]
    r = s130[i]
    print(f"--- {label}: task {i} ({use_case[i]}, {ei[i]['specialty']}), acc@130={r['acc']:.3f}, user_turns={n_user[i]}, ans_len={len(r['extracted_answer'] or '')} ---")
    print("LAST USER TURN:", last_u[:300].replace("\n", " "))
    print("ANSWER:", (r["extracted_answer"] or "")[:600].replace("\n", " "))
    unmet = [(rubrics[i][j]['points'], rubrics[i][j]['criterion_text'][:100]) for j, c in enumerate(r["rubric_met"]) if rubrics[i][j]["points"] > 0 and not c["met"]]
    print("UNMET (+):")
    for p, t in sorted(unmet, reverse=True)[:8]:
        print(f"   p{p}: {t}")
# a) high-weight recall miss: pick from cand a criterion that looks like specific-fact recall (contains a digit)
pick = None
for pts, i, j in cand:
    if any(ch.isdigit() for ch in rubrics[i][j]["criterion_text"]):
        pick = i; break
show(pick if pick is not None else cand[0][1], "high-weight recall miss")
# b) multi-turn near-0
mt = [i for i in range(550) if n_user[i] >= 2]
mt.sort(key=lambda i: a130[i])
show(mt[0], "multi-turn near-zero")
print("\nDone.")
