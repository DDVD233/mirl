#!/usr/bin/env python3
"""PRBench validation failure taxonomy (steps 125+130). Print-only; run on pod via ssh stdin.
Aggregates only -- no verbatim criterion text in final report (samples printed here are for
class refinement in tool output only)."""
import json, re, collections, statistics
import pandas as pd

BASE='/scratch/sheng/self_evolving/logs_hb9b/val_generations/prbench9b_specgap_ship_websearch'
df=pd.read_parquet('/scratch/sheng/self_evolving/prbench_hard_val.parquet')
steps=[125,130]
rows={s:[json.loads(l) for l in open(f'{BASE}/{s}.jsonl')] for s in steps}
N=len(df); assert all(len(rows[s])==N for s in steps)

# ---------- criterion classifiers (priority order; first match wins) ----------
def C(*pats): return [re.compile(p,re.I) for p in pats]
POS_CLASSES=[
 ('named-authority', C(r'\b(section|§|\bs\.)\s*\d', r'\brule\s+\d', r'\barticle\s+\d',
    r'\b\d+\s*(U\.?S\.?C|C\.?F\.?R)', r'\b(usc|cfr|frcp|fre)\b', r'\bv\.\s+[A-Z]',
    r'\b(IFRS|IAS|ASC|GAAP|FASB|SFAS|Basel|MiFID|EMIR|UCITS|Solvency II|Dodd[- ]Frank|Sarbanes|UCC|ERISA|HIPAA|GDPR|CCPA|FLSA|FMLA|ADA|ADEA|NLRA|OSHA|RESPA|TILA|FCRA|FDCPA|SEC Rule|Reg(ulation)?\s+[A-Z]{1,4}\b|Treas\.?\s*Reg|IRC\s*§?\s*\d|Restatement|Model\s+(Penal|Rules)|Uniform\s+[A-Z])',
    r'\bcit(e|es|ation|ing)\b.*\b(statute|case|authority|regulation|provision|rule)',
    r'\b(statute|statutory provision|case law|controlling case|precedent)\b.*\b(specif|name|identif|cit)',
    r'\b(names?|identif\w+|references?)\b.*\b(the\s+)?(act|statute|rule|standard|doctrine|case)\b',
    r'\bcit(e|es|ed|ing|ation)\b')),
 ('computation-from-facts', C(r'\b(calculat|comput|arithmetic)\b', r'\bcorrectly\s+(derives?|determines?)\b.*\d',
    r'\b(equal(s|ing)?|approximately|about|roughly|=)\s*[\$€£]?\d', r'\bshows?\s+the\s+(math|work(ing)?)\b',
    r'\b(sum|total|net|difference|product)\b.*\b(of|to)\s*[\$€£]?\d')),
 ('numeric-threshold-recall', C(r'\d+(\.\d+)?\s*%', r'[\$€£]\s?\d', r'\bwithin\s+\d+\s*(days?|months?|years?|hours?|business days?)',
    r'\b\d+[- ](day|month|year|hour)\b', r'\b(deadline|statute of limitations|limitation period|filing period)\b',
    r'\b(threshold|cap|limit|minimum|maximum|floor|ceiling|ratio|rate)\b.*\d', r'\b\d+\s*(basis points|bps)\b')),
 ('jurisdiction-regime', C(r'\bjurisdiction', r'\bchoice[- ]of[- ]law\b', r'\bgoverning law\b',
    r'\b(federal|state)\b.*\b(law|preempt|versus|vs)\b', r'\bpreempt', r'\bforum\b',
    r'\bunder\s+(New York|California|Delaware|Texas|Florida|Illinois|English|EU|German|French|Japanese|UK)\b',
    r'\bapplies?\b.*\b(regime|framework)\b', r'\bwhich\s+(law|court|regulator|agency)\b')),
 ('formula-or-model', C(r'\b(formula|equation|closed[- ]form|expression)\b', r'\bmodel\b.*\b(specif|correct|appropriate|named)',
    r'\b(CAPM|Black[- ]Scholes|VaR|CVaR|WACC|DCF|NPV|IRR|Monte Carlo|copula|GARCH|duration|convexity)\b')),
 ('named-instrument-mechanism', C(r'\bat least (one|two|three|four|\d+)\b',
    r'\bspecific\b.*\b(instrument|order type|product|structure|vehicle|mechanism|tool|strateg|clause|provision|venue|contract type|derivative|metric|ratio|indicator|test|doctrine|remedy|filing|form)',
    r'\b(names?|identif\w+|mentions?|lists?|includes?|recommends?|specifies|proposes?|suggests?)\b.*\((e\.?g\.?|i\.?e\.?|such as)',
    r'\b(names?|identif\w+|mentions?|lists?|includes?|recommends?|specifies)\b.*\bsuch as\b',
    r'\b(names?|identif\w+|mentions?|lists?)\b.*\bspecific\b',
    r'\b(collar|swap(tion)?|forward|futures|option|repo|tranche|covenant|indenture|letter of credit|escrow|earnout|indemnif\w+ clause|NDA)\b')),
 ('conditional-decision-rule', C(r'\bif\b.{3,80}\b(then|must|should|disable|halt|stop|switch|escalate|notify|adjust)\b',
    r'\b(trigger|contingency|fallback|escalation)\b', r'\bwhen\b.{3,60}\b(then|must|should)\b')),
 ('audience-calibration', C(r'\bavoids? (defin|explain|belabor|basic)', r'\bnon[- ]expert\b', r'\bexpert (audience|level|reader)\b',
    r'\bwithout (unnecessary|excessive|basic)\b', r'\bassumes? (familiarity|knowledge)\b')),
 ('procedural-sequencing', C(r'\bstep[- ]by[- ]step\b', r'\b(sequence|sequenc\w+|ordered|chronolog)\b',
    r'\b(first|before)\b.*\bthen\b', r'\bprocess\b.*\b(steps?|stages?|phases?)\b',
    r'\b(procedure|procedural|workflow|timeline)\b', r'\bsteps?\b.*\b(file|filing|obtain|register|notify|appeal)',
    r'\bin\s+the\s+(correct|proper|right)\s+order\b')),
 ('conversation-consistency', C(r'\b(earlier|previous(ly)?|prior)\b.*\b(turn|conversation|message|question|answer|response|discussion|stated)',
    r'\bconsistent with\b', r'\b(recalls?|remembers?|carr\w+ (over|forward))\b',
    r"\b(the user|the client|the customer)'?s?\b.*\b(stated|mentioned|provided|given|earlier|specific (facts|situation|numbers))",
    r'\bfacts? (given|provided|stated|in the (scenario|prompt|question))\b',
    r'\b(uses?|based on|applies?)\b.*\b(the )?(given|provided|stated)\b.*\b(facts|figures|numbers|amounts|dates|details)',
    r'\bth(is|e) specific (situation|case|transaction|client|scenario|facts)\b',
    r"\b(the (client|user|company|fund|portfolio)'s)\b.*\b(specific|actual|particular|concrete)\b")),
 ('format-deliverable', C(r'\b(table|bullet|numbered list|headings?|sections?\s+titled|template|memo(randum)?|letter|email|draft|checklist|outline|summary section|word (limit|count)|concise|brief(ly)? (in|within)|format)\b',
    r'\bstructured?\b.*\b(as|into|with)\b', r'\borganiz\w+\b.*\b(sections?|parts?)\b',
    r'\b(plain (words|language|english)|layman|non[- ]technical|accessible)\b')),
 ('risk-caveat-disclosure', C(r'\b(risk|downside|drawback|limitation|caveat|warn\w*|caution|disclaim\w*|disclos\w+)\b',
    r'\bconsult\b.*\b(attorney|lawyer|counsel|advisor|professional)\b', r'\bnot\b.*\b(legal|financial|tax)\s+advice\b',
    r'\b(trade[- ]?offs?|side effects?|adverse|penalt|liabilit|exposure)\b')),
 ('uncertainty-handling', C(r'\b(uncertain\w*|ambigu\w*|unsettled|unclear|open question)\b',
    r'\b(assumption|assumes?|assuming)\b', r'\b(clarif\w+|asks?\b.*\bquestions?|missing information|more information|additional (facts|details))\b',
    r'\b(may vary|depends on|fact[- ]specific|case[- ]by[- ]case)\b', r'\backnowledg\w+\b.*\b(unknown|not (certain|clear|settled)|limits)\b')),
 ('actionable-recommendation', C(r'\brecommend\w*\b', r'\badvis\w+\b', r'\bsuggests?\b', r'\bpropos\w+\b',
    r'\bactionable\b', r'\bshould\s+(do|take|pursue|consider|avoid)\b', r'\bnext steps?\b',
    r'\bcourse of action\b', r'\bstrateg(y|ies)\b.*\b(specific|concrete|practical)\b')),
 ('quantification-decomposition', C(r'\b(quantif\w+|decompos\w+|numerical(ly)?|order of magnitude|in (points|bps|basis points)|estimates? (the|a|how))\b')),
 ('concrete-anchoring-application', C(r'\b(concrete|historical (event|episode|example|stress)|explicit(ly)? (example|categoriz\w+|map\w+|state\w+ .{0,30}assumption)|maps? (its|the|each)|real[- ]world example|worked example|illustrat\w+ (with|using)|for a representative)\b',
    r'\b(states?|lists?|documents?)\b.{0,50}\bassumptions?\b')),
 ('domain-mechanism-explanation', C(r'\b(explains?|explanation|describ\w+|discuss\w+|articulat\w+|analyz\w+|addresses?|clarifies|distinguish\w+|compares?|defines?|identifies|states?|notes?|mentions?|covers?|acknowledges?|highlights?|recogniz\w+)\b.*\b(that|how|why|the|between|what)\b')),
]
NEG_CLASSES=[
 ('fabricated-or-unsupported-claim', C(r'\b(fabricat|invent|non[- ]?existent|does not exist|made[- ]up|hallucinat|fake|miscit)\b',
    r'\bwithout\b.{0,200}\b(support|evidence|justif|basis|citation|source|substantiat|derivation|showing|explain|disclos)',
    r'\b(unsupported|unsubstantiated|unverifi|uncited|no (source|basis|support))\b')),
 ('wrong-substantive-position', C(r'\b(incorrect(ly)?|wrong(ly)?|misappl\w+|misstates?|mischaracter|confus\w+|conflat\w+|outdated|repealed|superseded|inappropriate|erroneous)\b',
    r'\b(states?|suggests?|claims?|asserts?|concludes?|recommends?|bases)\b.{0,80}\b(any meaningful|feasib|viable|permissible|allowed|can be used|offset)\b')),
 ('numeric-or-calc-error', C(r'\b(mathematical error|miscalculat|calculation error|arithmetic)\b')),
 ('absolutist-overgeneralization', C(r'\b(universal|absolut|always|never\b|all\b.{0,30}(currencies|cases|markets)|guarantee|assur\w+|certainty|overstat|overconfiden|definitive(ly)?)\b')),
 ('over-hedging-no-commitment', C(r'\b(fails? to (commit|take a position|answer|recommend|conclude)|refus\w+|only (generic|general)|non[- ]?committal|avoids? (answering|the question)|deflect|hedges?\b)\b')),
 ('overbreadth-irrelevance', C(r'\bmore than (one|two|three|four|\d+)\b', r'\b(extraneous|unnecessary|irrelevant|not directly actionable|tangential|generic|high[- ]level|laundry list|exceeds?|too many)\b')),
 ('audience-mismatch', C(r'\b(non[- ]expert|basic (industry )?terms|as if addressing|jargon|too technical|lay (audience|person)|condescend)\b',
    r'\b(defines?|explains?)\b.{0,40}\bbasic\b')),
 ('style-format-trap', C(r'\b(reads like|manual|boilerplate|verbose|dense formatting|wall of text|padding|filler|numbered lists with subpoints)\b')),
 ('unsafe-noncompliant-advice', C(r'\b(violat|unethical|illegal|breach|non[- ]?complian|circumvent|evade|evasion|conceal|mislead|frivolous|unauthorized)\b')),
 ('scope-drift', C(r'\b(non[- ]US|outside the|different (market|jurisdiction|asset)|unrelated|not directly actionable|for this specific)\b')),
 ('commits-wrong-position', C(r'\b(states?|claims?|asserts?|concludes?|suggests?|recommends?|bases?|treats?|assumes?|advises?|frames?|presents?|proposes?|argues?|implies)\b')),
 ('includes-unfit-content', C(r'\b(includes?|introduces?|adds?|provides?|uses?|utilizes?|defines?|mentions?|lists?|relies|offers?|gives?|contains?)\b')),
]
def classify(text, classes):
    for name, pats in classes:
        if any(p.search(text) for p in pats): return name
    return 'other'

# ---------- build joined per-criterion table ----------
crit_rows=[]  # dict per (task, item)
task_acc={}
for i in range(N):
    e=df.iloc[i]['extra_info']
    uc=e['use_case']; cats=list(e.get('categories',[]))
    rms={s:json.loads(rows[s][i]['rubric_met']) for s in steps}
    n=len(rms[steps[0]])
    task_acc[i]=statistics.mean(rows[s][i]['acc'] for s in steps)
    for j in range(n):
        c125,c130=rms[125][j],rms[130][j]
        assert c125['criterion']==c130['criterion']
        crit_rows.append(dict(task=i,item=j,uc=uc,pts=c125['points'],
            cat=cats[j] if j<len(cats) else '?',
            text=c125['criterion'],met125=c125['met'],met130=c130['met'],
            nturns=len(df.iloc[i]['prompt'])))
print(f'total criteria: {len(crit_rows)}  tasks: {N}  (finance {sum(df.data_source=="prbench/finance_hard")}, legal {sum(df.data_source=="prbench/legal_hard")})')
pos=[c for c in crit_rows if c['pts']>0]
posw=[c for c in crit_rows if c['pts']>=5]
neg=[c for c in crit_rows if c['pts']<0]
print(f'positive criteria: {len(pos)} (weight>=5: {len(posw)}), negative: {len(neg)}')

unmet_both=[c for c in posw if not c['met125'] and not c['met130']]
met_both  =[c for c in posw if c['met125'] and c['met130']]
mixed     =[c for c in posw if c['met125']!=c['met130']]
print(f'weight>=5: unmet-both {len(unmet_both)}, met-both {len(met_both)}, mixed {len(mixed)}')
print(f'overall mean acc step125 {statistics.mean(r["acc"] for r in rows[125]):.4f}  step130 {statistics.mean(r["acc"] for r in rows[130]):.4f}')

def taxonomy_report(items, label, classes=POS_CLASSES):
    for c in items: c['cls']=classify(c['text'], classes)
    total_pts=sum(abs(c['pts']) for c in items) or 1
    agg=collections.defaultdict(lambda: dict(n=0,pts=0.0,fin=0,leg=0,ptslist=[]))
    for c in items:
        a=agg[c['cls']]; a['n']+=1; a['pts']+=abs(c['pts']); a['ptslist'].append(abs(c['pts']))
        a['fin' if c['uc']=='finance' else 'leg']+=1
    print(f'\n===== {label} (n={len(items)}, total |pts|={total_pts:.0f}) =====')
    print(f'{"class":34s} {"n":>5s} {"pts%":>6s} {"meanpt":>6s} {"fin":>5s} {"leg":>5s}')
    for name,a in sorted(agg.items(), key=lambda kv:-kv[1]['pts']):
        print(f'{name:34s} {a["n"]:5d} {100*a["pts"]/total_pts:6.1f} {statistics.mean(a["ptslist"]):6.2f} {a["fin"]:5d} {a["leg"]:5d}')
    return agg

agg_unmet=taxonomy_report(unmet_both,'UNMET BOTH STEPS (pos, w>=5)')
agg_met=taxonomy_report(met_both,'MET BOTH STEPS (pos, w>=5)')

# met-rate per class (delta view)
print('\n===== per-class met-rate (met-both / (met-both+unmet-both)) =====')
for name in set(list(agg_unmet)+list(agg_met)):
    u=agg_unmet.get(name,{}).get('n',0); m=agg_met.get(name,{}).get('n',0)
    if u+m>0: print(f'{name:34s} met-rate {m/(u+m):5.1%}  (met {m} / unmet {u})')

# 'other' inspection for refinement (tool-output only)
others=[c for c in unmet_both if c['cls']=='other']
print(f'\n--- OTHER bucket ({len(others)}) top bigrams ---')
words=collections.Counter()
for c in others:
    toks=re.findall(r'[a-z]+',c['text'].lower())
    for a,b in zip(toks,toks[1:]): words[a+' '+b]+=1
for w,n in words.most_common(25): print(f'  {n:3d} {w}')
for c in others[:12]: print('  SAMPLE:',c['text'][:130].replace('\n',' '))

# category cross-tab for unmet
print('\n--- unmet-both by benchmark-native category ---')
catagg=collections.Counter(); catpts=collections.Counter()
for c in unmet_both: catagg[c['cat']]+=1; catpts[c['cat']]+=c['pts']
tot=sum(catpts.values())
for k,v in catpts.most_common(): print(f'  {k:45s} n={catagg[k]:4d} pts%={100*v/tot:5.1f}')

# ---------- negative criteria triggered ----------
neg_trig=[c for c in neg if c['met125'] or c['met130']]
neg_both=[c for c in neg if c['met125'] and c['met130']]
print(f'\nnegative criteria: {len(neg)} total; triggered either step {len(neg_trig)}, both steps {len(neg_both)}')
taxonomy_report(neg_trig,'NEGATIVE TRIGGERED (either step)',NEG_CLASSES)
nothers=[c for c in neg_trig if c['cls']=='other']
for c in nothers[:25]: print('  NEG-OTHER SAMPLE:',c['text'][:130].replace('\n',' '))

# task-level trap stats
trap_tasks=collections.Counter()
for c in neg_trig: trap_tasks[c['task']]+=1
neg_acc_tasks=sum(1 for i in range(N) if task_acc[i]<0)
print(f'tasks triggering >=1 negative criterion (either step): {len(trap_tasks)}/{N}; tasks with mean acc<0: {neg_acc_tasks}')
trap_pts_lost=sum(abs(c['pts']) for c in neg_trig)
pos_pts_missed=sum(c['pts'] for c in unmet_both)
print(f'points lost to triggered traps: {trap_pts_lost:.0f} vs positive points missed (w>=5, both): {pos_pts_missed:.0f} (ratio {trap_pts_lost/pos_pts_missed:.2%})')

# ---------- response-side diagnostics ----------
def response_text(r):
    out=r['output']
    k=out.rfind('</think>')
    return out[k+8:] if k>=0 else out
def diag(r):
    t=response_text(r); L=max(len(t),1)
    digits=sum(ch.isdigit() for ch in t)
    cites=len(re.findall(r'(§|\bSection\s+\d|\bRule\s+\d|\bArticle\s+\d|\b\d+\s*(U\.S\.C|C\.F\.R)|\bv\.\s+[A-Z])',t))
    hedges=len(re.findall(r'\b(it depends|consult (a|an|your|with)|generally|typically|may vary|in general|not (legal|financial) advice|professional advice)\b',t,re.I))
    commits=len(re.findall(r'\b(I recommend|you should|the answer is|specifically|the correct)\b',t,re.I))
    return dict(digits_per_1k=1000*digits/L, cites_per_1k=1000*cites/L,
                hedges_per_1k=1000*hedges/L, commits_per_1k=1000*commits/L, resp_chars=L)
order=sorted(range(N), key=lambda i:task_acc[i])
lo,hi=order[:100],order[-100:]
def diag_group(idxs,label):
    ds=[diag(rows[s][i]) for i in idxs for s in steps]
    ns=[rows[s][i]['n_search'] for i in idxs for s in steps]
    accs=[task_acc[i] for i in idxs]
    print(f'\n{label}: mean acc {statistics.mean(accs):.3f}')
    for k in ['digits_per_1k','cites_per_1k','hedges_per_1k','commits_per_1k','resp_chars']:
        print(f'  {k:16s} mean {statistics.mean(d[k] for d in ds):8.2f}  median {statistics.median(d[k] for d in ds):8.2f}')
    print(f'  n_search mean {statistics.mean(ns):.2f}')
    fin=sum(df.iloc[i]["data_source"].endswith("finance_hard") for i in idxs)
    mt=sum(len(df.iloc[i]["prompt"])>1 for i in idxs)
    print(f'  finance {fin}/100, multi-turn {mt}/100')
diag_group(lo,'BOTTOM-100 by acc'); diag_group(hi,'TOP-100 by acc')

# ---------- web-search behavior on recall-class failures ----------
RECALL={'named-authority','numeric-threshold-recall','computation-from-facts','named-instrument-mechanism'}
fail_tasks=sorted({c['task'] for c in unmet_both if c['cls'] in RECALL and c['pts']>=7})
qpat=re.compile(r'<parameter=query>\s*\n?(.*?)\n?\s*</parameter>',re.S)
def queries(i):
    qs=[]
    for s in steps:
        r=rows[s][i]
        qs+= qpat.findall(r['input'])+qpat.findall(r['output'])
    return list(dict.fromkeys(q.strip() for q in qs if q.strip()))
targeted=re.compile(r'(\d|§|\bsection\b|\brule\b|\bact\b|\bcfr\b|\busc\b|\bifrs\b|\bgaap\b|\basc\b|\bbasel\b|"[^"]+")',re.I)
allq=[];tf=0
zero_search=0
for i in fail_tasks:
    qs=queries(i)
    if not qs: zero_search+=1
    for q in qs:
        allq.append(q); tf+= bool(targeted.search(q))
print(f'\n===== SEARCH BEHAVIOR on tasks with >=1 high-weight(>=7) recall-class miss =====')
print(f'tasks: {len(fail_tasks)}; tasks with zero extracted queries: {zero_search}')
print(f'queries: {len(allq)}; targeted-fact: {tf} ({tf/max(len(allq),1):.1%}), topical-background: {len(allq)-tf} ({(len(allq)-tf)/max(len(allq),1):.1%})')
qlens=[len(q.split()) for q in allq]
if qlens: print(f'query length words: mean {statistics.mean(qlens):.1f} median {statistics.median(qlens)}')
print('sample queries (tool-output only):')
for q in allq[:15]: print('  Q:',q[:110])
# search volume vs acc
ns_all=collections.defaultdict(list)
for i in range(N):
    ns=statistics.mean(rows[s][i]['n_search'] for s in steps)
    ns_all[min(int(ns),3)].append(task_acc[i])
print('acc by mean n_search bucket:', {k:(len(v),round(statistics.mean(v),3)) for k,v in sorted(ns_all.items())})

# ---------- multi-turn constraint check material (15 low-acc multi-turn) ----------
mt_lo=[i for i in order if len(df.iloc[i]['prompt'])>=3][:15]
print('\n===== 15 LOW-ACC MULTI-TURN CASES (for by-eye constraint-consistency judging; tool-output only) =====')
for i in mt_lo:
    p=df.iloc[i]['prompt']
    print(f'\n### task {i} acc={task_acc[i]:.3f} turns={len(p)} uc={df.iloc[i]["extra_info"]["use_case"]}')
    for t in p[:-1]:
        if t['role']=='user': print('  EARLIER-USER:',t['content'][:220].replace('\n',' '))
    print('  FINAL-USER:',p[-1]['content'][:250].replace('\n',' '))
    rt=response_text(rows[125][i])
    print('  RESPONSE-HEAD:',rt[:500].replace('\n',' '))
    print('  RESPONSE-TAIL:',rt[-300:].replace('\n',' '))

print('\n===== FULL CONTEXT: ambiguous constraint cases 264,250,396,279 =====')
for i in [264,250,396,279]:
    p=df.iloc[i]['prompt']
    print(f'\n### task {i}')
    for t in p[:-1]:
        if t['role']=='user': print('  EARLIER-USER(full-500):',t['content'][:500].replace('\n',' '))
    print('  FINAL-USER(500):',p[-1]['content'][:500].replace('\n',' '))
    rt=response_text(rows[125][i])
    print('  RESPONSE(1500):',rt[:1500].replace('\n',' '))
