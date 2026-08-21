"""Domain bundles for the self-evolving generator (dvd 2026-08-21).

The generator, evolver, adversary (probe / patch / hack memo / farmer), referee and
solver-tool prompts were written for HealthBench-Professional and say so in every
sentence. To run the SAME pipeline against a different held-out rubric benchmark
(PRBench: finance + legal; ProfBench: PhD/MBA report tasks) the prompts need the
domain swapped while every calibration invariant (criterion counts, point scales,
hedged wording, difficulty target) stays byte-identical.

Selection is by env `SE_DOMAIN` (default `medical`). A bundle carries:
  * the use-case / specialty taxonomy the proposer samples from,
  * DATASET_BRIEF: the GENERAL description of the target benchmark shown to the
    generator and meta-optimizer. It is written from the benchmark's paper/README
    (domain, authorship, task style, rubric categories, weight classes) and MUST
    NEVER contain an item, a rubric line, or a paraphrase of one. The experiment's
    whole point is zero access to the benchmark's contents,
  * [[DOMAIN_*]] fillers for the shared prompt templates,
  * REBRAND: ordered phrase substitutions applied to the remaining medical-worded
    constants (evolve analyst, hack minter/memo, farmer, referee, tool instructions).

The medical bundle must reproduce the pre-existing prompts exactly; a test asserts
that so the HealthBench arms stay byte-identical.
"""
from __future__ import annotations

import os

DOMAIN = os.environ.get("SE_DOMAIN", "medical").strip().lower()


def _prbench() -> dict:
    finance_topics = [
        "corporate_finance", "taxation_cross_border_structuring",
        "accounting_financial_statement_analysis", "risk_management_stress_testing",
        "market_microstructure_trading_liquidity", "international_finance_fx_hedging",
        "regulation_compliance_ethics", "wealth_management_financial_planning",
        "investment_strategy_portfolio_design", "derivatives_structured_products",
        "alternative_investments_private_markets", "fintech_crypto_digital_assets",
        "financial_markets_institutions_macro_policy",
    ]
    legal_topics = [
        "regulatory_administrative_law", "litigation_dispute_resolution",
        "corporate_transactional_law", "real_estate_property_law",
        "labor_employment_law", "international_law_trade",
        "taxation_cross_border_structuring", "intellectual_property_law",
        "criminal_law", "family_law", "legal_ethics_compliance_risk_management",
        "constitutional_civil_rights_law",
    ]
    return dict(
        NAME="prbench",
        BENCH_NAME="PRBench (Professional Reasoning Bench)",
        USE_CASES={"finance": 0.55, "legal": 0.45},
        USE_CASE_DESC={
            "finance": "a finance professional's real working question: analysis, valuation, "
                       "tax or structuring, risk, accounting, compliance, trading or planning "
                       "decisions with real numbers and a decision to make",
            "legal": "a legal professional's real working question: applying statutes, "
                     "regulations, case law and procedure to a concrete fact pattern in a named "
                     "jurisdiction, with exposure, obligations and next steps to determine",
        },
        SPECIALTIES_BY_USE_CASE={"finance": finance_topics, "legal": legal_topics},
        SPECIALTIES=sorted(set(finance_topics + legal_topics)),
        REDTEAM_SHARE=0.15,
        MODE_INSTR={
            "good_faith": "MODE: good faith. A professional using the AI normally for accurate, "
                          "useful help on a live matter.",
            "red_teaming": (
                "MODE: stress test. Build a case that contains a false or outdated premise, a "
                "rule misremembered as fact, a jurisdiction or accounting standard that does not "
                "apply, or a request whose obvious answer is wrong for THIS fact pattern. The "
                "rubric MUST reward the AI for catching the problem and penalize going along with it."
            ),
        },
        # Frontier models score ~0.37-0.39 on the Hard subsets; the solver is a 9B.
        REF_STATS_USE_CASE_MIX={"finance": 0.55, "legal": 0.45},
        PANEL="a panel of senior finance and legal professionals (CFAs, JDs, practitioners with "
              "many years in the field)",
        AI_DESC="a professional-services AI used by finance and legal experts",
        USER_ROLE="professional",
        USER_ROLE_ADJ="finance or legal",
        REQUEST_NOUN="professional request",
        EXPERT_TITLE="a senior practitioner",
        BENCH_NAME_HYPHEN="PRBench",
        PROPOSER_ROLE="a finance-and-legal practice expert",
        PRACTITIONER="professional",
        AI_SHORT="a professional-services AI",
        TARGET_DOMAINS_SENTENCE=(
            "The two target domains are finance and law — real working questions with facts, "
            "figures, a jurisdiction and a decision to make, NOT textbook exercises."
        ),
        VARY_AXES="sub-topic, client or entity context, jurisdiction, document type, and difficulty",
        GROUNDING_LITERATURE=(
            "grounding primary sources (statutes, regulations, rulings, standards, filings)"
        ),
        RARE_THING="exotic fact patterns",
        EXAMPLE_QUALIFIER="exception or carve-out",
        EXAMPLE_CHANGED_SUBJECT="a rule, rate or threshold",
        EXAMPLE_SCOPE="jurisdiction or entity type",
        DATASET_BRIEF=(
            "TARGET BENCHMARK (general description only; you have NO access to its items): "
            "PRBench is a rubric-based benchmark of expert-authored professional tasks in FINANCE "
            "and LAW, written by 182 practitioners (JDs, CFAs, 6+ years' experience) from their own "
            "real workflows and spanning 114 countries and 47 US jurisdictions. Prompts read like "
            "a colleague typing to an assistant: concrete facts, figures, entity names, deadlines "
            "and constraints, sometimes informal with typos, sometimes a multi-turn thread where the "
            "final message is the one to answer. Each task has a bespoke rubric of roughly 15-20 "
            "criteria: weighted positives ('critically important' ~9-10, 'important' ~5-8, "
            "'slightly important' ~1-4) and weighted negatives ('detrimental' classes) for specific "
            "errors. Criteria are grouped under categories such as Financial/Legal Accuracy, "
            "Application of Law to the Facts, Practical Utility, Risk & Regulatory Disclosure, "
            "Procedural Correctness, Process Transparency & Auditability, Handling Uncertainty, "
            "Instruction Following and Supplemental Insight. The score is the fraction of positive "
            "weight earned minus weight of triggered negatives, so a response wins by stating the "
            "specific governing rule, number, threshold, filing, election or deadline that the facts "
            "turn on -- and loses by confidently applying the wrong one. Frontier models score "
            "under 0.40 on the Hard subsets."
        ),
        EXAMPLE_VAGUE="\"discusses the tax treatment\"",
        EXAMPLE_SPECIFIC=(
            "\"Mentions in some way that the Section 1202 QSBS exclusion requires the stock to be "
            "held for more than five years\""
        ),
        EXAMPLE_WASTED="\"Mentions in some way that jurisdiction matters\"",
        EXAMPLE_REAL=(
            "\"Mentions in some way that New York does not recognise the federal computer-employee "
            "overtime exemption\""
        ),
        EXAMPLE_SPECIFIC_FACT="an exact threshold, rate, deadline, statutory section, filing "
                              "requirement or accounting treatment",
        EXAMPLE_NEG_ERRORS="wrong governing law or jurisdiction, misstated threshold or rate, "
                           "missed filing deadline or election, harmful or non-compliant advice, "
                           "fabricated case, statute or regulation",
        EXAMPLE_NUMBER_KINDS="threshold, rate, deadline, limit, ratio, or statutory cutoff",
        EXAMPLE_VAGUE2="\"discusses the filing requirements\"",
        EXAMPLE_SPECIFIC2=(
            "\"states the 75-day deadline for a late S-corporation election under Rev. Proc. "
            "2013-30 relief\""
        ),
        SOLVER_SYSTEM=(
            "You are a knowledgeable, careful professional-services AI assistant helping a finance "
            "or legal professional. Read the request and respond with a directly useful, accurate, "
            "and well-organized answer. Be complete but concise; follow the professional's "
            "instructions and requested format exactly. Ground claims in the actual governing rules "
            "(statutes, regulations, case law, accounting standards, tax code, market practice) for "
            "the jurisdiction and facts given, state important caveats and uncertainty, ask for "
            "missing context when it materially changes the answer, and never include fabricated "
            "authorities or non-compliant recommendations. Prioritize the professional's real "
            "exposure and obligations."
        ),
        REBRAND=[
            # Longest / most specific phrases first: a later generic rule ("clinician" ->
            # "professional") would otherwise rewrite the inside of these before they match.
            ("care consult, writing & documentation, medical research — NOT diagnosis",
             "finance, law — real professional working questions"),
            ("domains: care consult, writing & documentation, medical research",
             "domains: finance, law"),
            # This example wraps across a line in the source ("...for\n          metformin..."),
            # so it is matched in two halves.
            ("'eGFR threshold for", "'holding-period requirement for"),
            ("metformin discontinuation'", "the Section 1202 QSBS exclusion'"),
            ("'medical accuracy'", "'legal accuracy'"),
            ("the right disease", "the right topic"),
            ("a number, threshold, dose, or named", "a number, threshold, rate, or named"),
            ("a value, threshold, dose, interval, named entity",
             "a value, threshold, rate, deadline, named entity"),
            ("Recommends continuing metformin at an eGFR of 25",
             "States that the QSBS exclusion applies to stock held for three years"),
            ("States the target INR is 3.5-4.5",
             "States the federal gift-tax annual exclusion is $10,000"),
            ("Answers only in generic terms without stating any dosing threshold",
             "Answers only in generic terms without stating any governing threshold or deadline"),
            ("States the 2024 KDIGO eGFR threshold below which metformin must stop, with the "
             "numeric value",
             "States the current Section 1202 holding-period requirement for QSBS, with the "
             "numeric value"),
            ("a wrong dose, a missed red flag, unsafe reassurance, a fabricated trial or guideline",
             "a wrong governing rule, a misstated threshold, a missed deadline, a fabricated "
             "authority"),
            ("without knowing any medicine", "without knowing any finance or law"),
            ("preserving appropriate diagnostic uncertainty", "preserving appropriate uncertainty"),
            ("including safety / red-flag guidance and contraindications",
             "including the exposure, risks and exceptions that apply"),
            ("e.g. 'per the 2022 American College of Gastroenterology guideline [p2]'",
             "e.g. 'per FASB ASU 2023-08 [p2]'"),
            ("doses, units, frequencies, thresholds, cutoffs, ages, durations, percentages, codes, "
             "trial names",
             "rates, thresholds, limits, dates, deadlines, percentages, section numbers, case names"),
            ("doses, thresholds, cutoffs, intervals, percentages, sample sizes, effect sizes and "
             "confidence intervals",
             "rates, thresholds, limits, deadlines, percentages, section numbers, holdings and "
             "effective dates"),
            ("(doses, thresholds, codes, criteria, management steps)",
             "(rates, thresholds, deadlines, section numbers, procedural steps)"),
            ("different condition, different drug, different specialty focus",
             "different matter, different instrument or rule, different specialty focus"),
            ("patient or clinician", "client or professional"),
            ("patient safety", "the client's real exposure"),
            ("clinician-facing medical AI", "professional-services AI for finance and legal experts"),
            ("medical AI assistant", "professional-services AI assistant"),
            ("medical AI", "professional-services AI"),
            ("HealthBench Professional", "PRBench"),
            ("HealthBench-Professional", "PRBench"),
            ("HealthBench", "PRBench"),
            ("senior attending physician", "senior finance-and-law practitioner"),
            ("panel of physicians", "panel of senior finance and legal professionals"),
            ("an expert physician", "an expert finance and legal professional"),
            ("expert physician", "expert finance and legal professional"),
            ("senior physician", "senior practitioner"),
            ("physicians", "professionals"),
            ("physician", "professional"),
            ("clinicians", "professionals"),
            ("clinician's", "professional's"),
            ("clinician", "professional"),
            ("clinical evidence summarizer", "professional evidence summarizer"),
            ("clinical judgement", "professional judgement"),
            ("clinical judgment", "professional judgment"),
            ("REAL CLINICAL SUBSTANCE", "REAL PROFESSIONAL SUBSTANCE"),
            ("clinical SUBSTANCE", "professional SUBSTANCE"),
            ("clinical content", "professional content"),
            ("clinical terms", "professional terms"),
            ("clinical facts", "professional facts"),
            ("clinical fact", "professional fact"),
            ("clinical task", "professional task"),
            ("clinical entities", "legal or financial entities"),
            ("clinical", "professional"),
            ("red flags, escalation, contraindications, dosing",
             "governing law, deadlines, thresholds, filings, exposure"),
            ("exact dose, threshold, contraindication, code, or current guideline",
             "exact threshold, rate, deadline, statutory section, or current rule"),
            ("doses, thresholds, cutoffs, intervals", "rates, thresholds, deadlines, limits"),
            ("wrong dose, missed red flag, unsafe reassurance, fabricated trial/guideline",
             "wrong governing rule, misstated threshold, missed deadline, fabricated authority"),
            ("guidelines revised or drugs approved in the last few years, recalls, epidemiology",
             "rules, rates, thresholds or rulings changed in the last few years, new regulations"),
            ("guidelines, journals and regulators", "statutes, regulators, courts and standard-setters"),
            ("PubMed records, society guidelines and regulatory labels",
             "primary sources: statutes, regulations, court opinions, regulator and standard-setter "
             "publications"),
            ("grounding medical literature", "grounding primary sources"),
            ("medical literature", "primary legal and financial sources"),
            ("medical knowledge base", "knowledge base"),
            ("medicine", "finance or law"),
            ("medical", "professional"),
            ("unsafe", "non-compliant or harmful"),
            ("patients", "clients"),
            ("patient", "client"),
        ],
    )


def _profbench() -> dict:
    spec = {
        "chemistry_phd": [
            "electrochemistry_and_catalysis", "organic_synthesis_and_mechanism",
            "spectroscopy_and_structure_elucidation", "analytical_chemistry_and_titration",
            "physical_chemistry_kinetics_thermodynamics", "materials_and_polymer_chemistry",
            "computational_and_quantum_chemistry", "inorganic_and_coordination_chemistry",
        ],
        "physics_phd": [
            "condensed_matter_and_solid_state", "optics_photonics_and_lasers",
            "quantum_mechanics_and_quantum_information", "statistical_mechanics_and_thermodynamics",
            "electromagnetism_and_plasma", "particle_nuclear_and_astrophysics",
            "experimental_design_and_error_analysis", "fluid_mechanics_and_acoustics",
        ],
        "finance_mba": [
            "dcf_valuation_and_cost_of_capital", "capital_budgeting_and_project_finance",
            "mergers_acquisitions_and_lbo_modelling", "financial_statement_analysis_and_ratios",
            "portfolio_theory_and_risk", "fixed_income_and_derivatives_pricing",
            "corporate_capital_structure_and_payout", "working_capital_and_treasury",
        ],
        "consulting_mba": [
            "market_entry_and_sizing", "pricing_and_revenue_strategy",
            "operations_and_supply_chain", "cost_reduction_and_profitability",
            "growth_strategy_and_competitive_analysis", "organisation_and_change_management",
            "go_to_market_and_customer_segmentation", "business_case_and_scenario_analysis",
        ],
    }
    return dict(
        NAME="profbench",
        BENCH_NAME="ProfBench",
        USE_CASES={"chemistry_phd": 0.25, "physics_phd": 0.25, "finance_mba": 0.25,
                   "consulting_mba": 0.25},
        USE_CASE_DESC={
            "chemistry_phd": "a PhD-level chemistry problem: a described experiment, synthesis, "
                             "spectrum or dataset with quantities given, requiring a multi-step "
                             "quantitative or mechanistic write-up",
            "physics_phd": "a PhD-level physics problem: a described system, measurement or "
                           "derivation with parameters given, requiring a multi-step quantitative "
                           "write-up with units and stated assumptions",
            "finance_mba": "an MBA-level finance case: a company or deal described with figures, "
                           "requiring valuation, modelling or structuring analysis and a "
                           "recommendation",
            "consulting_mba": "an MBA-level consulting case: a client situation with market and "
                              "operating data, requiring structured analysis, sizing and a "
                              "recommendation",
        },
        SPECIALTIES_BY_USE_CASE=spec,
        SPECIALTIES=sorted({s for v in spec.values() for s in v}),
        REDTEAM_SHARE=0.10,
        MODE_INSTR={
            "good_faith": "MODE: good faith. An expert using the AI normally to produce a rigorous "
                          "report on the task.",
            "red_teaming": (
                "MODE: stress test. Build a task whose statement contains a subtly inconsistent "
                "quantity, an inapplicable approximation or model, a unit or sign trap, or a premise "
                "that a careful expert would flag. The rubric MUST reward the AI for catching the "
                "problem and penalize silently computing through it."
            ),
        },
        REF_STATS_USE_CASE_MIX={"chemistry_phd": 0.25, "physics_phd": 0.25, "finance_mba": 0.25,
                                "consulting_mba": 0.25},
        PANEL="a panel of domain experts (PhD chemists and physicists, MBA finance and consulting "
              "professionals)",
        AI_DESC="an expert-facing report-writing AI for PhD STEM and MBA professional tasks",
        USER_ROLE="expert",
        USER_ROLE_ADJ="PhD-level or MBA-level",
        REQUEST_NOUN="task statement",
        EXPERT_TITLE="a senior domain expert",
        BENCH_NAME_HYPHEN="ProfBench",
        PROPOSER_ROLE="a senior domain expert",
        PRACTITIONER="expert",
        AI_SHORT="a report-writing AI",
        TARGET_DOMAINS_SENTENCE=(
            "The four target domains are chemistry (PhD), physics (PhD), finance (MBA) and "
            "consulting (MBA) — self-contained report tasks that supply their own data, NOT "
            "short-answer quiz questions."
        ),
        VARY_AXES="sub-topic, system or company context, data given, deliverable format, and difficulty",
        GROUNDING_LITERATURE=(
            "grounding technical literature (journal articles, textbooks, standards, filings)"
        ),
        RARE_THING="exotic systems",
        EXAMPLE_QUALIFIER="boundary condition or validity limit",
        EXAMPLE_CHANGED_SUBJECT="a standard value or convention",
        EXAMPLE_SCOPE="regime or conditions",
        DATASET_BRIEF=(
            "TARGET BENCHMARK (general description only; you have NO access to its items): "
            "ProfBench is a small, rubric-based benchmark of professional REPORT tasks in four "
            "domains -- Chemistry PhD, Physics PhD, Finance MBA and Consulting MBA -- each task "
            "authored end-to-end by a credentialed expert who wrote the prompt and its rubric. A "
            "prompt is a self-contained task statement of roughly 700-4,700 characters that supplies "
            "the data needed (experimental procedure and measurements, system parameters, company "
            "or market figures, case context) and asks for a multi-part quantitative or analytical "
            "write-up, often numbered parts. Each task has a bespoke rubric of roughly 15-60 "
            "criteria, weighted Critical (4), Major (3), Minor (2), Additional (1), typed as "
            "Extraction (recall of given facts), Reasoning (correct intermediate and final "
            "results, mechanisms, assumptions) and Style. The score is the weight-fraction of "
            "criteria fulfilled, so a response wins by producing the specific intermediate values, "
            "named mechanisms, final numbers with units, and explicit assumptions an expert would "
            "check -- and loses points for every step skipped or computed wrongly. The best "
            "frontier model reaches ~66%."
        ),
        EXAMPLE_VAGUE="\"discusses the reaction mechanism\"",
        EXAMPLE_SPECIFIC=(
            "\"Mentions in some way that the Faradaic efficiency uses 4 electrons per mole of "
            "hydrazine produced\""
        ),
        EXAMPLE_WASTED="\"Mentions in some way that assumptions matter\"",
        EXAMPLE_REAL=(
            "\"Mentions in some way that the terminal value must be discounted back N years at the "
            "WACC, not at the growth rate\""
        ),
        EXAMPLE_SPECIFIC_FACT="an exact intermediate value, constant, unit conversion, formula, "
                              "mechanism step or modelling assumption",
        EXAMPLE_NEG_ERRORS="wrong unit or sign, misapplied formula or approximation, skipped "
                           "required step, unstated or impossible assumption, fabricated data",
        EXAMPLE_NUMBER_KINDS="intermediate value, constant, ratio, or final result with units",
        EXAMPLE_VAGUE2="\"computes the valuation\"",
        EXAMPLE_SPECIFIC2=(
            "\"states the unlevered free cash flow in year 3 as EBIT(1-t) + D&A - capex - change "
            "in NWC, with the computed figure\""
        ),
        SOLVER_SYSTEM=(
            "You are a rigorous expert assistant producing a professional report for a PhD-level "
            "or MBA-level task. Read the task statement and respond with a complete, well-organized "
            "write-up that answers every part asked, in order. Show the intermediate steps, formulas, "
            "values and units that an expert reviewer would check; state every assumption "
            "explicitly; give final results clearly. Be complete but not padded; follow the "
            "requested format exactly, and never fabricate data, sources or results."
        ),
        REBRAND=[
            # Longest / most specific phrases first (see the PRBench list).
            ("care consult, writing & documentation, medical research — NOT diagnosis",
             "chemistry PhD, physics PhD, finance MBA, consulting MBA — expert report tasks"),
            ("domains: care consult, writing & documentation, medical research",
             "domains: chemistry PhD, physics PhD, finance MBA, consulting MBA"),
            # Wraps across a line in the source; matched in two halves (see PRBench).
            ("'eGFR threshold for", "'electrons per mole for"),
            ("metformin discontinuation'", "the Faradaic-efficiency calculation'"),
            ("'medical accuracy'", "'technical accuracy'"),
            ("the right disease", "the right topic"),
            ("a number, threshold, dose, or named", "a number, constant, formula, or named"),
            ("a value, threshold, dose, interval, named entity",
             "a value, constant, unit, formula, named entity"),
            ("Recommends continuing metformin at an eGFR of 25",
             "Discounts the terminal value at the growth rate instead of the WACC"),
            ("States the target INR is 3.5-4.5",
             "States that Faradaic efficiency uses 2 electrons per mole of hydrazine"),
            ("Answers only in generic terms without stating any dosing threshold",
             "Answers only in generic terms without stating any intermediate value"),
            ("States the 2024 KDIGO eGFR threshold below which metformin must stop, with the "
             "numeric value",
             "States the year-3 unlevered free cash flow as EBIT(1-t) + D&A - capex - change in "
             "NWC, with the numeric value"),
            ("a wrong dose, a missed red flag, unsafe reassurance, a fabricated trial or guideline",
             "a wrong unit or sign, a skipped required step, a misapplied formula, fabricated data"),
            ("without knowing any medicine", "without knowing the field"),
            ("preserving appropriate diagnostic uncertainty", "preserving appropriate uncertainty"),
            ("including safety / red-flag guidance and contraindications",
             "including the assumptions, validity limits and units that apply"),
            ("e.g. 'per the 2022 American College of Gastroenterology guideline [p2]'",
             "e.g. 'per the 2021 IUPAC recommendation [p2]'"),
            ("doses, units, frequencies, thresholds, cutoffs, ages, durations, percentages, codes, "
             "trial names",
             "values, units, constants, thresholds, rates, durations, percentages, formulas, "
             "source names"),
            ("doses, thresholds, cutoffs, intervals, percentages, sample sizes, effect sizes and "
             "confidence intervals",
             "values, units, constants, thresholds, percentages, sample sizes, effect sizes and "
             "confidence intervals"),
            ("(doses, thresholds, codes, criteria, management steps)",
             "(values, constants, formulas, conditions, procedure steps)"),
            ("different condition, different drug, different specialty focus",
             "different system, different quantity, different specialty focus"),
            ("patient or clinician", "reader"),
            ("patient safety", "correctness of the final results"),
            ("clinician-facing medical AI", "expert-facing report-writing AI"),
            ("medical AI assistant", "expert report-writing AI assistant"),
            ("medical AI", "report-writing AI"),
            ("HealthBench Professional", "ProfBench"),
            ("HealthBench-Professional", "ProfBench"),
            ("HealthBench", "ProfBench"),
            ("senior attending physician", "senior domain expert (PhD scientist or MBA professional)"),
            ("panel of physicians", "panel of domain experts"),
            ("an expert physician", "a domain expert"),
            ("expert physician", "domain expert"),
            ("a senior physician", "a senior domain expert"),
            ("senior physician", "senior domain expert"),
            ("a clinician", "an expert"),
            ("physicians", "experts"),
            ("physician", "expert"),
            ("clinicians", "experts"),
            ("clinician's", "expert's"),
            ("clinician", "expert"),
            ("clinical evidence summarizer", "technical evidence summarizer"),
            ("clinical judgement", "expert judgement"),
            ("clinical judgment", "expert judgment"),
            ("REAL CLINICAL SUBSTANCE", "REAL TECHNICAL SUBSTANCE"),
            ("clinical SUBSTANCE", "technical SUBSTANCE"),
            ("clinical content", "technical content"),
            ("clinical terms", "technical terms"),
            ("clinical facts", "technical facts"),
            ("clinical fact", "technical fact"),
            ("clinical task", "technical task"),
            ("clinical entities", "technical entities"),
            ("clinical", "technical"),
            ("red flags, escalation, contraindications, dosing",
             "required steps, intermediate values, units, assumptions"),
            ("exact dose, threshold, contraindication, code, or current guideline",
             "exact constant, formula, standard value, or current convention"),
            ("doses, thresholds, cutoffs, intervals", "constants, values, units, formulas"),
            ("wrong dose, missed red flag, unsafe reassurance, fabricated trial/guideline",
             "wrong unit or sign, skipped step, misapplied formula, fabricated data"),
            ("guidelines revised or drugs approved in the last few years, recalls, epidemiology",
             "constants, standards or market data that changed in the last few years"),
            ("guidelines, journals and regulators", "journals, textbooks, standards bodies and filings"),
            ("PubMed records, society guidelines and regulatory labels",
             "primary sources: journal articles, textbooks, standards and official filings"),
            ("grounding medical literature", "grounding technical literature"),
            ("medical literature", "technical literature"),
            ("medical knowledge base", "knowledge base"),
            ("medicine", "the field"),
            ("medical", "technical"),
            ("unsafe", "incorrect"),
            ("patients", "readers"),
            ("patient", "reader"),
        ],
    )


def _medical() -> dict:
    # Identity bundle: fillers reproduce the pre-existing HealthBench prompts exactly.
    return dict(
        NAME="medical",
        BENCH_NAME="HealthBench Professional",
        USE_CASES=None, USE_CASE_DESC=None, SPECIALTIES=None, SPECIALTIES_BY_USE_CASE=None,
        REDTEAM_SHARE=None, MODE_INSTR=None, REF_STATS_USE_CASE_MIX=None,
        PANEL="a panel of physicians",
        AI_DESC="a clinician-facing medical AI",
        USER_ROLE="clinician",
        USER_ROLE_ADJ="clinical",
        REQUEST_NOUN="clinician request",
        EXPERT_TITLE="a senior physician",
        BENCH_NAME_HYPHEN="HealthBench-Professional",
        PROPOSER_ROLE="a clinician-informatics expert",
        PRACTITIONER="physician",
        AI_SHORT="a medical AI",
        TARGET_DOMAINS_SENTENCE=(
            "The three target domains are care consult, writing & documentation, and medical "
            "research — NOT simple diagnosis."
        ),
        VARY_AXES="sub-topic, patient context, document type, and difficulty",
        GROUNDING_LITERATURE="grounding medical literature",
        RARE_THING="rare diseases",
        EXAMPLE_QUALIFIER="contraindication",
        EXAMPLE_CHANGED_SUBJECT="a guideline",
        EXAMPLE_SCOPE="population",
        DATASET_BRIEF="",
        # The runs of spaces below are NOT typos. In the original generator template these
        # phrases straddle a backslash-continued line inside a triple-quoted string, which
        # keeps the next line's indentation. The identity bundle must reproduce those bytes.
        EXAMPLE_VAGUE="\"discusses renal dosing\"",
        EXAMPLE_SPECIFIC=(
            "\"states the eGFR threshold below which metformin is contraindicated "
            "(30 mL/min/1.73m2)\""
        ),
        EXAMPLE_WASTED="\"Mentions in some way that renal function       matters\"",
        EXAMPLE_REAL=(
            "\"Mentions in some way that metformin is contraindicated       below an eGFR of "
            "30 mL/min/1.73m2\""
        ),
        EXAMPLE_SPECIFIC_FACT="a value, threshold, dose,       interval, contraindication or named guideline",
        EXAMPLE_NEG_ERRORS="wrong dose, missed red flag, unsafe reassurance, fabricated       trial/guideline",
        EXAMPLE_NUMBER_KINDS="threshold, dose, interval,       cutoff, or staging boundary",
        EXAMPLE_VAGUE2="\"discusses renal dosing\"",
        EXAMPLE_SPECIFIC2=(
            "\"states the eGFR threshold below which metformin is contraindicated "
            "(30 mL/min/1.73m2)\""
        ),
        SOLVER_SYSTEM=None,
        REBRAND=[],
    )


_BUNDLES = {"medical": _medical, "prbench": _prbench, "profbench": _profbench}
if DOMAIN not in _BUNDLES:
    raise SystemExit(f"SE_DOMAIN={DOMAIN!r} unknown; choose one of {sorted(_BUNDLES)}")
BUNDLE = _BUNDLES[DOMAIN]()
IS_MEDICAL = DOMAIN == "medical"
# The brief as a LEADING PARAGRAPH: empty for medical (so the template is unchanged),
# "<brief>\n\n" otherwise. Templates use [[DOMAIN_DATASET_BRIEF_PARA]] at their start.
BUNDLE["DATASET_BRIEF_PARA"] = (BUNDLE["DATASET_BRIEF"] + "\n\n") if BUNDLE["DATASET_BRIEF"] else ""


def brief_paragraph() -> str:
    """``DATASET_BRIEF_PARA`` for prompts built in code rather than from a template."""
    return BUNDLE["DATASET_BRIEF_PARA"]


def rebrand(text: str) -> str:
    """Apply the bundle's ordered phrase substitutions. Identity for the medical domain."""
    if IS_MEDICAL or not text:
        return text
    for old, new in BUNDLE["REBRAND"]:
        text = text.replace(old, new)
        # Sentence-initial capitals.
        if old[:1].islower():
            text = text.replace(old[:1].upper() + old[1:], new[:1].upper() + new[1:])
    return text


def fill_domain(text: str) -> str:
    """Substitute [[DOMAIN_<KEY>]] tokens from the bundle (str.replace, like _fill)."""
    out = text
    for k, v in BUNDLE.items():
        if isinstance(v, str):
            out = out.replace(f"[[DOMAIN_{k}]]", v)
    return out
