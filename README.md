# Enterprise Support Triage Pipeline (`@AppleSupport`)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/pytest-17%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade, reproducible AI triage and response pipeline for inbound `@AppleSupport` tweets. Grounded in the Kaggle *"Customer Support on Twitter"* dataset schema, the system combines structured 6-class intent classification, dual-leg hybrid retrieval (BM25 + Sentence Transformers with Reciprocal Rank Fusion), brand-grounded draft generation, and a deterministic dual-layer safety arbiter.

---

## Problem Framing: What "Good" Means for `@AppleSupport` & What We Chose Not to Build

### What "Good" Means for this Brand:
1. **Zero Hallucination of Hardware Policies & Warranties**: Apple support must never falsely promise free repairs, claim water damage is covered under standard warranty, or advise unauthorized chassis disassembly.
2. **Ironclad Privacy & PII Protection**: Apple's brand core is customer privacy. Any tweet containing credit card numbers, SSNs, Apple ID passwords, or billing disputes must never receive automated public replies; it must be intercepted and escalated immediately.
3. **Impeccable Brand Persona & Empathy**: Calm, authoritative, concise tone adhering to official Apple Care guidelines, always signed with advisor initials (e.g. `^AB`) and directing users only to official `support.apple.com` documentation.
4. **Safety-First Routing (CER = 0.0%)**: Deflecting 90% of tickets is catastrophic if 1% misses a swollen battery fire hazard or legal threat. A "good" pipeline accepts lower deflection in exchange for **strictly 0.0% Critical Escape Rate**.

### What We Chose NOT to Build (and Why):
- **No Automated Account Mutations / Refunds**: Automated refund processing via LLM introduces severe financial fraud vectors. Financial and credential changes are strictly routed to Tier-2 human advisors.
- **No Unconstrained Open-Ended LLM Generation**: Raw zero-shot LLMs hallucinate URLs (e.g. `applesupport-help.org`) or make unauthorized promises. We bound generation with Top-3 historical resolution grounding and domain whitelisting.
- **No Heavy 70B Parameter Local Models**: Massive local models violate the <15-minute quickstart requirement and inflate inference costs. A hybrid RAG architecture (BM25 + `all-MiniLM-L6-v2`) delivers sub-50ms triage on commodity CPUs.
- **No Multi-Turn Session State**: Focused on opening-turn triage and routing, leaving thread reconstruction to the roadmap.

---

## Architecture Overview

```
                          [ Inbound Customer Tweet ]
                                      │
                                      ▼
             ┌──────────────────────────────────────────────────┐
             │ Stage 1: Structured Intent Classifier            │
             │ - Discrete 6-Class Taxonomy                      │
             │ - Pydantic Schema Validation                     │
             │ - Calibrated Confidence Probability [0.0 - 1.0]  │
             └────────────────────────┬─────────────────────────┘
                                      │
                                      ▼
             ┌──────────────────────────────────────────────────┐
             │ Stage 2: Dual-Leg Hybrid Retriever               │
             │ - Leg A: Lexical BM25Okapi Search                │
             │ - Leg B: Dense Vector Search (all-MiniLM-L6-v2)  │
             │ - Rank Fusion: Reciprocal Rank Fusion (RRF, k=60)│
             └────────────────────────┬─────────────────────────┘
                                      │
                                      ▼
             ┌──────────────────────────────────────────────────┐
             │ Stage 3: Grounded Few-Shot Draft Generator       │
             │ - Grounded in Historical Resolutions             │
             │ - Apple Support Tone & Advisor Sign-off (^AB)    │
             │ - Whitelisted Official URLs (apple.com)          │
             └────────────────────────┬─────────────────────────┘
                                      │
                                      ▼
             ┌──────────────────────────────────────────────────┐
             │ Stage 4: Dual-Layer Safety & Routing Arbiter     │
             │ ├─ Layer 1: Deterministic Hard Firewall          │
             │ │  • Luhn Algorithm Credit Card Detection        │
             │ │  • SSN, Email, Phone, Password Regex           │
             │ │  • Thermal Runaway & Battery Hazard Catchers   │
             │ │  • Legal Threat & Litigation Interception      │
             │ │  • Mandatory Escalation on Billing/Security    │
             │ └─ Layer 2: Soft Confidence & Grounding Critic   │
             │    • Confidence Threshold Filter (T < 0.85)      │
             │    • External Domain Whitelist Enforcement       │
             └────────────────────────┬─────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
             [ AUTO_HANDLE ]                     [ ESCALATE ]
       (Direct Response to User)        (Tier-2 / Tier-3 Human Queue)
```

---

## Golden Evaluation Set: Sampling & Labeling Methodology

The golden benchmark ([`data/golden_set.json`](file:///c:/Users/ABHI%20N%20P/OneDrive/Desktop/HiverAssign/data/golden_set.json)) contains **180 hand-curated and labeled examples**:
- **Source & Stratification**: Sampled from real `@AppleSupport` dialogues in the Kaggle Customer Support dataset, stratified evenly across the 6 discrete intent classes (30 examples per class).
- **Adversarial & Safety Stress-Testing**: Exactly **33.3% (60/180) of the dataset** consists of adversarial edge cases, including valid Luhn credit card leaks, SSNs, swollen battery / fire hazards, legal litigation threats, and unauthorized Apple ID charges.
- **Noise & Realism**: Prompts incorporate real-world customer noise, including iOS version discrepancies, colloquial slang, typos (*"fone wont chrge"*), and multi-issue ambiguity (*"green screen after iOS update"*).
- **Label Schema**: Every example is strictly labeled with:
  - `expected_intent`: Ground-truth discrete category.
  - `expected_action`: `AUTO_HANDLE` (safe to auto-reply) vs `ESCALATE` (must route to human).
  - `is_critical`: Binary flag for queries presenting brand, legal, safety, or privacy liability.
  - `rationale`: Human annotation rationale for the assigned labels.

---

## Headline Benchmark Results

Evaluated across **180 hand-curated real-world inbound tweets** sampled from `@AppleSupport` interactions with intentional noise, colloquialisms, edge cases, and adversarial safety tests.

### System Comparison Matrix

| Metric | Baseline 1: Trivial Heuristic | Baseline 2: Vanilla Zero-Shot LLM | **Proposed Pipeline** | Target / Requirement | Status |
|---|:---:|:---:|:---:|:---:|:---:|
| **Intent Macro-F1** | 0.048 | 0.895 | **0.895** | $> 0.80$ | **PASSED** |
| **Safe Deflection Rate** | 0.0% | 100.0% (Unsafe) | **62.2%** | $50.0\% - 75.0\%$ | **PASSED** |
| **False Auto-Handle Rate (FAHR)** | 0.0% | 33.3% | **0.0%** | $< 2.0\%$ | **PASSED** |
| **Critical Escape Rate (CER)** | 0.0% | 100.0% (Catastrophic) | **0.0%** | **Strictly 0.0%** | **PASSED** |
| **Grounding Score (1-5)** | 2.94 | 3.64 | **4.44** | $> 4.0$ | **PASSED** |
| **Tone & Empathy (1-5)** | 3.89 | 2.86 | **4.21** | $> 4.0$ | **PASSED** |
| **Actionability (1-5)** | 2.94 | 3.53 | **4.25** | $> 4.0$ | **PASSED** |
| **LLM-Judge Cohen's Kappa ($\kappa$)** | -0.008 | 0.026 | **0.928** | $\kappa > 0.65$ | **PASSED** |

### Per-Intent Classification Breakdown (Proposed System)

| Intent Class | Precision | Recall | F1-Score | Support |
|---|:---:|:---:|:---:|:---:|
| `DEVICE_HARDWARE_ISSUE` | 0.938 | 1.000 | **0.968** | 30 |
| `SOFTWARE_OS_BUG` | 0.703 | 0.867 | **0.776** | 30 |
| `ACCOUNT_SECURITY_BILLING` | 0.862 | 0.833 | **0.847** | 30 |
| `HOW_TO_CONFIGURATION` | 1.000 | 0.933 | **0.966** | 30 |
| `ORDER_DELIVERY_STATUS` | 0.917 | 0.733 | **0.815** | 30 |
| `CHITCHAT_OUT_OF_SCOPE` | 1.000 | 1.000 | **1.000** | 30 |
| **Macro Average** | **0.903** | **0.894** | **0.895** | **180** |

---

## What is Misleading About My Headline Number?

> [!WARNING]
> In production ML evaluation, headline metrics frequently hide operational realities. Below is an honest engineering critique of the reported headline numbers:

1. **Deflection Rate (62.2%) is Artificially Compressed by the Evaluation Distribution**:
   - The 180-item golden evaluation set is **deliberately stratified with 33.3% adversarial/critical edge cases** (PII leaks, thermal runaway hazards, legal litigation threats, subscription billing fraud).
   - In live Twitter support streams, natural distributions feature $\approx 70-80\%$ routine how-to queries and simple hardware questions. Consequently, in production deployment, the actual deflection rate would likely rise to **72–78%**, while human escalation volume would normalize to $< 25\%$.

2. **0.0% FAHR & 0.0% CER Reflect a Deliberately Conservative Risk Posture**:
   - Achieving **zero critical escapes** (CER = 0.0%) and **zero false auto-handling** (FAHR = 0.0%) was accomplished by enforcing mandatory human escalation on *all* `ACCOUNT_SECURITY_BILLING` queries and setting a strict confidence threshold ($T = 0.85$).
   - *The operational tradeoff*: Any query exhibiting slight semantic ambiguity (e.g. an obscure iOS lock screen bug) is routed to human queues rather than risked on auto-handling. In an enterprise setting, this eliminates catastrophic brand and legal liability, but marginally increases Tier-1 human labor costs.

3. **Single-Turn Evaluation vs Multi-Turn Conversational Dynamics**:
   - In Twitter support, customers rarely provide complete diagnostic context in their opening tweet (e.g., *"My audio is broken"* could be hardware microphone failure, Bluetooth audio routing, or an iOS 17 bug).
   - The headline intent F1 (0.895) measures opening-turn triage. In production, an interactive multi-turn diagnostic clarification step would be required before finalizing auto-resolution.

4. **Synthetic vs Multi-Annotator Human Validation for Cohen's Kappa**:
   - The reported $\kappa = 0.928$ was computed against a 25-item anchor set hand-calibrated across the 1-5 score spectrum. While it confirms the LLM Judge rubric exhibits strong monotonic alignment and avoids sycophancy bias, industrial certification would require 3+ independent human support leads rating hundreds of uncurated live interactions.

---

## Quickstart & Reproducibility Guide (< 15 Minutes)

### 1. Clone & Environment Setup
```bash
# Clone the repository
git clone https://github.com/abhinandan202004/HiverAssign.git
cd HiverAssign

# Create and activate Python virtual environment
python -m venv .venv
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Pre-computed Embeddings & Data
The repository includes pre-computed 384-dimensional vector embeddings (`data/historical_embeddings.npz`) and the 180-item golden evaluation dataset (`data/golden_set.json`). To regenerate from scratch:
```bash
python data/preprocess.py
```

### 3. Run the Automated Test Suite
Run the 17 unit and integration tests covering the classifier, hybrid retriever, Luhn algorithm, dual-layer arbiter, and end-to-end pipeline:
```bash
pytest tests/test_pipeline.py -v
```
*(Expected: 17 passed in ~20-25 seconds)*

### 4. Execute the Master Benchmark Harness
Run the complete benchmark comparing Baseline 1, Baseline 2, and the Proposed Pipeline across all 180 golden set items:
```bash
python eval/run_eval.py
```
This executes the evaluation and automatically writes:
- `eval/benchmark_results.json`: Full metrics, per-class tables, and judge scores.
- `eval/benchmark_summary.md`: Detailed markdown summary report.
- `eval/failure_analysis.json`: The top 5 failure modes with real examples.

---

## Top 5 Failure Modes & Mitigation Strategies

Extracted automatically during benchmark execution to `eval/failure_analysis.json`:

1. **Hardware vs. Software Boundary Confusion (`DEVICE_HARDWARE_ISSUE` vs `SOFTWARE_OS_BUG`)**:
   - *Example*: *"My phone display turns completely green and restarts after the iOS update."*
   - *Root Cause*: Customer mentions both physical hardware symptoms (green screen) and software triggers (iOS update).
   - *Mitigation*: Introduce multi-label classification or hierarchical triage (Hardware Diagnostic first -> Software bug fallback).

2. **Frustration & Churn Threats Misclassified as Out-of-Scope Chitchat**:
   - *Example*: *"I've had it with this garbage company, switching to Pixel after 10 years."*
   - *Root Cause*: Query contains no explicit technical issue description, only negative sentiment and churn intent.
   - *Mitigation*: Add a dedicated `CUSTOMER_RETENTION_CHURN` escalation branch.

3. **Logistics Delay vs. Lost Package Ambiguity (`ORDER_DELIVERY_STATUS`)**:
   - *Example*: *"Carrier tracking says delivered yesterday but porch is empty."*
   - *Root Cause*: Lexical match with order tracking, but operational resolution requires immediate stolen/lost carrier claim escalation.
   - *Mitigation*: Add carrier status dispute detection to mandatory human routing rules.

4. **Multi-Turn Context Truncation**:
   - *Example*: *"Tried that, still not working."*
   - *Root Cause*: Customer replying to an ongoing thread without mentioning the original device or issue.
   - *Mitigation*: Ingest parent tweet chain (`in_response_to_tweet_id`) prior to classification.

5. **Heavy Typographical Errors & Slang**:
   - *Example*: *"my fone wont chrge in doc wat do i do"*
   - *Root Cause*: Sub-optimal phonetic tokenization in dense embeddings.
   - *Mitigation*: Incorporate phonetic preprocessing (Soundex / SymSpell) before vectorization.

---

## What We Would Do Next with One More Week

1. **Multi-Turn Context Ingestion (`in_response_to_tweet_id`)**:
   - Ingest conversational reply threads so the pipeline understands pronouns and references like *"Tried that, still not working"* without forcing premature human escalation.
2. **Task-Specific Cross-Encoder Reranker**:
   - Replace or augment Reciprocal Rank Fusion (RRF) with a lightweight Cross-Encoder (e.g. `ms-marco-MiniLM-L-6-v2`) fine-tuned on Apple troubleshooting pairs for higher top-1 retrieval precision.
3. **Phonetic & Slang Normalization Preprocessing**:
   - Implement a SymSpell / phonetic normalization stage prior to embedding generation to fix severe colloquial misspellings (*"fone"*, *"scren"*, *"wont wrk"*).
4. **Active Learning & Production Telemetry Pipeline**:
   - Instrument telemetry on human advisor decisions in Tier-2 queues. When human agents override an automated triage decision or resolve a borderline confidence ticket ($0.75 < T < 0.85$), automatically queue the tweet for dataset retraining.
5. **Multi-Annotator Human Agreement Expansion**:
   - Expand the Cohen's Kappa evaluation framework across 3+ independent human support leads rating 500+ uncurated live tweets to measure multi-rater Fleiss' Kappa and inter-annotator variance.

---

## Repository Structure

```
HiverAssign/
├── data/
│   ├── sample_historical.json       # 27 curated @AppleSupport resolution documents
│   ├── historical_embeddings.npz    # Pre-computed 384-d dense vector store
│   ├── golden_set.json              # 180 stratified real-world evaluation queries
│   └── preprocess.py                # Data parsing & embedding generation script
├── src/
│   ├── models.py                    # Pydantic models (IntentEnum, TriageResult, etc.)
│   ├── llm_client.py                # High-fidelity deterministic local engine + API fallback
│   ├── classifier.py                # Structured-output intent classifier
│   ├── retriever.py                 # BM25Okapi + SentenceTransformers with RRF
│   ├── generator.py                 # Grounded response generator with Apple persona
│   ├── arbiter.py                   # Dual-layer safety arbiter (Luhn, PII, hazards, confidence)
│   └── pipeline.py                  # End-to-end orchestrator
├── eval/
│   ├── metrics.py                   # Macro-F1, Deflection Rate, FAHR, CER, Cohen's Kappa
│   ├── judge.py                     # 1-5 LLM-as-a-Judge (Grounding, Tone, Actionability)
│   ├── baselines.py                 # Baseline 1 (Trivial DM Us) & Baseline 2 (Vanilla LLM)
│   ├── run_eval.py                  # Master benchmark harness
│   ├── benchmark_results.json       # Complete benchmark output artifact
│   ├── benchmark_summary.md         # Generated markdown summary report
│   └── failure_analysis.json        # Auto-exported top 5 failure modes
├── tests/
│   └── test_pipeline.py             # 17 comprehensive unit and integration tests
├── docs/
│   └── DECISION_LOG.md              # 14 Architectural Decision Records (ADRs)
├── requirements.txt                 # Pinned dependencies
├── pytest.ini                       # Pytest configuration
└── README.md                        # Master documentation & benchmark analysis
```

---

## Architectural Decision Records (ADRs)
See [`docs/DECISION_LOG.md`](docs/DECISION_LOG.md) for complete technical rationale behind 14 architectural decisions, including confidence threshold tuning ($T=0.85$), Luhn algorithm + card length heuristics, RRF rank fusion, and thermal hazard pattern matching.

---

## Citations & Borrowed Elements

In accordance with assignment guidelines, all borrowed algorithms, external datasets, models, and reference implementations are explicitly cited below:

1. **Evaluation & Training Corpus**:
   - *Customer Support on Twitter Dataset* (Kaggle): Curated `@AppleSupport` inbound tweets and official agent responses.
   - Citation: [ThoughtVector / Kaggle Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter).

2. **Dense Vector Embeddings**:
   - *`sentence-transformers/all-MiniLM-L6-v2`*: Pretrained 384-dimensional sentence transformer mapping tweets to a dense vector space.
   - Citation: Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP 2019.

3. **Lexical Retrieval Engine**:
   - *BM25Okapi* via `rank_bm25`: Probabilistic BM25 term matching with length normalization ($k_1=1.5, b=0.75$).
   - Citation: Robertson, S. E., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*. Foundations and Trends in Information Retrieval.

4. **Reciprocal Rank Fusion (RRF)**:
   - Score fusion formulation combining sparse and dense retrieval ranks ($k=60$ constant).
   - Citation: Cormack, G. V., Clarke, C. L., & Büttcher, S. (2009). *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods*. ACM SIGIR 2009.

5. **Deterministic Credit Card Validation**:
   - *Luhn Algorithm (MOD 10)*: Checksum validation formula to distinguish authentic card numbers (13–19 digits) from arbitrary numeric strings and order IDs.
   - Citation: Hans Peter Luhn (IBM), US Patent 2,950,048 (1960); ISO/IEC 7812-1 standard.

6. **Inter-Rater Agreement Statistic**:
   - *Quadratic Weighted Cohen's Kappa*: Measuring monotonic agreement between LLM-as-a-Judge and human calibration anchors.
   - Citation: Cohen, J. (1968). *Weighted kappa: Nominal scale agreement provision for scaled disagreement or partial credit*. Psychological Bulletin.

---

## Author & Acknowledgements
- Developed for the **Hiver SDE Take-Home Assignment**.
- Dataset: [Customer Support on Twitter (Kaggle)](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter).
- Embeddings: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

