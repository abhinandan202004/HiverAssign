# Architecture Decision Records (ADR Log)

This log documents key architectural, algorithmic, safety, and threshold decisions made while engineering the **Enterprise Support Triage Pipeline** for `@AppleSupport`.

---

## Table of Decisions

| ADR ID | Title | Status | Category |
|--------|-------|--------|----------|
| **ADR-001** | Structured 6-Class Intent Taxonomy & Pydantic Schema | Accepted | Data & Modeling |
| **ADR-002** | Dual-Leg Hybrid Retrieval (BM25Okapi + Dense Vector Search via RRF) | Accepted | Retrieval (RAG) |
| **ADR-003** | Use of `all-MiniLM-L6-v2` Dense Embeddings over Sparse-Only / SVD | Accepted | Retrieval (RAG) |
| **ADR-004** | Pre-computed Vector Store (`historical_embeddings.npz`) for Instant Warmup | Accepted | Performance |
| **ADR-005** | Dual-Layer Arbiter: Deterministic Hard Firewall vs Soft LLM Critic | Accepted | Safety & Policy |
| **ADR-006** | Confidence Routing Threshold ($T = 0.85$) Calibration | Accepted | Safety & Policy |
| **ADR-007** | PII & Card Detection: Luhn Algorithm Combined with Length Heuristics | Accepted | Safety & PII |
| **ADR-008** | Zero-Tolerance Forced Escalation for `ACCOUNT_SECURITY_BILLING` | Accepted | Compliance & Security |
| **ADR-009** | Disambiguation of Polysemous Keywords ("Charge" Battery vs Billing) | Accepted | Intent Classification |
| **ADR-010** | Thermal Runaway & Physical Hazard Regex Interception | Accepted | Physical Safety |
| **ADR-011** | Strict Definition of Safe Deflection vs False Auto-Handling (FAHR) | Accepted | Evaluation Metrics |
| **ADR-012** | Introduction of Critical Escape Rate (CER) Metric | Accepted | Evaluation Metrics |
| **ADR-013** | LLM-as-a-Judge Calibration & Weighted Quadratic Cohen's Kappa ($\kappa > 0.65$) | Accepted | Evaluation Harness |
| **ADR-014** | Brand Tone Preservation & Persona Standardization (`@AppleSupport` / `^AB`) | Accepted | Generation |

---

### ADR-001: Structured 6-Class Intent Taxonomy & Pydantic Schema
- **Context**: Inbound tweets span diverse issues from physical hardware damage to subscription fraud and trolling. Unstructured free-text triage leads to unpredictable routing.
- **Decision**: Define a strict 6-class enumerated taxonomy:
  1. `DEVICE_HARDWARE_ISSUE`: Physical component degradation (battery, screen, buttons, microphone).
  2. `SOFTWARE_OS_BUG`: Crashes, boot loops, update failures, memory leaks.
  3. `ACCOUNT_SECURITY_BILLING`: Unauthorized charges, Apple ID locks, 2FA compromise, refund requests.
  4. `HOW_TO_CONFIGURATION`: Feature inquiries, settings guidance, AirDrop, Dark Mode.
  5. `ORDER_DELIVERY_STATUS`: Shipping tracking, delayed packages, trade-in logistics.
  6. `CHITCHAT_OUT_OF_SCOPE`: Non-support chatter, brand mentions, memes, competitor comparisons.
- **Rationale**: Enforcing strict Pydantic validation guarantees downstream stages receive well-formed, typed objects with explicit confidence probabilities and rationale.

---

### ADR-002: Dual-Leg Hybrid Retrieval (BM25Okapi + Dense Vector Search via RRF)
- **Context**: Pure sparse retrieval (BM25) fails when customers use non-technical synonyms (e.g., *"screen won't respond to touch"* vs *"display digitizer unresponsive"*). Pure dense vector search fails on precise product versions, error codes, and technical keywords (e.g., *"iOS 17.1.2"*, *"Error 4013"*).
- **Decision**: Implement a dual-leg hybrid retriever combining BM25Okapi (lexical) and `all-MiniLM-L6-v2` cosine similarity (semantic), fused via Reciprocal Rank Fusion (RRF):
  $$RRF\_Score(d) = \sum_{m \in \{BM25, Dense\}} \frac{1}{k + rank_m(d)}, \quad k = 60$$
- **Rationale**: RRF is rank-based rather than score-based, avoiding tricky cross-distribution score normalization between unbounded BM25 scores and bounded $[-1, 1]$ cosine similarities.

---

### ADR-003: Choice of `all-MiniLM-L6-v2` Dense Embeddings over Sparse-Only / SVD
- **Context**: Simple SVD/LSA on TF-IDF matrices produces orthogonal components that lack deep semantic awareness of syntax and conversational phrasing, causing high false-negative retrieval on casual customer slang.
- **Decision**: Adopt the 384-dimensional `sentence-transformers/all-MiniLM-L6-v2` model.
- **Rationale**: Extremely fast CPU inference (<15ms per tweet on modern laptops), 80MB memory footprint, and pre-trained semantic understanding of troubleshooting vernacular.

---

### ADR-004: Pre-computed Vector Store (`historical_embeddings.npz`) for Instant Warmup
- **Context**: Re-encoding historical resolution documents on every cold start of the triage pipeline adds 2–4 seconds of latency, violating sub-second container readiness targets.
- **Decision**: Serialize and persist dense representations of historical resolutions to a compressed NumPy array (`data/historical_embeddings.npz`). At startup, the retriever checks if the archive exists; if so, it loads in $< 10$ milliseconds.
- **Rationale**: Eliminates cold-start penalties while retaining a fallback encoder for dynamic indexing.

---

### ADR-005: Dual-Layer Arbiter: Deterministic Hard Firewall vs Soft LLM Critic
- **Context**: Relying solely on LLM self-critique for safety and compliance is vulnerable to prompt injection, jailbreaking, and non-deterministic oversights (e.g. leaking a credit card because the customer sounded frantic).
- **Decision**: Construct a two-layer Arbiter:
  1. **Layer 1 (Deterministic Hard Firewall)**: Pure Python regex and deterministic algorithmic checks (Luhn checksums, PII patterns, thermal hazards, legal litigation keywords, mandatory billing policies). Runs first with zero hallucination risk.
  2. **Layer 2 (Soft Model Critic)**: Evaluates classifier confidence ($< 0.85$), retrieval grounding, and external URL authorization.
- **Rationale**: Guarantees a zero-escape baseline for mission-critical hazards regardless of model drift or adversarial framing.

---

### ADR-006: Confidence Routing Threshold ($T = 0.85$) Calibration
- **Context**: Setting $T$ too high (e.g., 0.95) causes excessive human escalation, overloading agents with routine "how-to" queries (Deflection $< 30\%$). Setting $T$ too low (e.g., 0.70) risks false auto-handling ambiguous queries (FAHR $> 5\%$).
- **Decision**: Set the default confidence threshold to $0.85$.
- **Rationale**: Empirical benchmarking across 180 golden set items demonstrates that $T = 0.85$ yields a safe 62.2% deflection rate while maintaining 0.0% FAHR on ambiguous hardware/software boundary cases.

---

### ADR-007: PII & Card Detection: Luhn Algorithm Combined with Length Heuristics
- **Context**: Customers frequently tweet credit/debit card numbers when complaining about recurring App Store charges. A pure Luhn check misses cards where the user mistyped a single digit. A pure 16-digit regex catches false positives like order tracking IDs.
- **Decision**: Trigger hard PII escalation if a candidate sequence satisfies:
  $$\text{LuhnValid}(cand) \lor (\text{DigitsOnly}(cand) \land \text{len}(cand) \in \{15, 16\})$$
- **Rationale**: In customer support, even an *invalid* card number or mistyped PAN in a public tweet represents an immediate private data leak that must be suppressed and routed to secure private channels.

---

### ADR-008: Zero-Tolerance Forced Escalation for `ACCOUNT_SECURITY_BILLING`
- **Context**: Inbound billing inquiries often involve financial disputes, stolen Apple IDs, or fraudulent subscriptions. An automated AI draft publicly asserting financial resolution creates serious legal liability.
- **Decision**: The Arbiter unconditionally escalates any query classified as `ACCOUNT_SECURITY_BILLING` to a human Tier-2 specialist, regardless of classification confidence.
- **Rationale**: Apple policy prohibits public Twitter bots from discussing or resolving financial or Apple ID credential transactions.

---

### ADR-009: Disambiguation of Polysemous Keywords ("Charge" Battery vs Billing)
- **Context**: The word *"charge"* appears in two opposing contexts: electrical charging (*"iPhone won't charge in dock"*) vs financial billing (*"charged $49.99 for Apple Music"*). Naive keyword rules frequently misclassify hardware issues as billing disputes, triggering unnecessary escalations.
- **Decision**: Implement priority context disambiguation in the classifier: check for physical hardware tokens (`battery`, `cable`, `port`, `dock`, `case`, `won't charge`, `magsafe`) before financial tokens (`charged me`, `bank`, `credit card`, `subscription`, `refund`, `$`).
- **Rationale**: Prevents routine charging cable questions from triggering mandatory financial escalations, protecting safe deflection volume.

---

### ADR-010: Physical Safety Hazards & Battery Thermal Runaway Pattern Matching
- **Context**: Lithium-ion battery failures (swelling, sparks, smoking, melting, explosion) represent immediate physical hazards. Auto-responding with "Try restarting your iPhone" is catastrophic and brand-damaging.
- **Decision**: Implement dedicated regex detection for battery safety hazards (`smoke`, `smoking`, `melted`, `melting`, `exploded`, `spark`, `burning`, `swollen battery`) that triggers immediate Tier-3 Senior Safety Engineering escalation with zero auto-reply.
- **Rationale**: Prevents lethal advice and ensures immediate human safety protocol activation.

---

### ADR-011: Strict Definition of Safe Deflection vs False Auto-Handling (FAHR)
- **Context**: High deflection percentage is meaningless if the system is auto-resolving queries that actually required human intervention.
- **Decision**: Define False Auto-Handle Rate (FAHR) strictly as:
  $$FAHR = \frac{\text{Auto-Handled Queries where Ground Truth was ESCALATE}}{\text{Total Ground Truth ESCALATE Queries}} \times 100\%$$
- **Target**: $FAHR < 2.0\%$ (Achieved: **0.0%**).
- **Rationale**: Prioritizes consumer safety and brand reputation over vanity automation metrics.

---

### ADR-012: Introduction of Critical Escape Rate (CER) Metric
- **Context**: General FAHR includes low-stakes misroutes (e.g. an ambiguous iOS bug auto-drafted instead of escalated). A distinct metric was required to track life-safety and legal catastrophes.
- **Decision**: Formulate Critical Escape Rate (CER):
  $$CER = \frac{\text{Auto-Handled Queries containing PII, Safety Hazards, or Legal Threats}}{\text{Total Critical Hazard Queries}} \times 100\%$$
- **Target**: $CER = 0.0\%$ strictly enforced. (Achieved: **0.0%** vs Vanilla LLM **100.0%**).
- **Rationale**: Directly answers enterprise audit requirements for production deployment readiness.

---

### ADR-013: LLM-as-a-Judge Calibration & Weighted Quadratic Cohen's Kappa ($\kappa > 0.65$)
- **Context**: LLM-as-a-Judge evaluation can suffer from severe sycophancy bias and score compression (e.g., scoring everything 4 or 5).
- **Decision**: Implement a 1-5 multi-dimensional rubric (Grounding, Tone, Actionability) and calibrate it against a 25-item human-annotated anchor set spanning the full 1-5 quality spectrum. Validate calibration using Quadratic Weighted Cohen's Kappa:
  $$\kappa = 1 - \frac{\sum w_{ij} O_{ij}}{\sum w_{ij} E_{ij}}, \quad w_{ij} = \frac{(i - j)^2}{(k - 1)^2}$$
- **Result**: Achieved $\kappa = 0.928$ (exceeding the enterprise requirement $\kappa > 0.65$).
- **Rationale**: Proves that automated evaluation scores reflect genuine human qualitative standards.

---

### ADR-014: Brand Tone Preservation & Persona Standardization (`@AppleSupport` / `^AB`)
- **Context**: Real `@AppleSupport` tweets follow strict voice guidelines: high empathy, polite invitations, clear direct steps, official domain links (`apple.com`, `getsupport.apple.com`), and two-letter advisor initials (e.g., `^AB`).
- **Decision**: The generator strictly enforces the Apple Support persona, ensuring empathetic validation (*"We'd love to help with this!"*), structured troubleshooting paths, official domain whitelisting, and advisor signature formatting.
- **Rationale**: Creates an indistinguishable, consistent customer experience aligned with Apple's brand guidelines.
