from __future__ import annotations

import uuid
from typing import Optional
from src.models import (
    ArbiterAction,
    TriageResult,
    ClassificationResult,
    DraftReply,
    ArbiterDecision,
)
from src.llm_client import LLMClient
from src.classifier import IntentClassifier
from src.retriever import HybridRetriever
from src.generator import DraftGenerator
from src.arbiter import DualLayerArbiter


class SupportTriagePipeline:
    """
    End-to-end Enterprise Customer Support Triage Pipeline for @AppleSupport.
    Orchestrates:
    1. Intent Classification (6 discrete intents + confidence scoring)
    2. Hybrid Retrieval (BM25 + all-MiniLM-L6-v2 vector search with RRF)
    3. Grounded Draft Generation (few-shot RAG with Apple Support persona)
    4. Dual-Layer Safety Arbiter (Hard PII/threat/billing rules + Soft critic checks)
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        retriever: Optional[HybridRetriever] = None,
        signoff: str = "^AB",
    ):
        self.llm = llm_client or LLMClient()
        self.classifier = IntentClassifier(self.llm)
        self.retriever = retriever or HybridRetriever()
        self.generator = DraftGenerator(self.llm, default_signoff=signoff)
        self.arbiter = DualLayerArbiter()

    def triage(self, tweet_text: str, tweet_id: Optional[str] = None) -> TriageResult:
        t_id = tweet_id or f"twt_{uuid.uuid4().hex[:8]}"

        # Stage 1: Intent Classification
        classification = self.classifier.classify(tweet_text)

        # Stage 2: Hybrid RAG Retrieval (Top-3 historical resolutions)
        retrieved_docs = self.retriever.retrieve(tweet_text, top_k=3)

        # Stage 3: Grounded Draft Generation
        draft = self.generator.generate(tweet_text, classification, retrieved_docs)

        # Stage 4: Dual-Layer Safety Arbiter
        decision = self.arbiter.evaluate(
            inbound_text=tweet_text,
            classification=classification,
            draft=draft,
            retrieved_docs=retrieved_docs,
        )

        if decision.action == ArbiterAction.AUTO_HANDLE:
            final_action = ArbiterAction.AUTO_HANDLE
            final_response = draft.draft_text
        else:
            final_action = ArbiterAction.ESCALATE
            reasons_str = "; ".join(decision.reasons)
            final_response = f"[ROUTED TO HUMAN SPECIALIST] Escalation Reason: {reasons_str}"

        return TriageResult(
            tweet_id=t_id,
            inbound_text=tweet_text,
            classification=classification,
            retrieved_resolutions=retrieved_docs,
            draft=draft,
            arbiter_decision=decision,
            final_action=final_action,
            final_response=final_response,
        )
