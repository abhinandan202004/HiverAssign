from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class IntentEnum(str, Enum):
    DEVICE_HARDWARE_ISSUE = "DEVICE_HARDWARE_ISSUE"
    SOFTWARE_OS_BUG = "SOFTWARE_OS_BUG"
    ACCOUNT_SECURITY_BILLING = "ACCOUNT_SECURITY_BILLING"
    HOW_TO_CONFIGURATION = "HOW_TO_CONFIGURATION"
    ORDER_DELIVERY_STATUS = "ORDER_DELIVERY_STATUS"
    CHITCHAT_OUT_OF_SCOPE = "CHITCHAT_OUT_OF_SCOPE"


class ArbiterAction(str, Enum):
    AUTO_HANDLE = "AUTO_HANDLE"
    ESCALATE = "ESCALATE"


class ClassificationResult(BaseModel):
    intent: IntentEnum = Field(description="The discrete predicted intent category")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Calibrated confidence score between 0.0 and 1.0"
    )
    reasoning: str = Field(description="Chain-of-thought justification for the predicted intent")


class RetrievedDocument(BaseModel):
    tweet_id: str = Field(description="Historical tweet resolution ID")
    query_text: str = Field(description="Customer inbound text from historical dialogue")
    resolution_text: str = Field(description="Official @AppleSupport agent resolution text")
    intent: IntentEnum = Field(description="Associated intent category")
    bm25_score: float = Field(default=0.0, description="BM25 lexical relevance score")
    dense_score: float = Field(default=0.0, description="Dense semantic cosine similarity score")
    rrf_score: float = Field(default=0.0, description="Reciprocal Rank Fusion hybrid score")


class DraftReply(BaseModel):
    draft_text: str = Field(description="Contextual, brand-aligned drafted response")
    grounded_sources: List[str] = Field(
        default_factory=list, description="IDs of historical resolutions used for grounding"
    )
    agent_signoff: str = Field(default="^AB", description="Official brand agent signature")


class ArbiterDecision(BaseModel):
    action: ArbiterAction = Field(description="Final routing decision: AUTO_HANDLE or ESCALATE")
    layer: str = Field(
        default="NONE",
        description="Filter layer that triggered the decision ('HARD_RULE', 'SOFT_RULE', or 'PASSED')",
    )
    reasons: List[str] = Field(
        default_factory=list, description="Stated reasons for routing or escalation"
    )
    is_critical: bool = Field(
        default=False,
        description="Flag indicating whether query involves critical risk (PII, billing fraud, legal threats)",
    )
    confidence_passed: bool = Field(
        default=True, description="Whether intent confidence meets the 0.85 threshold"
    )
    grounding_passed: bool = Field(
        default=True, description="Whether factual claims in draft are verified against RAG context"
    )
    sanitized_draft: Optional[str] = Field(
        default=None, description="Public reply text if auto-handled, or sanitized message"
    )


class TriageResult(BaseModel):
    tweet_id: str
    inbound_text: str
    classification: ClassificationResult
    retrieved_resolutions: List[RetrievedDocument] = Field(default_factory=list)
    draft: Optional[DraftReply] = None
    arbiter_decision: ArbiterDecision
    final_action: ArbiterAction
    final_response: str
