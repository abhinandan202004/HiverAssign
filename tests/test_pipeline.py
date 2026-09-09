from __future__ import annotations

import os
import pytest
from src.models import (
    IntentEnum,
    ClassificationResult,
    ArbiterAction,
    DraftReply,
    RetrievedDocument,
)
from src.classifier import IntentClassifier
from src.retriever import HybridRetriever
from src.generator import DraftGenerator
from src.arbiter import DualLayerArbiter, luhn_checksum_valid
from src.pipeline import SupportTriagePipeline


@pytest.fixture(scope="module")
def pipeline():
    return SupportTriagePipeline()


@pytest.fixture
def arbiter():
    return DualLayerArbiter()


# ------------------------------------------------------------
# 1. Intent Classifier Tests
# ------------------------------------------------------------

def test_intent_classifier_hardware():
    classifier = IntentClassifier()
    result = classifier.classify("My iPhone 14 Pro screen is shattered and touch is unresponsive")
    assert isinstance(result, ClassificationResult)
    assert result.intent == IntentEnum.DEVICE_HARDWARE_ISSUE
    assert result.confidence >= 0.85


def test_intent_classifier_billing():
    classifier = IntentClassifier()
    result = classifier.classify("I was charged $89 for an unauthorized subscription on Apple ID")
    assert result.intent == IntentEnum.ACCOUNT_SECURITY_BILLING
    assert result.confidence >= 0.85


def test_intent_classifier_software_bug():
    classifier = IntentClassifier()
    result = classifier.classify("iPhone is stuck on Apple logo bootloop after updating to iOS 17")
    assert result.intent == IntentEnum.SOFTWARE_OS_BUG
    assert result.confidence >= 0.85


def test_intent_classifier_how_to():
    classifier = IntentClassifier()
    result = classifier.classify("How do I turn on AirDrop to share photos with non-contacts?")
    assert result.intent == IntentEnum.HOW_TO_CONFIGURATION
    assert result.confidence >= 0.85


def test_intent_classifier_order_status():
    classifier = IntentClassifier()
    result = classifier.classify("Where is my delivery tracking number for my Apple Store MacBook order?")
    assert result.intent == IntentEnum.ORDER_DELIVERY_STATUS
    assert result.confidence >= 0.85


# ------------------------------------------------------------
# 2. Hybrid Retriever Tests
# ------------------------------------------------------------

def test_hybrid_retriever_returns_top_3():
    retriever = HybridRetriever()
    results = retriever.retrieve("battery dying fast degraded capacity", top_k=3)
    assert len(results) == 3
    for doc in results:
        assert isinstance(doc, RetrievedDocument)
        assert doc.rrf_score > 0.0
        assert doc.resolution_text is not None


# ------------------------------------------------------------
# 3. Dual-Layer Arbiter Tests (Hard Rules)
# ------------------------------------------------------------

def test_luhn_algorithm():
    # Valid Luhn number (4532015012345671)
    assert luhn_checksum_valid("4532015012345671") is True
    # Invalid card number
    assert luhn_checksum_valid("4532015012345679") is False


def test_arbiter_catches_credit_card_pii(arbiter):
    tweet = "Can you help? My card 4532-0150-1234-5678 has a problem"
    classification = ClassificationResult(
        intent=IntentEnum.DEVICE_HARDWARE_ISSUE, confidence=0.95, reasoning="hardware"
    )
    decision = arbiter.evaluate(tweet, classification)
    assert decision.action == ArbiterAction.ESCALATE
    assert decision.layer == "HARD_RULE"
    assert decision.is_critical is True
    assert any("CREDIT_CARD" in r for r in decision.reasons)


def test_arbiter_catches_email_and_phone_pii(arbiter):
    tweet = "Call me at 415-555-0199 or email user@testdomain.com about my issue"
    classification = ClassificationResult(
        intent=IntentEnum.HOW_TO_CONFIGURATION, confidence=0.92, reasoning="how to"
    )
    decision = arbiter.evaluate(tweet, classification)
    assert decision.action == ArbiterAction.ESCALATE
    assert decision.is_critical is True
    assert any("PHONE_NUMBER" in r or "EMAIL" in r for r in decision.reasons)


def test_arbiter_catches_safety_hazards(arbiter):
    tweet = "My iPhone battery exploded and caught fire on my desk!"
    classification = ClassificationResult(
        intent=IntentEnum.DEVICE_HARDWARE_ISSUE, confidence=0.95, reasoning="hazard"
    )
    decision = arbiter.evaluate(tweet, classification)
    assert decision.action == ArbiterAction.ESCALATE
    assert decision.is_critical is True
    assert any("safety hazard" in r.lower() for r in decision.reasons)


def test_arbiter_catches_legal_threats(arbiter):
    tweet = "I am filing a lawsuit and my lawyer will be contacting Apple Support"
    classification = ClassificationResult(
        intent=IntentEnum.DEVICE_HARDWARE_ISSUE, confidence=0.90, reasoning="legal"
    )
    decision = arbiter.evaluate(tweet, classification)
    assert decision.action == ArbiterAction.ESCALATE
    assert decision.is_critical is True
    assert any("legal threat" in r.lower() for r in decision.reasons)


def test_arbiter_mandatory_billing_escalation(arbiter):
    tweet = "Need a refund for subscription charge"
    classification = ClassificationResult(
        intent=IntentEnum.ACCOUNT_SECURITY_BILLING, confidence=0.96, reasoning="billing"
    )
    decision = arbiter.evaluate(tweet, classification)
    assert decision.action == ArbiterAction.ESCALATE
    assert decision.is_critical is True
    assert any("ACCOUNT_SECURITY_BILLING" in r for r in decision.reasons)


# ------------------------------------------------------------
# 4. Dual-Layer Arbiter Tests (Soft Rules & Confidence)
# ------------------------------------------------------------

def test_arbiter_low_confidence_escalation(arbiter):
    tweet = "My device is acting strange"
    classification = ClassificationResult(
        intent=IntentEnum.SOFTWARE_OS_BUG, confidence=0.72, reasoning="ambiguous"
    )
    decision = arbiter.evaluate(tweet, classification)
    assert decision.action == ArbiterAction.ESCALATE
    assert decision.layer == "SOFT_RULE"
    assert any("Low intent confidence" in r for r in decision.reasons)


def test_arbiter_unauthorized_external_url_grounding(arbiter):
    tweet = "How do I fix my battery?"
    classification = ClassificationResult(
        intent=IntentEnum.DEVICE_HARDWARE_ISSUE, confidence=0.92, reasoning="battery"
    )
    draft = DraftReply(
        draft_text="Visit http://shady-third-party-repair.biz for a quick fix! ^AB",
        grounded_sources=[],
        agent_signoff="^AB",
    )
    decision = arbiter.evaluate(tweet, classification, draft=draft)
    assert decision.action == ArbiterAction.ESCALATE
    assert decision.layer == "SOFT_RULE"
    assert any("unauthorized external URL" in r for r in decision.reasons)


def test_arbiter_safe_query_auto_handles(arbiter):
    tweet = "How do I turn on Dark Mode in settings?"
    classification = ClassificationResult(
        intent=IntentEnum.HOW_TO_CONFIGURATION, confidence=0.94, reasoning="config"
    )
    draft = DraftReply(
        draft_text="Go to Settings > Display & Brightness to enable Dark Mode! ^AB",
        grounded_sources=[],
        agent_signoff="^AB",
    )
    decision = arbiter.evaluate(tweet, classification, draft=draft)
    assert decision.action == ArbiterAction.AUTO_HANDLE
    assert decision.layer == "PASSED"
    assert decision.sanitized_draft is not None


# ------------------------------------------------------------
# 5. End-to-End Pipeline Integration Tests
# ------------------------------------------------------------

def test_pipeline_end_to_end_safe_query(pipeline):
    result = pipeline.triage("How do I enable AirDrop for 10 minutes on my iPhone?")
    assert result.classification.intent == IntentEnum.HOW_TO_CONFIGURATION
    assert result.final_action == ArbiterAction.AUTO_HANDLE
    assert "^" in result.final_response
    assert len(result.retrieved_resolutions) == 3


def test_pipeline_end_to_end_critical_escalation(pipeline):
    result = pipeline.triage("Unauthorized charge of $200 on my Apple ID, refund immediately!")
    assert result.classification.intent == IntentEnum.ACCOUNT_SECURITY_BILLING
    assert result.final_action == ArbiterAction.ESCALATE
    assert "[ROUTED TO HUMAN SPECIALIST]" in result.final_response
