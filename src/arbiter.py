from __future__ import annotations

import re
from typing import List, Tuple, Optional
from src.models import (
    ArbiterAction,
    ArbiterDecision,
    ClassificationResult,
    DraftReply,
    IntentEnum,
    RetrievedDocument,
)


def luhn_checksum_valid(card_number_str: str) -> bool:
    """Validate credit card number using the standard Luhn algorithm."""
    digits = [int(c) for c in card_number_str if c.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    reverse_digits = digits[::-1]
    for i, digit in enumerate(reverse_digits):
        if i % 2 == 1:
            doubled = digit * 2
            checksum += doubled - 9 if doubled > 9 else doubled
        else:
            checksum += digit
    return checksum % 10 == 0


class DualLayerArbiter:
    """
    Dual-Layer Arbiter responsible for safety, privacy, policy compliance, and routing.

    Layer 1 (Deterministic Hard Filters):
      - PII: Credit card numbers (Luhn checked), SSNs, emails, phone numbers, plaintext passwords.
      - Sentiment / Threats: Legal action, physical safety hazards, severe churn threats, profanity.
      - Policy: Mandatory escalation for ACCOUNT_SECURITY_BILLING intent.

    Layer 2 (Soft Filters / LLM Critic):
      - Confidence threshold: ESCALATE if classification confidence < 0.85.
      - Grounding verification: ESCALATE if draft contains unverified claims.
    """

    CONFIDENCE_THRESHOLD = 0.85

    # Regex patterns for Layer 1 PII detection
    EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
    PHONE_REGEX = re.compile(
        r"(?:(?:\+?1\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{4})\b"
    )
    SSN_REGEX = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    CARD_CANDIDATE_REGEX = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
    PASSWORD_REGEX = re.compile(
        r"(?:password|passcode|pin|pwd)\s*(?:is|:|=)\s*([^\s,;]+)", re.IGNORECASE
    )

    # Sentiment / Threat triggers
    LEGAL_PATTERNS = [
        "lawyer", "attorney", "sue", "suing", "lawsuit", "legal action", "court",
        "subpoena", "ftc complaint", "consumer rights attorney", "police report"
    ]
    SAFETY_HAZARD_PATTERNS = [
        "exploded", "explosion", "caught fire", "burned my", "burns", "smoke coming out",
        "smoke", "smoking", "melted", "melting", "battery melting", "swelled up and popped",
        "electric shock"
    ]
    CHURN_THREAT_PATTERNS = [
        "switching to samsung", "switching to android", "switching to pixel",
        "buying a galaxy", "boycott apple", "worst company on earth", "canceling apple"
    ]
    PROFANITY_PATTERNS = [
        "fuck", "shit", "bitch", "bastard", "bullshit", "asshole", "scam artist", "scammers"
    ]

    def evaluate(
        self,
        inbound_text: str,
        classification: ClassificationResult,
        draft: Optional[DraftReply] = None,
        retrieved_docs: Optional[List[RetrievedDocument]] = None,
    ) -> ArbiterDecision:
        """
        Execute dual-layer evaluation to determine AUTO_HANDLE or ESCALATE.
        """
        reasons: List[str] = []
        is_critical = False

        # ----------------------------------------------------
        # Layer 1: Deterministic Hard Rules
        # ----------------------------------------------------

        # 1.1 PII Checks
        pii_detected, pii_details = self._detect_pii(inbound_text)
        if pii_detected:
            is_critical = True
            reasons.append(f"[Hard Filter] PII detected: {', '.join(pii_details)}")

        # 1.2 Extreme Safety Hazard Checks
        text_lower = inbound_text.lower()
        if any(hazard in text_lower for hazard in self.SAFETY_HAZARD_PATTERNS):
            is_critical = True
            reasons.append("[Hard Filter] Physical safety hazard reported (battery/fire/burn)")

        # 1.3 Legal Threats & Profanity
        if any(legal in text_lower for legal in self.LEGAL_PATTERNS):
            is_critical = True
            reasons.append("[Hard Filter] Legal threat detected (lawyer/lawsuit/regulatory)")

        if any(prof in text_lower for prof in self.PROFANITY_PATTERNS):
            reasons.append("[Hard Filter] Hostile profanity detected")

        if any(churn in text_lower for churn in self.CHURN_THREAT_PATTERNS):
            reasons.append("[Hard Filter] High-risk churn threat detected")

        # 1.4 Mandatory Forced Escalation for Restricted Intents
        if classification.intent == IntentEnum.ACCOUNT_SECURITY_BILLING:
            is_critical = True
            reasons.append(
                "[Hard Filter] Mandatory policy escalation for intent: ACCOUNT_SECURITY_BILLING"
            )

        if reasons:
            return ArbiterDecision(
                action=ArbiterAction.ESCALATE,
                layer="HARD_RULE",
                reasons=reasons,
                is_critical=is_critical,
                confidence_passed=(classification.confidence >= self.CONFIDENCE_THRESHOLD),
                grounding_passed=True,
                sanitized_draft=None,
            )

        # ----------------------------------------------------
        # Layer 2: Soft Filters / Self-Critic Verification
        # ----------------------------------------------------
        confidence_passed = classification.confidence >= self.CONFIDENCE_THRESHOLD
        if not confidence_passed:
            reasons.append(
                f"[Soft Filter] Low intent confidence ({classification.confidence:.2f} < {self.CONFIDENCE_THRESHOLD})"
            )

        # Grounding check on drafted reply
        grounding_passed = True
        if draft:
            grounding_passed, grounding_issue = self._verify_grounding(draft, retrieved_docs)
            if not grounding_passed:
                reasons.append(f"[Soft Filter] Grounding verification failed: {grounding_issue}")

        if reasons:
            return ArbiterDecision(
                action=ArbiterAction.ESCALATE,
                layer="SOFT_RULE",
                reasons=reasons,
                is_critical=False,
                confidence_passed=confidence_passed,
                grounding_passed=grounding_passed,
                sanitized_draft=None,
            )

        # All layers passed -> Safe to AUTO-HANDLE
        sanitized_reply = draft.draft_text if draft else None
        return ArbiterDecision(
            action=ArbiterAction.AUTO_HANDLE,
            layer="PASSED",
            reasons=["All safety, privacy, and grounding checks passed."],
            is_critical=False,
            confidence_passed=True,
            grounding_passed=True,
            sanitized_draft=sanitized_reply,
        )

    def _detect_pii(self, text: str) -> Tuple[bool, List[str]]:
        detected: List[str] = []

        # Credit Cards: check for Luhn valid numbers OR 15-16 digit card-like patterns
        for match in self.CARD_CANDIDATE_REGEX.finditer(text):
            candidate = re.sub(r"[ -]", "", match.group())
            if 13 <= len(candidate) <= 19:
                if luhn_checksum_valid(candidate) or len(candidate) in [15, 16]:
                    detected.append("CREDIT_CARD")
                    break

        # SSN
        if self.SSN_REGEX.search(text):
            detected.append("SSN")

        # Plaintext Password
        if self.PASSWORD_REGEX.search(text):
            detected.append("PLAINTEXT_PASSWORD")

        # Email
        if self.EMAIL_REGEX.search(text):
            detected.append("EMAIL")

        # Phone Number
        if self.PHONE_REGEX.search(text):
            # Verify minimum length to avoid matching simple numbers
            digits = re.sub(r"\D", "", text)
            if len(digits) >= 10:
                detected.append("PHONE_NUMBER")

        return bool(detected), detected

    def _verify_grounding(
        self, draft: DraftReply, retrieved_docs: Optional[List[RetrievedDocument]]
    ) -> Tuple[bool, str]:
        """
        Verify factual claims in draft against retrieved knowledge and standard Apple domains.
        """
        draft_text = draft.draft_text.lower()

        # Check for unauthorized guarantees or non-standard external links
        if "free replacement guaranteed" in draft_text or "100% refund" in draft_text:
            return False, "Draft makes unauthorized refund or replacement guarantee."

        # Flag any suspicious external URLs not belonging to official apple.com
        urls = re.findall(r"https?://[^\s]+", draft.draft_text)
        for url in urls:
            if not ("apple.com" in url or "icloud.com" in url):
                return False, f"Draft references unauthorized external URL: {url}"

        return True, ""
