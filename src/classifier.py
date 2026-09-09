from __future__ import annotations

import json
from typing import Optional
from src.models import IntentEnum, ClassificationResult
from src.llm_client import LLMClient


SYSTEM_PROMPT = """You are the Lead Triage Classifier for @AppleSupport on Twitter.
Your role is to classify incoming customer tweets into exactly ONE of the 6 discrete intent categories with a calibrated confidence score [0.0 to 1.0].

Categories:
1. DEVICE_HARDWARE_ISSUE: Physical damage, screen crack, battery swelling/drain due to degradation, hardware buttons, charging port, speaker distortion, camera lens, overheating hardware.
2. SOFTWARE_OS_BUG: iOS updates, freezes, app crashing, bootloop / stuck on Apple logo, Wi-Fi/Bluetooth disconnects, system lag, Safari glitches.
3. ACCOUNT_SECURITY_BILLING: Apple ID locked, 2FA code issues, unauthorized charges, App Store subscriptions, refund requests, phishing/scams, password recovery.
4. HOW_TO_CONFIGURATION: Questions on how to configure settings (AirDrop, iCloud backup, Dark Mode, transfer data, pair AirPods, notifications).
5. ORDER_DELIVERY_STATUS: Tracking shipments, Apple Store pickup timing, parcel delivery delays, order dispatch status.
6. CHITCHAT_OUT_OF_SCOPE: Friendly fan banter, jokes, praise/criticism without actionable issue, Tim Cook mentions, competitor comparisons, spam.

Output format must be valid JSON strictly matching the ClassificationResult schema:
{
  "intent": "<ONE_OF_THE_6_INTENTS>",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<concise justification>"
}"""


class IntentClassifier:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    def classify(self, tweet_text: str) -> ClassificationResult:
        """
        Classify inbound tweet into one of 6 target intents with confidence score.
        """
        prompt = f"""Incoming Tweet: "{tweet_text}"

Analyze this customer inquiry, determine the primary intent category, assign a calibrated confidence score, and state your reasoning."""

        result = self.llm.generate_completion(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            response_model=ClassificationResult,
            temperature=0.0,
        )

        if not isinstance(result, ClassificationResult):
            # Safe fallback if raw string or dict is received
            if isinstance(result, dict):
                return ClassificationResult.model_validate(result)
            return ClassificationResult(
                intent=IntentEnum.SOFTWARE_OS_BUG,
                confidence=0.70,
                reasoning="Fallback classification due to unparsed output.",
            )

        return result
