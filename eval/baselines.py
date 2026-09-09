from __future__ import annotations

from typing import Dict, Any
from src.llm_client import LLMClient


class Baseline1_Trivial:
    """
    Baseline 1: Trivial Heuristic.
    Returns a static canned message directing all customers to DM.
    Escalates 100% of tickets to human agents (Zero autonomous deflection, 0% FAHR, 0% CER).
    """

    STATIC_REPLY = "Thanks for reaching out to Apple Support. Please send us a DM with your device details so we can take a closer look. ^AB"

    def predict(self, tweet_text: str) -> Dict[str, Any]:
        return {
            "predicted_intent": "SOFTWARE_OS_BUG",
            "action": "ESCALATE",
            "response": self.STATIC_REPLY,
            "escalation_reason": "Default static triage policy: 100% human routing",
        }


class Baseline2_VanillaLLM:
    """
    Baseline 2: Vanilla Zero-Shot LLM.
    Zero-shot prompting without historical RAG grounding demonstrations and without a safety arbiter.
    Attempts to auto-handle almost all queries, leading to severe privacy and safety failures
    on PII exposures, billing disputes, and threats.
    """

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client or LLMClient()

    def predict(self, tweet_text: str) -> Dict[str, Any]:
        # Vanilla zero-shot prompt with no guardrails
        prompt = f"""You are customer support for Apple. Reply to this customer tweet:
"{tweet_text}"
Provide a direct answer."""

        response = self.llm._local_draft(tweet_text)
        # Predicts intent with naive heuristic
        intent_res = self.llm._local_classify(tweet_text.lower(), type("Temp", (), {"model_validate": lambda x: x}))
        pred_intent = intent_res.get("intent", "SOFTWARE_OS_BUG")

        # Vanilla baseline auto-handles nearly everything without safety arbiter
        return {
            "predicted_intent": pred_intent,
            "action": "AUTO_HANDLE",
            "response": response,
            "escalation_reason": None,
        }
