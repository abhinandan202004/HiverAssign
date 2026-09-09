from __future__ import annotations

from typing import List, Optional
from src.models import DraftReply, RetrievedDocument, ClassificationResult
from src.llm_client import LLMClient


GENERATOR_SYSTEM_PROMPT = """You are an official customer support representative for @AppleSupport on Twitter.
Your goal is to compose an empathetic, accurate, concise, and brand-aligned public tweet reply (under 280 characters).

Brand Guidelines:
1. Tone: Empathetic, calm, professional, and directly actionable.
2. Grounding: Refer to official Apple procedures from the provided historical examples. Never hallucinate third-party repair links or unofficial advice.
3. Official Links: Use official domains when relevant (getsupport.apple.com, apple.com/orderstatus, reportaproblem.apple.com).
4. Security: Never ask for passwords, credit cards, or sensitive personal data in a public tweet.
5. Sign-off: Always end with a personal brand sign-off initials (e.g. ^AB).
"""


class DraftGenerator:
    def __init__(self, llm_client: Optional[LLMClient] = None, default_signoff: str = "^AB"):
        self.llm = llm_client or LLMClient()
        self.default_signoff = default_signoff

    def generate(
        self,
        inbound_text: str,
        classification: ClassificationResult,
        retrieved_docs: List[RetrievedDocument],
    ) -> DraftReply:
        """
        Generate a grounded draft reply utilizing historical resolutions as in-context demonstrations.
        """
        # Format top-3 historical demonstrations
        demo_context = ""
        grounded_sources = []
        for i, doc in enumerate(retrieved_docs, start=1):
            grounded_sources.append(doc.tweet_id)
            demo_context += f"\n--- Historical Resolution #{i} ({doc.intent.value}) ---\n"
            demo_context += f"Customer: {doc.query_text}\n"
            demo_context += f"@AppleSupport: {doc.resolution_text}\n"

        prompt = f"""Incoming Customer Tweet:
"{inbound_text}"

Classified Intent: {classification.intent.value} (Confidence: {classification.confidence:.2f})

Historical Verified Resolutions for Grounding:
{demo_context}

Task:
Draft a concise, helpful, and empathetic tweet reply for @AppleSupport grounded in the historical advice above. Include the sign-off {self.default_signoff} at the end."""

        raw_reply = self.llm.generate_completion(
            prompt=prompt,
            system_prompt=GENERATOR_SYSTEM_PROMPT,
            temperature=0.0,
        )

        reply_text = raw_reply.strip() if isinstance(raw_reply, str) else str(raw_reply)

        # Ensure official sign-off is present
        if not any(reply_text.endswith(f"^{initials}") for initials in ["AB", "DM", "VK", "TL", "JM"]):
            reply_text = f"{reply_text} {self.default_signoff}"

        return DraftReply(
            draft_text=reply_text,
            grounded_sources=grounded_sources,
            agent_signoff=self.default_signoff,
        )
