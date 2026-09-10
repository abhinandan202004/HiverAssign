import sys
import argparse
from src.pipeline import SupportTriagePipeline
from src.models import ArbiterAction

# Fix unicode output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

def print_separator(title=""):
    width = 70
    if title:
        padding = (width - len(title) - 2) // 2
        print("=" * padding + f" {title} " + "=" * padding)
    else:
        print("=" * width)

def run_triage(tweet: str, pipeline: SupportTriagePipeline):
    print("\n" + "-" * 70)
    print(f"📥 INBOUND TWEET: {tweet}")
    print("-" * 70)
    
    result = pipeline.triage(tweet)
    
    cls = result.classification
    print(f"\n[1] INTENT CLASSIFICATION:")
    print(f"    • Intent:     {cls.intent.value}")
    print(f"    • Confidence: {cls.confidence:.2f}")
    print(f"    • Rationale:  {cls.reasoning}")
    
    print(f"\n[2] TOP RETRIEVED RESOLUTIONS (RAG):")
    for i, doc in enumerate(result.retrieved_resolutions, 1):
        score_info = f"(RRF Score: {doc.rrf_score:.4f})" if doc.rrf_score else ""
        print(f"    {i}. [{doc.intent.value}] {score_info}")
        print(f"       Inbound:  \"{doc.query_text[:80]}...\"")
        print(f"       Response: \"{doc.resolution_text[:90]}...\"")
        
    print(f"\n[3] GENERATED DRAFT:")
    print(f"    \"{result.draft.draft_text}\"")
    
    decision = result.arbiter_decision
    action_str = "🟢 AUTO_HANDLE (Safe to send)" if decision.action == ArbiterAction.AUTO_HANDLE else "🔴 ESCALATE (Requires human review)"
    print(f"\n[4] SAFETY ARBITER DECISION:")
    print(f"    • Final Action: {action_str}")
    if decision.reasons:
        print(f"    • Decision Reasons:")
        for r in decision.reasons:
            print(f"      - {r}")
            
    print(f"\n[5] FINAL SYSTEM OUTPUT:")
    print(f"    {result.final_response}")
    print("-" * 70 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Test inbound tweets through AppleSupport Triage Pipeline")
    parser.add_argument(
        "tweet",
        nargs="?",
        type=str,
        help="The inbound tweet text to test. If omitted, starts interactive prompt."
    )
    args = parser.parse_args()

    print_separator("AppleSupport Triage Pipeline")
    print("Initializing models and retriever (this may take a few seconds)...")
    pipeline = SupportTriagePipeline()
    print("Pipeline ready!")
    print_separator()

    if args.tweet:
        run_triage(args.tweet, pipeline)
        return

    sample_queries = [
        "How do I turn on AirDrop to share photos with non-contacts?",
        "My battery is swollen and phone is extremely hot to touch! What should I do?",
        "I was charged $89.99 for an unauthorized subscription on Apple ID, refund now!",
        "Why is my screen going black randomly after the latest iOS 17 update?",
        "Where is my MacBook order #W192837482? It was supposed to arrive yesterday.",
    ]

    print("\nSelect an option or type your own tweet:")
    for i, sample in enumerate(sample_queries, 1):
        print(f"  [{i}] {sample}")
    print("  [c] Enter custom tweet")
    print("  [q] Quit\n")

    while True:
        try:
            choice = input("Enter choice (1-5, c, q) or paste a tweet: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not choice:
            continue
        if choice.lower() == "q":
            print("Exiting.")
            break
        elif choice in [str(i) for i in range(1, len(sample_queries) + 1)]:
            run_triage(sample_queries[int(choice) - 1], pipeline)
        elif choice.lower() == "c":
            custom_tweet = input("\nEnter your tweet: ").strip()
            if custom_tweet:
                run_triage(custom_tweet, pipeline)
        else:
            # Treat direct input as a tweet
            run_triage(choice, pipeline)

if __name__ == "__main__":
    main()
