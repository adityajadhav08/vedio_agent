import os
import sys

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.extractor import analyze_full_meeting
from core.rag_engine import build_rag_chain, ask_question

def run_pipeline(source :str, language :str = "english") -> dict:
    print("\n=== Starting AI Video Assistant ===")

    chunks = process_input(source)

    transcript = transcribe_all(chunks, language)
    print(f"\n[TRANSCRIPTION] First 300 characters:\n{transcript[:300]}...\n")

    # Save transcript locally so it is never lost
    transcript_file = os.path.join("downloades", "transcript.txt")
    with open(transcript_file, "w", encoding="utf-8") as f:
        f.write(transcript)
    print(f"[SAVED] Full transcript saved to: {transcript_file}")

    # Phase 1 — Comprehensive meeting analysis in 1 fast Gemini call
    print("\n[AI ANALYSIS] Analyzing meeting with Gemini (Title, Summary, Actions, Decisions, Questions)...")
    analysis = analyze_full_meeting(transcript)
    print("[AI ANALYSIS] Analysis complete!")

    # Phase 2 — Build ChromaDB Vector Store for RAG
    print("\n[RAG PIPELINE] Building Vector Store for Chat & Q&A...")
    rag_chain = build_rag_chain(transcript)

    return {
        "title": analysis["title"],
        "transcript": transcript,
        "summary": analysis["summary"],
        "action_items": analysis["action_items"],
        "key_decisions": analysis["key_decisions"],
        "open_questions": analysis["open_questions"],
        "rag_chain": rag_chain,
    }

if __name__ == "__main__":
    # CLI entry point
    source = input("Enter YouTube URL or local file path: ").strip().strip("\"'")
    language = input("Language (english/hinglish): ").strip() or "english"
    result = run_pipeline(source, language)

    print("\n" + "=" * 60)
    print(f"[TITLE]: {result['title']}")
    print(f"\n[SUMMARY]:\n{result['summary']}")
    print(f"\n[ACTION ITEMS]:\n{result['action_items']}")
    print(f"\n[KEY DECISIONS]:\n{result['key_decisions']}")
    print(f"\n[OPEN QUESTIONS]:\n{result['open_questions']}")
    print("=" * 60)

    # Phase 2 — Chat with your meeting via RAG
    print("\n--- Chat with your meeting (type 'exit' to quit) ---\n")
    rag_chain = result["rag_chain"]
    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break
        if not question:
            continue
        answer = ask_question(rag_chain, question)
        print(f"\nAssistant: {answer}\n")