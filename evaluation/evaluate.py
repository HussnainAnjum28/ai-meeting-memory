"""
evaluate.py

Evaluates the AI Meeting Memory RAG pipeline against a small
hand-labeled test set (evaluation/test_cases.json).

Measures:
  - Retrieval quality: did any retrieved chunk contain an expected keyword?
  - Answer correctness (proxy): does the generated answer contain the
    expected keywords for answerable questions?
  - Hallucination rate: for questions with NO answer in the transcript,
    does the system correctly say "I don't know" instead of inventing one?

This is a lightweight, keyword-based evaluation (not a full NLP metric
suite), chosen deliberately so it runs instantly with no extra
dependencies and is easy to read and extend.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

from src.retrieval.vector_store import VectorStore
from src.rag.llm_interface import get_llm
from src.rag.qa_pipeline import RAGPipeline

REFUSAL_PHRASES = [
    "couldn't find", "could not find", "don't know", "do not know",
    "no information", "not mentioned", "not discussed", "not present",
    "not stated", "not found", "unable to find", "not covered",
]


def is_refusal(answer: str) -> bool:
    """Checks whether the answer indicates the system correctly declined
    to answer (used to detect hallucination on unanswerable questions)."""
    lower = answer.lower()
    return any(phrase in lower for phrase in REFUSAL_PHRASES)


def contains_any_keyword(text: str, keywords: list) -> bool:
    lower = text.lower()
    return any(kw.lower() in lower for kw in keywords)


def run_evaluation(test_cases_path: str):
    with open(test_cases_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    meeting_id = data["meeting_id"]
    test_cases = data["test_cases"]

    print("Loading pipeline (this may take a minute)...")
    store = VectorStore()
    llm = get_llm(provider="ollama", model_name="llama3.2")
    pipeline = RAGPipeline(vector_store=store, llm=llm)

    results = []
    retrieval_hits = 0
    retrieval_total = 0
    answerable_correct = 0
    answerable_total = 0
    hallucination_count = 0
    unanswerable_total = 0

    print(f"\nRunning {len(test_cases)} test case(s) against meeting '{meeting_id}'...\n")

    for i, case in enumerate(test_cases, 1):
        question = case["question"]
        expected_keywords = case.get("expected_keywords", [])
        case_type = case.get("type", "answerable")

        result = pipeline.answer(question, meeting_id=meeting_id)
        answer = result.answer
        evidence_text = " ".join(e.text for e in result.evidence)

        case_result = {
            "question": question,
            "type": case_type,
            "answer": answer,
        }

        if case_type == "answerable":
            answerable_total += 1
            retrieval_total += 1

            retrieved_ok = contains_any_keyword(evidence_text, expected_keywords) if expected_keywords else True
            if retrieved_ok:
                retrieval_hits += 1
            case_result["retrieval_hit"] = retrieved_ok

            answer_ok = contains_any_keyword(answer, expected_keywords) if expected_keywords else True
            if answer_ok:
                answerable_correct += 1
            case_result["answer_correct"] = answer_ok

        elif case_type == "unanswerable":
            unanswerable_total += 1
            refused = is_refusal(answer)
            if not refused:
                hallucination_count += 1
            case_result["correctly_refused"] = refused

        results.append(case_result)

        status_parts = []
        if "retrieval_hit" in case_result:
            status_parts.append(f"retrieval={'OK' if case_result['retrieval_hit'] else 'MISS'}")
        if "answer_correct" in case_result:
            status_parts.append(f"answer={'OK' if case_result['answer_correct'] else 'MISS'}")
        if "correctly_refused" in case_result:
            status_parts.append(f"refused={'YES' if case_result['correctly_refused'] else 'NO (hallucinated)'}")

        print(f"[{i}/{len(test_cases)}] {question}")
        print(f"   {' | '.join(status_parts)}")

    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    if retrieval_total > 0:
        recall_at_k = retrieval_hits / retrieval_total * 100
        print(f"\nRetrieval Recall@K:        {recall_at_k:.1f}%  ({retrieval_hits}/{retrieval_total})")

    if answerable_total > 0:
        answer_accuracy = answerable_correct / answerable_total * 100
        print(f"Answer Correctness:       {answer_accuracy:.1f}%  ({answerable_correct}/{answerable_total})")

    if unanswerable_total > 0:
        hallucination_rate = hallucination_count / unanswerable_total * 100
        print(f"Hallucination Rate:       {hallucination_rate:.1f}%  ({hallucination_count}/{unanswerable_total})")
        print(f"   (lower is better - this measures how often the system")
        print(f"    invented an answer instead of saying 'I don't know')")

    print("\n" + "=" * 60)

    report_path = Path("evaluation/last_run_report.json")
    report_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDetailed results saved to {report_path}")


if __name__ == "__main__":
    test_path = sys.argv[1] if len(sys.argv) > 1 else "evaluation/test_cases.json"
    run_evaluation(test_path)

