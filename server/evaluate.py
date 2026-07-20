import argparse
import json
from pathlib import Path

try:
    from .rag import REFUSAL_MESSAGE, answer_question, retrieve
    from .vector_store import VectorStore
except ImportError:
    from rag import REFUSAL_MESSAGE, answer_question, retrieve
    from vector_store import VectorStore


def load_questions(path):
    with open(path, "r", encoding="utf-8") as file:
        questions = json.load(file)

    if not isinstance(questions, list) or len(questions) != 15:
        raise ValueError("The evaluation file must contain exactly 15 questions.")
    return questions


def source_was_retrieved(expected_sources, matches):
    retrieved_sources = {match["source_name"] for match in matches}
    return bool(set(expected_sources) & retrieved_sources)


def citations_are_valid(citations, matches):
    if not citations:
        return False
    retrieved = {
        (match["source_name"], str(match["page_number"]))
        for match in matches
    }
    return all((citation["source"], str(citation["page"])) in retrieved for citation in citations)


def run_evaluation(question_path, report_path):
    questions = load_questions(question_path)
    store = VectorStore()
    results = []

    retrieval_hits = 0
    valid_citations = 0
    correct_refusals = 0
    grounded_answers = 0

    for item in questions:
        question = item["question"]
        expected_sources = item.get("expected_sources", [])
        answerable = item.get("answerable", True)
        document_id = item.get("document_id")

        matches = retrieve(question, store, item.get("top_k", 5), document_id)
        result = answer_question(question, store, item.get("top_k", 5), document_id, matches)
        refused = result["answer"] == REFUSAL_MESSAGE
        retrieval_hit = source_was_retrieved(expected_sources, matches) if answerable else not matches
        citation_valid = citations_are_valid(result["citations"], matches) if not refused else False
        refusal_correct = refused if not answerable else not refused
        grounded = refused if not answerable else citation_valid

        retrieval_hits += int(retrieval_hit)
        valid_citations += int(citation_valid)
        correct_refusals += int(refusal_correct)
        grounded_answers += int(grounded)

        results.append(
            {
                "question": question,
                "answer": result["answer"],
                "retrieval_hit": retrieval_hit,
                "citation_valid": citation_valid,
                "refusal_correct": refusal_correct,
                "grounded": grounded,
            }
        )

    total = len(questions)
    answerable_count = sum(item.get("answerable", True) for item in questions)
    report = {
        "questions_evaluated": total,
        "retrieval_hit_rate": round(retrieval_hits / total, 3),
        "citation_accuracy": round(valid_citations / max(1, answerable_count), 3),
        "refusal_accuracy": round(correct_refusals / total, 3),
        "basic_groundedness_score": round(grounded_answers / total, 3),
        "results": results,
    }

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate DocuChat on exactly 15 questions.")
    parser.add_argument("questions", help="Path to the evaluation JSON file.")
    parser.add_argument("--output", default="evaluation_report.json")
    arguments = parser.parse_args()

    report = run_evaluation(Path(arguments.questions), Path(arguments.output))
    summary = {key: value for key, value in report.items() if key != "results"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

