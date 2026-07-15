import json
from pathlib import Path
from typing import Any


DATASET_PATH = Path("data/evaluation/evaluation_dataset.json")

REQUIRED_FIELDS = {
    "question_id",
    "question",
    "expected_answer",
    "expected_document",
    "expected_pages",
    "category",
    "difficulty",
    "answerable",
    "notes",
}


def validate_record(record: dict[str, Any], index: int) -> list[str]:
    errors: list[str] = []

    missing_fields = REQUIRED_FIELDS - record.keys()

    if missing_fields:
        errors.append(
            f"Record {index} is missing fields: "
            f"{sorted(missing_fields)}"
        )

    question_id = record.get("question_id")

    if not isinstance(question_id, str) or not question_id.strip():
        errors.append(f"Record {index} has an invalid question_id.")

    question = record.get("question")

    if not isinstance(question, str) or not question.strip():
        errors.append(f"Record {index} has an invalid question.")

    answerable = record.get("answerable")

    if not isinstance(answerable, bool):
        errors.append(f"Record {index} has an invalid answerable value.")

    expected_pages = record.get("expected_pages")

    if not isinstance(expected_pages, list):
        errors.append(f"Record {index} expected_pages must be a list.")

    difficulty = record.get("difficulty")

    if difficulty not in {"easy", "medium", "hard"}:
        errors.append(
            f"Record {index} difficulty must be easy, medium, or hard."
        )

    if answerable is True:
        expected_document = record.get("expected_document")
        expected_answer = record.get("expected_answer")

        if not expected_document:
            errors.append(
                f"Record {index} is answerable but has no expected document."
            )

        if not expected_answer:
            errors.append(
                f"Record {index} is answerable but has no expected answer."
            )

        if not expected_pages:
            errors.append(
                f"Record {index} is answerable but has no expected pages."
            )

    return errors


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH.resolve()}"
        )

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list):
        raise ValueError("The dataset root must be a JSON list.")

    all_errors: list[str] = []
    question_ids: set[str] = set()

    for index, record in enumerate(dataset, start=1):
        if not isinstance(record, dict):
            all_errors.append(f"Record {index} must be a JSON object.")
            continue

        all_errors.extend(validate_record(record, index))

        question_id = record.get("question_id")

        if question_id in question_ids:
            all_errors.append(
                f"Duplicate question_id found: {question_id}"
            )

        question_ids.add(question_id)

    if all_errors:
        print("Dataset validation failed:")

        for error in all_errors:
            print(f"- {error}")

        raise SystemExit(1)

    answerable_count = sum(
        1 for item in dataset if item["answerable"]
    )
    unanswerable_count = len(dataset) - answerable_count

    print("Dataset validation passed.")
    print(f"Total questions: {len(dataset)}")
    print(f"Answerable questions: {answerable_count}")
    print(f"Unanswerable questions: {unanswerable_count}")


if __name__ == "__main__":
    main()