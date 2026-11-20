"""Retrieval evaluation against labelled source filenames and PDF pages."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import EVALUATION_CSV, REPORTS_DIR
from src.rag_core import load_index


def normalize_source(value: str) -> str:
    return Path(value.strip()).name.casefold()


def normalize_page(value: object) -> str:
    text = str(value or "").strip()
    return text.removeprefix("page ").removeprefix("Page ").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval against labelled source pages.")
    parser.add_argument("--top-k", type=int, default=4, help="Number of chunks retrieved per question.")
    args = parser.parse_args()
    if args.top_k < 1:
        raise SystemExit("--top-k must be at least 1.")
    if not EVALUATION_CSV.exists():
        raise SystemExit(f"Evaluation file not found: {EVALUATION_CSV}")

    cases = pd.read_csv(EVALUATION_CSV).fillna("")
    required_columns = {"question", "relevant_sources"}
    if not required_columns.issubset(cases.columns):
        raise SystemExit("CSV must contain question and relevant_sources columns.")

    index = load_index()
    retriever = index.as_retriever(similarity_top_k=args.top_k)
    rows: list[dict[str, object]] = []
    precision_values: list[float] = []
    recall_values: list[float] = []
    reciprocal_ranks: list[float] = []

    for _, case in cases.iterrows():
        question = str(case["question"]).strip()
        expected_sources = {
            normalize_source(item)
            for item in str(case["relevant_sources"]).split("|")
            if item.strip()
        }
        expected_pages = {
            normalize_page(item)
            for item in str(case.get("relevant_pages", "")).split("|")
            if item.strip()
        }
        if not question or not expected_sources:
            continue

        results = retriever.retrieve(question)
        retrieved = []
        for result in results:
            metadata = result.node.metadata or {}
            source = normalize_source(str(metadata.get("source") or metadata.get("file_name") or ""))
            page = normalize_page(metadata.get("page_label") or metadata.get("page_number"))
            retrieved.append((source, page))

        if expected_pages:
            relevant = [(source, page) for source, page in retrieved if source in expected_sources and page in expected_pages]
            relevant_units = {(source, page) for source in expected_sources for page in expected_pages}
            matched_units = set(relevant)
        else:
            relevant = [(source, page) for source, page in retrieved if source in expected_sources]
            relevant_units = {(source, "") for source in expected_sources}
            matched_units = {(source, "") for source, _ in relevant}

        precision = len(relevant) / args.top_k
        recall = len(matched_units & relevant_units) / len(relevant_units)
        reciprocal_rank = next(
            (1.0 / rank for rank, item in enumerate(retrieved, start=1)
             if item[0] in expected_sources and (not expected_pages or item[1] in expected_pages)),
            0.0,
        )
        precision_values.append(precision)
        recall_values.append(recall)
        reciprocal_ranks.append(reciprocal_rank)
        rows.append(
            {
                "question": question,
                "expected_sources": " | ".join(sorted(expected_sources)),
                "expected_pages": " | ".join(sorted(expected_pages)),
                "retrieved_sources_ranked": " | ".join(
                    f"{source} page {page}" if page else source for source, page in retrieved
                ),
                "precision_at_k": precision,
                "recall_at_k": recall,
                "reciprocal_rank": reciprocal_rank,
            }
        )

    if not rows:
        raise SystemExit("No labelled evaluation rows found. Fill relevant_sources in the CSV first.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = REPORTS_DIR / "retrieval_results.csv"
    pd.DataFrame(rows).to_csv(results_path, index=False)
    print(f"Queries evaluated: {len(rows)}")
    print(f"Precision@{args.top_k}: {sum(precision_values) / len(precision_values):.3f}")
    print(f"Recall@{args.top_k}:    {sum(recall_values) / len(recall_values):.3f}")
    print(f"MRR:              {sum(reciprocal_ranks) / len(reciprocal_ranks):.3f}")
    print(f"Per-question results: {results_path}")


if __name__ == "__main__":
    main()
