#!/usr/bin/env python3
"""Benchmark multiple OpenRouter models on the same Wikipedia substitutions."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import re
import sys
import traceback
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from openrouter_credits import start_credit_tracking
from wiki_substitution_test import (
    build_prompt,
    build_replacement_prompt,
    build_test_case,
    call_openrouter,
    evaluate,
    extract_json_object,
    fetch_wikipedia_extract,
    normalize_sentence,
    pick_sentence_index,
    split_sentences,
)

DEFAULT_ARTICLES = [
    "Alan Turing",
    "Marie Curie",
    "Isaac Newton",
    "Ada Lovelace",
    "Nikola Tesla",
    "Albert Einstein",
    "Rosalind Franklin",
    "Dmitri Mendeleev",
    "Grace Hopper",
    "Carl Linnaeus",
    "Johannes Kepler",
    "Barbara McClintock",
    "Niels Bohr",
    "James Clerk Maxwell",
    "Katherine Johnson",
    "Archimedes",
    "Michael Faraday",
    "Gregor Mendel",
    "Louis Pasteur",
    "Srinivasa Ramanujan",
]


def sanitize_slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip()).strip("-").lower()
    return slug or "run"


def choose_benchmark_id(seed: int) -> str:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_bench_s{seed}"


def json_write(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def json_read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def append_log(path: Path, message: str) -> None:
    stamp = dt.datetime.now().isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{stamp}] {message}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run multiple models on one shared set of Wikipedia substitutions."
    )
    p.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="OpenRouter model ids, e.g. openai/gpt-5-mini openai/gpt-5",
    )
    p.add_argument(
        "--articles-file",
        default=None,
        help="Optional file with one Wikipedia title per line",
    )
    p.add_argument(
        "--num-cases",
        type=int,
        default=15,
        help="How many article substitutions to benchmark",
    )
    p.add_argument("--seed", type=int, default=42, help="Seed for deterministic case selection")
    p.add_argument("--max-chars", type=int, default=7000, help="Max chars from each article")
    p.add_argument(
        "--replacement-model",
        default="openai/gpt-5-mini",
        help="Model used to generate replacement sentences",
    )
    p.add_argument(
        "--replacement-temperature",
        type=float,
        default=0.7,
        help="Temperature for replacement generation",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for evaluation models",
    )
    p.add_argument(
        "--benchmarks-dir",
        default="benchmarks",
        help="Directory for benchmark artifacts",
    )
    p.add_argument("--benchmark-id", default=None, help="Benchmark identifier")
    p.add_argument("--resume", action="store_true", help="Resume existing benchmark")
    return p.parse_args()


def load_titles(args: argparse.Namespace) -> list[str]:
    if args.articles_file:
        content = Path(args.articles_file).read_text(encoding="utf-8")
        titles = [line.strip() for line in content.splitlines() if line.strip()]
    else:
        titles = DEFAULT_ARTICLES
    if len(titles) < args.num_cases:
        raise ValueError(f"Need at least {args.num_cases} titles, got {len(titles)}.")
    return titles[: args.num_cases]


def generate_cases(
    args: argparse.Namespace,
    benchmark_dir: Path,
    api_key: str,
    titles: list[str],
    log_path: Path,
) -> list[dict]:
    cases_dir = benchmark_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    case_paths: list[str] = []
    cases: list[dict] = []

    for i, title in enumerate(titles, start=1):
        case_id = f"case_{i:02d}_{sanitize_slug(title)}"
        case_path = cases_dir / f"{case_id}.json"
        article_path = cases_dir / f"{case_id}_article.txt"

        append_log(log_path, f"building_case {case_id}")
        article = fetch_wikipedia_extract(title)
        article_path.write_text(article, encoding="utf-8")
        base_sentences = split_sentences(article[: args.max_chars])
        if len(base_sentences) < 8:
            raise ValueError(f"Article too short for case generation: {title}")

        rng = random.Random(args.seed + i)
        target_idx = pick_sentence_index(base_sentences, rng)
        original_sentence = base_sentences[target_idx]
        previous_sentence = base_sentences[target_idx - 1] if target_idx > 0 else ""
        next_sentence = base_sentences[target_idx + 1] if target_idx + 1 < len(base_sentences) else ""

        replacement_messages = build_replacement_prompt(
            title=title,
            original_sentence=original_sentence,
            previous_sentence=previous_sentence,
            next_sentence=next_sentence,
        )
        replacement_raw = call_openrouter(
            api_key=api_key,
            model=args.replacement_model,
            messages=replacement_messages,
            temperature=args.replacement_temperature,
        )
        (cases_dir / f"{case_id}_replacement_raw.txt").write_text(replacement_raw, encoding="utf-8")
        replacement_json = extract_json_object(replacement_raw)
        replacement_sentence = str(replacement_json.get("replacement_sentence", "")).strip()
        if not replacement_sentence:
            raise ValueError(f"Replacement model returned empty sentence for {title}")
        if normalize_sentence(replacement_sentence) == normalize_sentence(original_sentence):
            raise ValueError(f"Replacement equals original for {title}")

        test_case = build_test_case(
            title=title,
            article_text=article,
            replacement_sentence=replacement_sentence,
            seed=args.seed + i,
            max_chars=args.max_chars,
            forced_index=target_idx,
        )
        case = {
            "case_id": case_id,
            "title": title,
            "corrupted_index": test_case.corrupted_index,
            "original_sentence": test_case.original_sentence,
            "replacement_sentence": test_case.replacement_sentence,
            "corrupted_text": test_case.corrupted_text,
            "replacement_model": args.replacement_model,
            "replacement_temperature": args.replacement_temperature,
            "replacement_rationale": replacement_json.get("rationale"),
        }
        json_write(case_path, case)
        cases.append(case)
        case_paths.append(str(case_path))

    json_write(
        benchmark_dir / "dataset_manifest.json",
        {
            "num_cases": len(cases),
            "titles": titles,
            "case_paths": case_paths,
            "seed": args.seed,
            "max_chars": args.max_chars,
            "replacement_model": args.replacement_model,
            "replacement_temperature": args.replacement_temperature,
        },
    )
    return cases


def evaluate_model_on_case(
    api_key: str,
    model: str,
    case: dict,
    temperature: float,
) -> dict:
    class _Case:
        title = case["title"]
        corrupted_text = case["corrupted_text"]

    messages = build_prompt(_Case())
    raw = call_openrouter(
        api_key=api_key,
        model=model,
        messages=messages,
        temperature=temperature,
    )
    parsed = extract_json_object(raw)
    guessed = str(parsed.get("corrupted_sentence", "")).strip()
    success, score = evaluate(
        type("T", (), {"replacement_sentence": case["replacement_sentence"]})(),
        guessed,
    )
    return {
        "raw": raw,
        "parsed": parsed,
        "guessed_corrupted_sentence": guessed,
        "success": success,
        "similarity_score": score,
        "model_confidence": parsed.get("confidence"),
        "ground_truth_corrupted_sentence": case["replacement_sentence"],
    }


def load_cases_from_manifest(benchmark_dir: Path) -> list[dict]:
    manifest = json_read(benchmark_dir / "dataset_manifest.json")
    return [json_read(Path(p)) for p in manifest["case_paths"]]


def main() -> int:
    load_dotenv()
    args = parse_args()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("[error] Missing OPENROUTER_API_KEY in environment/.env", file=sys.stderr)
        return 1

    benchmark_id = args.benchmark_id or choose_benchmark_id(args.seed)
    benchmark_dir = Path(args.benchmarks_dir) / benchmark_id
    log_path = benchmark_dir / "benchmark.log"
    summary_path = benchmark_dir / "summary.json"

    if benchmark_dir.exists() and args.benchmark_id and not args.resume:
        print(
            f"[error] Benchmark exists at {benchmark_dir}. Use --resume to continue.",
            file=sys.stderr,
        )
        return 1
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    append_log(log_path, f"benchmark_started id={benchmark_id}")
    json_write(benchmark_dir / "args.json", vars(args))
    start_credit_tracking(
        api_key=api_key,
        log_fn=lambda msg: append_log(log_path, msg),
    )

    try:
        if args.resume and (benchmark_dir / "dataset_manifest.json").exists():
            append_log(log_path, "loading_existing_dataset")
            cases = load_cases_from_manifest(benchmark_dir)
        else:
            titles = load_titles(args)
            append_log(log_path, f"generating_dataset num_cases={len(titles)}")
            cases = generate_cases(args, benchmark_dir, api_key, titles, log_path)
    except Exception as exc:
        append_log(log_path, "dataset_generation_failed")
        append_log(log_path, traceback.format_exc())
        print(f"[error] Failed generating dataset: {exc}", file=sys.stderr)
        return 1

    results_root = benchmark_dir / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    overall: dict[str, Any] = {
        "benchmark_id": benchmark_id,
        "num_cases": len(cases),
        "models": {},
    }

    for model in args.models:
        model_slug = sanitize_slug(model)
        model_dir = results_root / model_slug
        model_dir.mkdir(parents=True, exist_ok=True)
        append_log(log_path, f"model_start {model}")

        model_records: list[dict] = []
        success_count = 0
        score_sum = 0.0

        for case in cases:
            case_result_path = model_dir / f"{case['case_id']}.json"
            case_raw_path = model_dir / f"{case['case_id']}_raw.txt"

            if args.resume and case_result_path.exists():
                record = json_read(case_result_path)
                model_records.append(record)
                success_count += int(bool(record.get("success")))
                score_sum += float(record.get("similarity_score", 0.0))
                append_log(log_path, f"skip_existing {model} {case['case_id']}")
                continue

            try:
                append_log(log_path, f"eval_case {model} {case['case_id']}")
                result = evaluate_model_on_case(
                    api_key=api_key,
                    model=model,
                    case=case,
                    temperature=args.temperature,
                )
                case_raw_path.write_text(result["raw"], encoding="utf-8")
                record = {
                    "case_id": case["case_id"],
                    "title": case["title"],
                    "model": model,
                    "success": result["success"],
                    "similarity_score": result["similarity_score"],
                    "model_confidence": result["model_confidence"],
                    "ground_truth_corrupted_sentence": result["ground_truth_corrupted_sentence"],
                    "guessed_corrupted_sentence": result["guessed_corrupted_sentence"],
                    "parsed_response": result["parsed"],
                }
                json_write(case_result_path, record)
                model_records.append(record)
                success_count += int(bool(record["success"]))
                score_sum += float(record["similarity_score"])
            except Exception as exc:
                error_record = {
                    "case_id": case["case_id"],
                    "title": case["title"],
                    "model": model,
                    "error": str(exc),
                }
                json_write(case_result_path, error_record)
                append_log(log_path, f"case_failed {model} {case['case_id']} {exc}")

        completed = len([r for r in model_records if "error" not in r])
        avg_score = (score_sum / completed) if completed else 0.0
        accuracy = (success_count / completed) if completed else 0.0
        model_summary = {
            "model": model,
            "completed_cases": completed,
            "success_count": success_count,
            "accuracy": accuracy,
            "avg_similarity_score": avg_score,
        }
        overall["models"][model] = model_summary
        json_write(model_dir / "model_summary.json", model_summary)
        append_log(log_path, f"model_done {model} accuracy={accuracy:.3f}")

    json_write(summary_path, overall)
    append_log(log_path, "benchmark_finished")

    print(f"Benchmark ID: {benchmark_id}")
    print(f"Benchmark directory: {benchmark_dir}")
    print(json.dumps(overall, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
