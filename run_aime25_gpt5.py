#!/usr/bin/env python3
"""Run GPT-5 on the math-ai/aime25 test set via OpenRouter."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

OPENROUTER_API = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_DATASET_URI = "hf://datasets/math-ai/aime25/test.jsonl"


def sanitize_slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip()).strip("-").lower()
    return slug or "run"


def choose_run_id(model: str) -> str:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{sanitize_slug(model)}_aime25"


def json_write(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def json_read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def append_log(path: Path, message: str) -> None:
    stamp = dt.datetime.now().isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{stamp}] {message}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run GPT-5 on the math-ai/aime25 dataset.")
    p.add_argument("--dataset-uri", default=DEFAULT_DATASET_URI, help="Dataset JSONL URI")
    p.add_argument("--model", default="openai/gpt-5", help="OpenRouter model id")
    p.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default="low",
        help="Reasoning effort sent to the model",
    )
    p.add_argument("--max-rows", type=int, default=None, help="Optional limit for quick runs")
    p.add_argument("--runs-dir", default="dataset_runs", help="Directory for run artifacts")
    p.add_argument("--run-id", default=None, help="Run identifier")
    p.add_argument("--resume", action="store_true", help="Resume an existing run")
    return p.parse_args()


def load_dataset(uri: str, max_rows: int | None) -> pd.DataFrame:
    df = pd.read_json(uri, lines=True)
    if max_rows is not None:
        df = df.head(max_rows).copy()
    return df.reset_index(drop=True)


def detect_problem_column(df: pd.DataFrame) -> str:
    for candidate in ["problem", "question", "prompt", "input"]:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Could not find problem column. Columns: {list(df.columns)}")


def detect_answer_column(df: pd.DataFrame) -> str | None:
    for candidate in ["answer", "final_answer", "solution", "target", "expected_answer"]:
        if candidate in df.columns:
            return candidate
    return None


def normalize_answer(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_final_answer(text: str) -> str:
    stripped = text.strip()
    patterns = [
        r"(?im)^final answer\s*[:\-]\s*(.+)$",
        r"(?im)^answer\s*[:\-]\s*(.+)$",
        r"\\boxed\{([^}]*)\}",
        r"(?im)^therefore[, ]+the answer is\s+(.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, stripped)
        if match:
            return match.group(1).strip().rstrip(".")
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def build_messages(problem_text: str) -> list[dict[str, str]]:
    system = (
        "You are solving an AIME-style math problem. "
        "Provide a concise solution and end with a line in the exact format "
        "'Final Answer: <answer>'."
    )
    user = f"Solve this problem:\n\n{problem_text}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def call_openrouter(
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    reasoning_effort: str,
) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "reasoning": {"effort": reasoning_effort},
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OPENROUTER_API,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost",
            "X-Title": "ProxyPrior AIME25 Runner",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def extract_text_from_response(payload: dict) -> str:
    choices = payload.get("choices", [])
    if not choices:
        raise ValueError("No choices returned by OpenRouter.")
    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [chunk.get("text", "") for chunk in content if isinstance(chunk, dict)]
        text = "".join(parts).strip()
        if text:
            return text
    raise ValueError("Could not parse text content from OpenRouter response.")


def main() -> int:
    load_dotenv()
    args = parse_args()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("[error] Missing OPENROUTER_API_KEY in environment/.env", file=sys.stderr)
        return 1

    run_id = args.run_id or choose_run_id(args.model)
    run_dir = Path(args.runs_dir) / run_id
    rows_dir = run_dir / "rows"
    log_path = run_dir / "run.log"

    if run_dir.exists() and args.run_id and not args.resume:
        print(f"[error] Run exists at {run_dir}. Use --resume to continue.", file=sys.stderr)
        return 1

    run_dir.mkdir(parents=True, exist_ok=True)
    rows_dir.mkdir(parents=True, exist_ok=True)
    json_write(run_dir / "args.json", vars(args))
    append_log(log_path, f"run_started id={run_id}")

    try:
        df = load_dataset(args.dataset_uri, args.max_rows)
        problem_col = detect_problem_column(df)
        answer_col = detect_answer_column(df)
        json_write(
            run_dir / "dataset_info.json",
            {
                "dataset_uri": args.dataset_uri,
                "num_rows": len(df),
                "columns": list(df.columns),
                "problem_column": problem_col,
                "answer_column": answer_col,
            },
        )
    except Exception as exc:
        append_log(log_path, "dataset_load_failed")
        append_log(log_path, traceback.format_exc())
        print(f"[error] Failed to load dataset: {exc}", file=sys.stderr)
        return 1

    correct = 0
    completed = 0

    for idx, row in df.iterrows():
        row_id = f"row_{idx:04d}"
        row_path = rows_dir / f"{row_id}.json"
        raw_path = rows_dir / f"{row_id}_raw.json"

        if args.resume and row_path.exists():
            record = json_read(row_path)
            if "is_correct" in record:
                completed += 1
                correct += int(bool(record["is_correct"]))
            append_log(log_path, f"skip_existing {row_id}")
            continue

        problem_text = str(row[problem_col])
        expected = normalize_answer(row[answer_col]) if answer_col else None
        messages = build_messages(problem_text)

        try:
            append_log(log_path, f"calling_model {row_id}")
            response_payload = call_openrouter(
                api_key=api_key,
                model=args.model,
                messages=messages,
                reasoning_effort=args.reasoning_effort,
            )
            json_write(raw_path, response_payload)
            response_text = extract_text_from_response(response_payload)
            predicted = extract_final_answer(response_text)
            is_correct = None if expected is None else normalize_answer(predicted) == expected

            record = {
                "row_id": row_id,
                "model": args.model,
                "reasoning_effort": args.reasoning_effort,
                "problem": problem_text,
                "expected_answer": expected,
                "predicted_answer": predicted,
                "is_correct": is_correct,
                "response_text": response_text,
                "source_row": row.to_dict(),
            }
            json_write(row_path, record)
            completed += 1
            correct += int(bool(is_correct))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            error_record = {
                "row_id": row_id,
                "model": args.model,
                "reasoning_effort": args.reasoning_effort,
                "error": f"HTTP {exc.code}",
                "body": body,
            }
            json_write(row_path, error_record)
            append_log(log_path, f"http_error {row_id} code={exc.code}")
        except Exception as exc:
            error_record = {
                "row_id": row_id,
                "model": args.model,
                "reasoning_effort": args.reasoning_effort,
                "error": str(exc),
            }
            json_write(row_path, error_record)
            append_log(log_path, f"row_failed {row_id} {exc}")
            append_log(log_path, traceback.format_exc())

    accuracy = (correct / completed) if completed else 0.0
    summary = {
        "run_id": run_id,
        "dataset_uri": args.dataset_uri,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "completed_rows": completed,
        "correct_rows": correct,
        "accuracy": accuracy,
    }
    json_write(run_dir / "summary.json", summary)
    append_log(log_path, f"run_finished accuracy={accuracy:.4f}")

    print(f"Run ID: {run_id}")
    print(f"Run directory: {run_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
