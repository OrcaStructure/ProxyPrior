#!/usr/bin/env python3
"""Wikipedia sentence substitution test using OpenRouter.

Flow:
1) Fetch a Wikipedia article as plain text.
2) Replace one real sentence with a plausible-but-wrong sentence.
3) Ask an OpenRouter model to identify the corrupted sentence.
4) Score whether the model identified the inserted sentence.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import re
import sys
import traceback
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from openrouter_credits import start_credit_tracking

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
OPENROUTER_API = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_REPLACEMENTS = [
    "By 2004, researchers briefly believed the city could be powered entirely by tidal clocks.",
    "A short-lived policy required all official maps to be drawn facing west instead of north.",
    "Several historians once argued that the project began as a theater experiment before becoming scientific.",
    "During early planning, administrators tested a rotating three-day week to reduce paperwork.",
]

STAGE_ORDER = [
    "INIT",
    "CASE_BUILT",
    "PROMPT_BUILT",
    "MODEL_CALLED",
    "EVALUATED",
    "DONE",
]


@dataclass
class TestCase:
    title: str
    original_sentence: str
    replacement_sentence: str
    corrupted_index: int
    corrupted_text: str


def sanitize_slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip()).strip("-").lower()
    return slug or "run"


def choose_run_id(title: str, seed: int) -> str:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{sanitize_slug(title)}_s{seed}"


def json_write(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def json_read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def append_log(path: Path, message: str) -> None:
    stamp = dt.datetime.now().isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{stamp}] {message}\n")


def set_stage(state_path: Path, log_path: Path, stage: str, extra: Optional[dict] = None) -> None:
    state = {"stage": stage, "updated_at": dt.datetime.now().isoformat(timespec="seconds")}
    if extra:
        state.update(extra)
    json_write(state_path, state)
    append_log(log_path, f"stage={stage}")


def stage_at_least(current_stage: str, target_stage: str) -> bool:
    return STAGE_ORDER.index(current_stage) >= STAGE_ORDER.index(target_stage)


def to_test_case(data: dict) -> TestCase:
    return TestCase(
        title=data["title"],
        original_sentence=data["original_sentence"],
        replacement_sentence=data["replacement_sentence"],
        corrupted_index=int(data["corrupted_index"]),
        corrupted_text=data["corrupted_text"],
    )


def fetch_wikipedia_extract(title: str) -> str:
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": "1",
        "titles": title,
        "redirects": "1",
    }
    url = f"{WIKIPEDIA_API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "ProxyPriorWikiTest/1.0"})

    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    pages = data.get("query", {}).get("pages", {})
    if not pages:
        raise ValueError("Wikipedia response did not include pages.")

    page = next(iter(pages.values()))
    extract = page.get("extract", "").strip()
    if not extract:
        raise ValueError(f"No extract found for title: {title}")
    return extract


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"\(])", text)
    return [p.strip() for p in parts if p.strip()]


def pick_sentence_index(sentences: list[str], rng: random.Random) -> int:
    candidates = [
        i
        for i, s in enumerate(sentences)
        if 60 <= len(s) <= 280 and "==" not in s and not s.lower().startswith("see also")
    ]
    if not candidates:
        raise ValueError("Could not find a suitable sentence to corrupt.")
    return rng.choice(candidates)


def build_test_case(
    title: str,
    article_text: str,
    replacement_sentence: Optional[str],
    seed: int,
    max_chars: int,
    forced_index: Optional[int] = None,
) -> TestCase:
    rng = random.Random(seed)
    text = article_text[:max_chars]
    sentences = split_sentences(text)
    if len(sentences) < 8:
        raise ValueError("Article extract is too short for a useful test.")

    idx = forced_index if forced_index is not None else pick_sentence_index(sentences, rng)
    if idx < 0 or idx >= len(sentences):
        raise ValueError("Forced sentence index is out of range.")
    original = sentences[idx]
    replacement = replacement_sentence or rng.choice(DEFAULT_REPLACEMENTS)
    sentences[idx] = replacement

    corrupted_text = " ".join(sentences)
    return TestCase(
        title=title,
        original_sentence=original,
        replacement_sentence=replacement,
        corrupted_index=idx,
        corrupted_text=corrupted_text,
    )


def build_prompt(test_case: TestCase) -> list[dict[str, str]]:
    system = (
        "You are a fact-checking assistant. A Wikipedia article has exactly one substituted sentence "
        "that is plausible but incorrect. Identify the substituted sentence. "
        "Return STRICT JSON with keys: corrupted_sentence, confidence, explanation."
    )

    user = (
        f"Wikipedia title: {test_case.title}\n\n"
        "Article text with one substitution:\n"
        f"{test_case.corrupted_text}\n\n"
        "Find the corrupted sentence."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_replacement_prompt(
    title: str,
    original_sentence: str,
    previous_sentence: str,
    next_sentence: str,
) -> list[dict[str, str]]:
    system = (
        "You rewrite one sentence from a Wikipedia article to be subtly wrong. "
        "It should be just a bit out of place, plausible, and close to believable. "
        "Keep roughly the same length and style as the original. "
        "Do not make it absurd or obviously fake. "
        "Return STRICT JSON with keys: replacement_sentence, rationale."
    )
    user = (
        f"Title: {title}\n"
        f"Previous sentence: {previous_sentence}\n"
        f"Original sentence: {original_sentence}\n"
        f"Next sentence: {next_sentence}\n\n"
        "Create exactly one replacement sentence."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def call_openrouter(
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    body = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        OPENROUTER_API,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost",
            "X-Title": "ProxyPrior Wiki Substitution Test",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=90) as resp:
        raw = resp.read().decode("utf-8")
        parsed = json.loads(raw)

    choices = parsed.get("choices", [])
    if not choices:
        raise ValueError("No choices returned by OpenRouter.")

    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        chunks = [c.get("text", "") for c in content if isinstance(c, dict)]
        text = "".join(chunks).strip()
        if text:
            return text

    raise ValueError("Could not parse model content from OpenRouter response.")


def extract_json_object(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def normalize_sentence(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return s


def evaluate(test_case: TestCase, guessed_corrupted_sentence: str) -> tuple[bool, float]:
    gold = normalize_sentence(test_case.replacement_sentence)
    pred = normalize_sentence(guessed_corrupted_sentence)

    if not pred:
        return False, 0.0
    if pred == gold:
        return True, 1.0

    gold_words = set(gold.split())
    pred_words = set(pred.split())
    if not gold_words or not pred_words:
        return False, 0.0

    overlap = len(gold_words & pred_words) / len(gold_words | pred_words)
    return overlap >= 0.7, overlap


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a Wikipedia substitution test against OpenRouter.")
    p.add_argument("--title", required=True, help="Wikipedia article title, e.g. 'Alan Turing'")
    p.add_argument("--model", required=True, help="OpenRouter model id")
    p.add_argument("--api-key", default=None, help="OpenRouter API key. Falls back to OPENROUTER_API_KEY env var")
    p.add_argument("--replacement", default=None, help="Custom strange-but-believable replacement sentence")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sentence selection")
    p.add_argument("--max-chars", type=int, default=7000, help="Max article chars sent to the model")
    p.add_argument(
        "--replacement-mode",
        choices=["llm", "template"],
        default="llm",
        help="How to create replacement sentence when --replacement is not provided",
    )
    p.add_argument(
        "--replacement-model",
        default=None,
        help="Model for generating replacement sentence (defaults to --model)",
    )
    p.add_argument(
        "--replacement-temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for replacement generation",
    )
    p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for the model")
    p.add_argument("--dry-run", action="store_true", help="Build test case but skip OpenRouter call")
    p.add_argument("--runs-dir", default="runs", help="Directory where run logs/artifacts are stored")
    p.add_argument("--run-id", default=None, help="Run identifier; reuse with --resume to continue")
    p.add_argument("--resume", action="store_true", help="Resume an existing run-id from saved state")
    return p.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")

    run_id = args.run_id or choose_run_id(args.title, args.seed)
    run_dir = Path(args.runs_dir) / run_id
    state_path = run_dir / "state.json"
    log_path = run_dir / "run.log"

    if run_dir.exists() and not args.resume and args.run_id:
        print(
            f"[error] Run directory already exists: {run_dir}. Use --resume to continue.",
            file=sys.stderr,
        )
        return 1

    run_dir.mkdir(parents=True, exist_ok=True)
    append_log(log_path, "run_started")
    json_write(run_dir / "args.json", vars(args))
    append_log(log_path, f"run_id={run_id}")
    start_credit_tracking(
        api_key=api_key,
        log_fn=lambda msg: append_log(log_path, msg),
    )
    print(f"Run ID: {run_id}")
    print(f"Run directory: {run_dir}")

    state = {"stage": "INIT"}
    if args.resume and state_path.exists():
        state = json_read(state_path)
        append_log(log_path, f"resuming_from_stage={state.get('stage', 'INIT')}")
    else:
        set_stage(state_path, log_path, "INIT")
    current_stage = state.get("stage", "INIT")

    test_case: Optional[TestCase] = None
    parsed: Optional[dict] = None

    if args.resume and state_path.exists() and stage_at_least(state.get("stage", "INIT"), "CASE_BUILT"):
        try:
            test_case = to_test_case(json_read(run_dir / "test_case.json"))
            append_log(log_path, "loaded_test_case_from_disk")
        except Exception:
            append_log(log_path, "failed_loading_saved_test_case; rebuilding")

    if test_case is None:
        try:
            append_log(log_path, f"fetching_wikipedia title={args.title}")
            article = fetch_wikipedia_extract(args.title)
            (run_dir / "article.txt").write_text(article, encoding="utf-8")
            base_sentences = split_sentences(article[: args.max_chars])
            if len(base_sentences) < 8:
                raise ValueError("Article extract is too short for a useful test.")

            target_idx = pick_sentence_index(base_sentences, random.Random(args.seed))
            original_sentence = base_sentences[target_idx]
            previous_sentence = base_sentences[target_idx - 1] if target_idx > 0 else ""
            next_sentence = base_sentences[target_idx + 1] if target_idx + 1 < len(base_sentences) else ""

            replacement_sentence = args.replacement
            if not replacement_sentence and args.replacement_mode == "template":
                replacement_sentence = random.Random(args.seed).choice(DEFAULT_REPLACEMENTS)
                append_log(log_path, "replacement_mode=template")

            if not replacement_sentence and args.replacement_mode == "llm":
                if not api_key:
                    raise ValueError(
                        "Missing API key for LLM replacement generation. "
                        "Set OPENROUTER_API_KEY, pass --api-key, or use --replacement-mode template."
                    )
                replacement_model = args.replacement_model or args.model
                replacement_messages = build_replacement_prompt(
                    title=args.title,
                    original_sentence=original_sentence,
                    previous_sentence=previous_sentence,
                    next_sentence=next_sentence,
                )
                json_write(run_dir / "replacement_prompt.json", {"messages": replacement_messages})
                append_log(log_path, f"generating_replacement model={replacement_model}")
                replacement_raw = call_openrouter(
                    api_key=api_key,
                    model=replacement_model,
                    messages=replacement_messages,
                    temperature=args.replacement_temperature,
                )
                (run_dir / "openrouter_replacement_raw.txt").write_text(replacement_raw, encoding="utf-8")
                replacement_json = extract_json_object(replacement_raw)
                json_write(run_dir / "replacement_output.json", replacement_json)
                replacement_sentence = str(replacement_json.get("replacement_sentence", "")).strip()
                if not replacement_sentence:
                    raise ValueError("Replacement model returned an empty replacement_sentence.")
                if normalize_sentence(replacement_sentence) == normalize_sentence(original_sentence):
                    raise ValueError("Replacement model returned the original sentence unchanged.")

            test_case = build_test_case(
                title=args.title,
                article_text=article,
                replacement_sentence=replacement_sentence,
                seed=args.seed,
                max_chars=args.max_chars,
                forced_index=target_idx,
            )
            json_write(run_dir / "test_case.json", asdict(test_case))
            set_stage(state_path, log_path, "CASE_BUILT")
            current_stage = "CASE_BUILT"
        except Exception as exc:
            append_log(log_path, "error_building_test_case")
            append_log(log_path, traceback.format_exc())
            print(f"[error] Failed to build test case: {exc}", file=sys.stderr)
            return 1

    print("=== Test Case ===")
    print(f"Title: {test_case.title}")
    print(f"Corrupted index: {test_case.corrupted_index}")
    print(f"Inserted sentence: {test_case.replacement_sentence}")
    print()

    if args.dry_run:
        print("Dry run enabled. Skipping OpenRouter call.")
        print("Original sentence was:")
        print(test_case.original_sentence)
        if not stage_at_least(current_stage, "DONE"):
            set_stage(state_path, log_path, "DONE")
            append_log(log_path, "run_finished")
        return 0

    if not api_key:
        print("[error] Missing API key. Set OPENROUTER_API_KEY or pass --api-key.", file=sys.stderr)
        return 1

    messages = build_prompt(test_case)
    json_write(run_dir / "prompt.json", {"messages": messages})
    if not stage_at_least(current_stage, "PROMPT_BUILT"):
        set_stage(state_path, log_path, "PROMPT_BUILT")
        current_stage = "PROMPT_BUILT"

    if args.resume and state_path.exists() and stage_at_least(current_stage, "MODEL_CALLED"):
        try:
            parsed = json_read(run_dir / "model_output.json")
            append_log(log_path, "loaded_model_output_from_disk")
        except Exception:
            append_log(log_path, "failed_loading_saved_model_output; recalling_model")

    if parsed is None:
        try:
            append_log(log_path, f"calling_openrouter model={args.model}")
            raw = call_openrouter(
                api_key=api_key,
                model=args.model,
                messages=messages,
                temperature=args.temperature,
            )
            (run_dir / "openrouter_raw.txt").write_text(raw, encoding="utf-8")
            parsed = extract_json_object(raw)
            json_write(run_dir / "model_output.json", parsed)
            set_stage(state_path, log_path, "MODEL_CALLED")
            current_stage = "MODEL_CALLED"
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            (run_dir / "openrouter_error.txt").write_text(body, encoding="utf-8")
            append_log(log_path, f"openrouter_http_error code={exc.code}")
            append_log(log_path, body)
            print(f"[error] OpenRouter HTTP {exc.code}: {body}", file=sys.stderr)
            return 1
        except Exception as exc:
            append_log(log_path, "openrouter_call_failed")
            append_log(log_path, traceback.format_exc())
            print(f"[error] OpenRouter call failed: {exc}", file=sys.stderr)
            return 1

    guessed_corrupted = str(parsed.get("corrupted_sentence", "")).strip()
    confidence = parsed.get("confidence", None)
    explanation = str(parsed.get("explanation", "")).strip()

    success, score = evaluate(test_case, guessed_corrupted)
    json_write(
        run_dir / "evaluation.json",
        {
            "success": success,
            "similarity_score": score,
            "model_confidence": confidence,
            "guessed_corrupted_sentence": guessed_corrupted,
            "ground_truth_corrupted_sentence": test_case.replacement_sentence,
            "explanation": explanation,
        },
    )
    if not stage_at_least(current_stage, "EVALUATED"):
        set_stage(state_path, log_path, "EVALUATED")
        current_stage = "EVALUATED"

    print("=== Model Output ===")
    print(json.dumps(parsed, indent=2, ensure_ascii=True))
    print()
    print("=== Evaluation ===")
    print(f"Success: {success}")
    print(f"Similarity score: {score:.3f}")
    print(f"Model confidence: {confidence}")
    print()
    print("Ground truth corrupted sentence:")
    print(test_case.replacement_sentence)
    print()
    print("Model guessed corrupted sentence:")
    print(guessed_corrupted)
    if explanation:
        print()
        print("Model explanation:")
        print(explanation)

    if not stage_at_least(current_stage, "DONE"):
        set_stage(state_path, log_path, "DONE")
        append_log(log_path, "run_finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
