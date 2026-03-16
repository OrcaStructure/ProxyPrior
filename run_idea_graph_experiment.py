#!/usr/bin/env python3
"""Build per-trace idea DAGs from proof traces and merge ideas across traces."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
import re
import traceback
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from openrouter_credits import start_credit_tracking

OPENROUTER_API = "https://openrouter.ai/api/v1/chat/completions"


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
        description="Extract idea graphs from math traces and relabel shared ideas across traces."
    )
    p.add_argument(
        "--run-dir",
        default=None,
        help="Path to math_trace_runs/<run_id>. If omitted, picks most recent run.",
    )
    p.add_argument(
        "--analysis-model",
        default="openai/gpt-5-mini",
        help="Model used for idea extraction and cross-trace merging",
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=8, help="Parallel workers for per-trace extraction")
    p.add_argument(
        "--max-traces",
        type=int,
        default=None,
        help="Optional limit for quick runs; default processes all traces",
    )
    p.add_argument("--resume", action="store_true", help="Reuse existing per-trace extraction files")
    return p.parse_args()


def detect_latest_run(base_dir: Path) -> Path:
    if not base_dir.exists():
        raise ValueError(f"Missing runs directory: {base_dir}")
    run_dirs = [p for p in base_dir.iterdir() if p.is_dir()]
    if not run_dirs:
        raise ValueError(f"No runs found in: {base_dir}")
    return max(run_dirs, key=lambda p: p.stat().st_mtime)


def call_openrouter(
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    app_title: str,
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
            "X-Title": app_title,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=240) as resp:
        parsed = json.loads(resp.read().decode("utf-8"))
    choices = parsed.get("choices", [])
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


def extract_json_object(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def extract_graph_prompt(problem_text: str, trace_text: str) -> list[dict[str, str]]:
    final_segment = extract_final_solution_segment(trace_text)
    system = (
        "You analyze a mathematical reasoning trace and extract an idea graph. "
        "Each idea must be atomic (a step, tactic, subgoal, transformation, or check). "
        "Return STRICT JSON with keys: ideas, edges. "
        "ideas: list of objects {id, text, appears_in_final_solution}. Use ids i1, i2, ... "
        "appears_in_final_solution is boolean and should be true only if that idea is explicitly present "
        "in the final proof section provided. "
        "edges: list of objects {source, target, relation} where source is a prerequisite for target. "
        "Only include edges between existing ids."
    )
    user = (
        f"Problem:\n{problem_text}\n\n"
        f"Reasoning trace:\n{trace_text}\n\n"
        f"Final proof section (for appearance marking):\n{final_segment}\n\n"
        "Extract the idea graph now."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def extract_final_solution_segment(trace_text: str) -> str:
    text = trace_text.strip()
    if not text:
        return ""
    markers = [
        r"(?im)^final\s+(proof|solution)\s*[:\-]?\s*$",
        r"(?im)^conclusion\s*[:\-]?\s*$",
        r"(?im)^therefore\b.*$",
    ]
    starts = []
    for pat in markers:
        for m in re.finditer(pat, text):
            starts.append(m.start())
    if starts:
        start = max(starts)
        segment = text[start:].strip()
        if segment:
            return segment
    # Fallback: use trailing chunk as approximate "final proof" section.
    max_chars = 1800
    return text[-max_chars:].strip()


def normalize_text_for_blocking(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\\s]", " ", text)
    words = [w for w in text.split() if len(w) >= 3]
    stop = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "then",
        "from",
        "show",
        "proof",
        "step",
        "have",
        "into",
        "using",
        "assume",
    }
    words = [w for w in words if w not in stop]
    words = sorted(set(words))
    return " ".join(words[:6])


def merge_prompt(candidates: list[dict[str, str]]) -> list[dict[str, str]]:
    lines = "\n".join([f"- {c['id']}: {c['text']}" for c in candidates])
    system = (
        "Group semantically equivalent mathematical idea statements. "
        "Return STRICT JSON with key clusters. "
        "clusters is a list of {canonical_label, member_ids}. "
        "Only group ideas that are clearly the same underlying idea."
    )
    user = f"Idea candidates:\n{lines}\n\nGroup equivalent ones."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def clean_graph(parsed: dict) -> dict:
    raw_ideas = parsed.get("ideas", [])
    raw_edges = parsed.get("edges", [])
    ideas: list[dict] = []
    id_set: set[str] = set()
    for i, obj in enumerate(raw_ideas, start=1):
        text = str(obj.get("text", "") if isinstance(obj, dict) else "").strip()
        if not text:
            continue
        idea_id = str(obj.get("id", f"i{i}")) if isinstance(obj, dict) else f"i{i}"
        if not re.fullmatch(r"i[0-9]+", idea_id):
            idea_id = f"i{i}"
        if idea_id in id_set:
            idea_id = f"i{i}"
        id_set.add(idea_id)
        appears = bool(obj.get("appears_in_final_solution", False)) if isinstance(obj, dict) else False
        ideas.append({"id": idea_id, "text": text, "appears_in_final_solution": appears})

    edges: list[dict] = []
    for obj in raw_edges:
        if not isinstance(obj, dict):
            continue
        src = str(obj.get("source", "")).strip()
        tgt = str(obj.get("target", "")).strip()
        rel = str(obj.get("relation", "prerequisite")).strip() or "prerequisite"
        if src in id_set and tgt in id_set and src != tgt:
            edges.append({"source": src, "target": tgt, "relation": rel})

    return {"ideas": ideas, "edges": edges}


def main() -> int:
    load_dotenv()
    args = parse_args()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("[error] Missing OPENROUTER_API_KEY in environment/.env")
        return 1

    run_dir = Path(args.run_dir) if args.run_dir else detect_latest_run(Path("math_trace_runs"))
    rows_dir = run_dir / "rows"
    task_path = run_dir / "task.json"
    if not run_dir.exists() or not rows_dir.exists():
        print(f"[error] Invalid run dir: {run_dir}")
        return 1
    if not task_path.exists():
        print(f"[error] Missing task file: {task_path}")
        return 1

    task = json_read(task_path)
    problem_text = str(task.get("problem_text", "")).strip()
    if not problem_text:
        print("[error] task.json missing problem_text")
        return 1

    out_dir = run_dir / "idea_graphs"
    traces_out_dir = out_dir / "per_trace"
    traces_out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "idea_graph.log"
    append_log(log_path, f"start run_dir={run_dir}")
    start_credit_tracking(
        api_key=api_key,
        log_fn=lambda msg: append_log(log_path, msg),
    )

    trace_paths = sorted([p for p in rows_dir.glob("sample_*.json") if not p.stem.endswith("_raw")])
    if args.max_traces is not None:
        trace_paths = trace_paths[: args.max_traces]
    if not trace_paths:
        print("[error] No trace rows found.")
        return 1

    pending: list[Path] = []
    loaded_graphs: list[dict] = []
    for trace_path in trace_paths:
        out_path = traces_out_dir / f"{trace_path.stem}_graph.json"
        if args.resume and out_path.exists():
            loaded_graphs.append(json_read(out_path))
            append_log(log_path, f"reuse {trace_path.name}")
            continue
        pending.append(trace_path)

    def extract_one(trace_path: Path) -> tuple[dict, str]:
        row = json_read(trace_path)
        trace_text = str(row.get("response_text", "")).strip()
        if not trace_text:
            raise ValueError("Missing response_text")
        messages = extract_graph_prompt(problem_text, trace_text)
        raw = call_openrouter(
            api_key=api_key,
            model=args.analysis_model,
            messages=messages,
            temperature=args.temperature,
            app_title="ProxyPrior Idea Graph Extractor",
        )
        parsed = extract_json_object(raw)
        graph = clean_graph(parsed)
        graph["trace_id"] = trace_path.stem
        graph["claimed_solved"] = row.get("claimed_solved")
        return graph, raw

    if pending:
        append_log(log_path, f"extract_parallel traces={len(pending)} workers={max(1, args.workers)}")
        print(
            f"[{dt.datetime.now().strftime('%H:%M:%S')}] Extracting per-trace idea graphs "
            f"for {len(pending)} traces..."
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            future_map = {executor.submit(extract_one, p): p for p in pending}
            done = 0
            total = len(future_map)
            for future in concurrent.futures.as_completed(future_map):
                trace_path = future_map[future]
                out_path = traces_out_dir / f"{trace_path.stem}_graph.json"
                raw_path = traces_out_dir / f"{trace_path.stem}_graph_raw.txt"
                try:
                    graph, raw = future.result()
                    json_write(out_path, graph)
                    raw_path.write_text(raw, encoding="utf-8")
                    loaded_graphs.append(graph)
                    append_log(log_path, f"done {trace_path.name}")
                except Exception as exc:
                    error = {"trace_id": trace_path.stem, "error": str(exc)}
                    json_write(out_path, error)
                    append_log(log_path, f"failed {trace_path.name} {exc}")
                    append_log(log_path, traceback.format_exc())
                done += 1
                print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] Extracted {done}/{total}")

    graphs = [g for g in loaded_graphs if "ideas" in g and isinstance(g["ideas"], list)]
    if not graphs:
        print("[error] No usable extracted graphs.")
        return 1

    # Flatten all ideas with unique global ids.
    idea_items: list[dict] = []
    local_to_global: dict[tuple[str, str], str] = {}
    idea_in_final: dict[tuple[str, str], bool] = {}
    for g in graphs:
        trace_id = g["trace_id"]
        for idx, idea in enumerate(g.get("ideas", []), start=1):
            local_id = str(idea.get("id", f"i{idx}"))
            text = str(idea.get("text", "")).strip()
            if not text:
                continue
            gid = f"{trace_id}:{local_id}"
            local_to_global[(trace_id, local_id)] = gid
            appears = bool(idea.get("appears_in_final_solution", False))
            idea_in_final[(trace_id, local_id)] = appears
            idea_items.append(
                {
                    "global_id": gid,
                    "trace_id": trace_id,
                    "local_id": local_id,
                    "text": text,
                    "appears_in_final_solution": appears,
                }
            )

    # Blocking pass for cross-trace merge candidates.
    blocks: dict[str, list[dict]] = defaultdict(list)
    for item in idea_items:
        key = normalize_text_for_blocking(item["text"])
        if not key:
            key = item["text"].lower()[:24]
        blocks[key].append(item)

    canonical_map: dict[str, str] = {}
    canonical_nodes: dict[str, dict] = {}
    canon_counter = 0

    def ensure_canon(label: str) -> str:
        nonlocal canon_counter
        canon_counter += 1
        cid = f"c{canon_counter:04d}"
        canonical_nodes[cid] = {"canonical_id": cid, "label": label.strip(), "members": []}
        return cid

    for key, items in blocks.items():
        if len(items) == 1:
            item = items[0]
            cid = ensure_canon(item["text"])
            canonical_map[item["global_id"]] = cid
            canonical_nodes[cid]["members"].append(item["global_id"])
            continue

        candidates = [{"id": i["global_id"], "text": i["text"]} for i in items]
        try:
            raw = call_openrouter(
                api_key=api_key,
                model=args.analysis_model,
                messages=merge_prompt(candidates),
                temperature=0.0,
                app_title="ProxyPrior Idea Graph Merger",
            )
            parsed = extract_json_object(raw)
            clusters = parsed.get("clusters", [])
            covered: set[str] = set()
            if isinstance(clusters, list):
                for cluster in clusters:
                    if not isinstance(cluster, dict):
                        continue
                    member_ids = [str(x) for x in cluster.get("member_ids", [])]
                    member_ids = [m for m in member_ids if any(c["id"] == m for c in candidates)]
                    if not member_ids:
                        continue
                    label = str(cluster.get("canonical_label", "")).strip() or items[0]["text"]
                    cid = ensure_canon(label)
                    for mid in member_ids:
                        if mid in covered:
                            continue
                        covered.add(mid)
                        canonical_map[mid] = cid
                        canonical_nodes[cid]["members"].append(mid)
            for item in items:
                gid = item["global_id"]
                if gid not in covered:
                    cid = ensure_canon(item["text"])
                    canonical_map[gid] = cid
                    canonical_nodes[cid]["members"].append(gid)
        except Exception:
            for item in items:
                cid = ensure_canon(item["text"])
                canonical_map[item["global_id"]] = cid
                canonical_nodes[cid]["members"].append(item["global_id"])

    # Relabel edges through canonical node ids and aggregate counts.
    edge_counts: dict[tuple[str, str], dict] = {}
    for g in graphs:
        trace_id = g["trace_id"]
        for edge in g.get("edges", []):
            src_local = str(edge.get("source", ""))
            tgt_local = str(edge.get("target", ""))
            src_gid = local_to_global.get((trace_id, src_local))
            tgt_gid = local_to_global.get((trace_id, tgt_local))
            if not src_gid or not tgt_gid:
                continue
            src_cid = canonical_map.get(src_gid)
            tgt_cid = canonical_map.get(tgt_gid)
            if not src_cid or not tgt_cid or src_cid == tgt_cid:
                continue
            src_in_final = bool(idea_in_final.get((trace_id, src_local), False))
            tgt_in_final = bool(idea_in_final.get((trace_id, tgt_local), False))
            edge_in_final = src_in_final and tgt_in_final
            key = (src_cid, tgt_cid)
            if key not in edge_counts:
                edge_counts[key] = {
                    "source_canonical_id": src_cid,
                    "target_canonical_id": tgt_cid,
                    "count": 0,
                    "final_solution_count": 0,
                    "trace_examples": [],
                }
            edge_counts[key]["count"] += 1
            if edge_in_final:
                edge_counts[key]["final_solution_count"] += 1
            if len(edge_counts[key]["trace_examples"]) < 5:
                edge_counts[key]["trace_examples"].append(trace_id)

    # Add per-canonical-node final-solution coverage.
    final_by_global = {item["global_id"]: bool(item.get("appears_in_final_solution", False)) for item in idea_items}
    for cid, node in canonical_nodes.items():
        members = list(node.get("members", []))
        present = sum(1 for gid in members if final_by_global.get(gid, False))
        total = len(members)
        node["final_solution_member_count"] = present
        node["final_solution_member_rate"] = (present / total) if total else 0.0

    canonical_graph = {
        "run_dir": str(run_dir),
        "analysis_model": args.analysis_model,
        "num_traces_processed": len(graphs),
        "num_ideas_total": len(idea_items),
        "num_canonical_ideas": len(canonical_nodes),
        "canonical_ideas": list(canonical_nodes.values()),
        "idea_mappings": [{"global_id": gid, "canonical_id": cid} for gid, cid in sorted(canonical_map.items())],
        "canonical_edges": sorted(edge_counts.values(), key=lambda e: (-e["count"], e["source_canonical_id"])),
    }
    json_write(out_dir / "canonical_idea_graph.json", canonical_graph)
    append_log(log_path, "done")

    print(f"Run dir: {run_dir}")
    print(f"Idea graph output: {out_dir / 'canonical_idea_graph.json'}")
    print(
        json.dumps(
            {
                "num_traces_processed": canonical_graph["num_traces_processed"],
                "num_ideas_total": canonical_graph["num_ideas_total"],
                "num_canonical_ideas": canonical_graph["num_canonical_ideas"],
                "num_canonical_edges": len(canonical_graph["canonical_edges"]),
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
