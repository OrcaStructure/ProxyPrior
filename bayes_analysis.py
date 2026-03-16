#!/usr/bin/env python3
"""Bayesian diagnostics for news real/fake multi-turn experiment outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def json_read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def json_write(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def clamp01(x: float) -> float:
    if x <= 0.0:
        return 1e-6
    if x >= 1.0:
        return 1.0 - 1e-6
    return x


def logit(p: float) -> float:
    q = clamp01(float(p))
    return math.log(q / (1.0 - q))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def entropy_bernoulli(p: float) -> float:
    q = clamp01(float(p))
    return -(q * math.log(q) + (1.0 - q) * math.log(1.0 - q))


def mean(xs: list[float]) -> float | None:
    return (sum(xs) / len(xs)) if xs else None


def variance(xs: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def brier_score(labels: list[int], probs: list[float]) -> float | None:
    if not labels:
        return None
    return sum((float(p) - int(y)) ** 2 for p, y in zip(probs, labels)) / len(labels)


def median(xs: list[float]) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    n = len(ys)
    mid = n // 2
    if n % 2 == 1:
        return ys[mid]
    return 0.5 * (ys[mid - 1] + ys[mid])


def expected_calibration_error(labels: list[int], probs: list[float], bins: int = 10) -> float | None:
    if not labels or len(labels) != len(probs):
        return None
    n = len(labels)
    total = 0.0
    for b in range(bins):
        lo = b / bins
        hi = (b + 1) / bins
        idxs = []
        for i, p in enumerate(probs):
            if b < bins - 1:
                if lo <= p < hi:
                    idxs.append(i)
            else:
                if lo <= p <= hi:
                    idxs.append(i)
        if not idxs:
            continue
        conf = sum(probs[i] for i in idxs) / len(idxs)
        acc = sum(labels[i] for i in idxs) / len(idxs)
        total += (len(idxs) / n) * abs(acc - conf)
    return total


def roc_auc(labels: list[int], scores: list[float]) -> float | None:
    pairs = [(s, y) for s, y in zip(scores, labels) if y in (0, 1)]
    if not pairs:
        return None
    pos = sum(1 for _, y in pairs if y == 1)
    neg = sum(1 for _, y in pairs if y == 0)
    if pos == 0 or neg == 0:
        return None
    wins = 0.0
    pos_scores = [s for s, y in pairs if y == 1]
    neg_scores = [s for s, y in pairs if y == 0]
    for s_pos in pos_scores:
        for s_neg in neg_scores:
            if s_pos > s_neg:
                wins += 1.0
            elif s_pos == s_neg:
                wins += 0.5
    return wins / (pos * neg)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze Bayesian update behavior in a news run directory.")
    p.add_argument(
        "--run-dir",
        required=True,
        help="Path to news_runs/<run_id>",
    )
    p.add_argument(
        "--reference-prior",
        type=float,
        default=0.5,
        help="Prior used to infer first-turn evidence from observed p1",
    )
    p.add_argument(
        "--prior-grid",
        default="0.2,0.5,0.8",
        help="Comma-separated priors for sensitivity analysis",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for classifying as real",
    )
    p.add_argument(
        "--cost-fp",
        type=float,
        default=1.0,
        help="Cost of false positive (predict real when fake)",
    )
    p.add_argument(
        "--cost-fn",
        type=float,
        default=1.0,
        help="Cost of false negative (predict fake when real)",
    )
    return p.parse_args()


def load_conversations(run_dir: Path) -> list[dict]:
    conv_dir = run_dir / "conversations"
    if not conv_dir.exists():
        raise ValueError(f"Missing conversations directory: {conv_dir}")

    conversations: list[dict] = []
    for path in sorted(conv_dir.glob("*.json")):
        data = json_read(path)
        turns = data.get("turns", [])
        if not turns:
            continue
        cleaned_turns = []
        for turn in turns:
            if "prob_real" not in turn or "label" not in turn:
                continue
            cleaned_turns.append(
                {
                    "conversation_id": str(turn.get("conversation_id", data.get("conversation_id", ""))),
                    "item_id": str(turn.get("item_id", "")),
                    "label": str(turn.get("label", "")).lower(),
                    "turn_index": int(turn.get("turn_index", 0)),
                    "num_turns": int(turn.get("num_turns", 0)),
                    "prob_real": float(turn["prob_real"]),
                }
            )
        if cleaned_turns:
            cleaned_turns.sort(key=lambda t: t["turn_index"])
            conversations.append({"conversation_id": data.get("conversation_id"), "turns": cleaned_turns})
    return conversations


def add_bayes_fields(conversations: list[dict], reference_prior: float) -> list[dict]:
    enriched: list[dict] = []
    prior_logit = logit(reference_prior)
    for convo in conversations:
        turns = convo["turns"]
        prev_logit = prior_logit
        out_turns = []
        cumulative = 0.0
        for turn in turns:
            p = clamp01(turn["prob_real"])
            current_logit = logit(p)
            log_bf = current_logit - prev_logit
            cumulative += log_bf
            out_turns.append(
                {
                    **turn,
                    "log_odds": current_logit,
                    "log_bf": log_bf,
                    "bf": math.exp(log_bf),
                    "cumulative_log_bf": cumulative,
                }
            )
            prev_logit = current_logit
        enriched.append({"conversation_id": convo["conversation_id"], "turns": out_turns})
    return enriched


def flatten_turns(conversations: list[dict]) -> list[dict]:
    out: list[dict] = []
    for convo in conversations:
        out.extend(convo["turns"])
    return out


def calibration_by_turn(turns: list[dict]) -> dict:
    by_turn: dict[int, list[dict]] = {}
    for t in turns:
        by_turn.setdefault(int(t["turn_index"]), []).append(t)
    out: dict[str, dict] = {}
    for k in sorted(by_turn):
        rows = by_turn[k]
        labels = [1 if r["label"] == "real" else 0 for r in rows]
        probs = [float(r["prob_real"]) for r in rows]
        empirical_real_rate = mean([float(y) for y in labels])
        mean_prob = mean(probs)
        out[str(k)] = {
            "n": len(rows),
            "mean_prob": mean_prob,
            "empirical_real_rate": empirical_real_rate,
            "calibration_gap_mean_prob_minus_base_rate": (
                (mean_prob - empirical_real_rate)
                if mean_prob is not None and empirical_real_rate is not None
                else None
            ),
            "brier": brier_score(labels, probs),
            "roc_auc": roc_auc(labels, probs),
            "ece_10": expected_calibration_error(labels, probs, bins=10),
            "ece_5": expected_calibration_error(labels, probs, bins=5),
        }
    return out


def information_gain(conversations: list[dict], reference_prior: float) -> dict:
    prior_entropy = entropy_bernoulli(reference_prior)
    turn_drops: dict[int, list[float]] = {}
    for convo in conversations:
        prev_h = prior_entropy
        for t in convo["turns"]:
            h = entropy_bernoulli(float(t["prob_real"]))
            turn_drops.setdefault(int(t["turn_index"]), []).append(prev_h - h)
            prev_h = h
    return {str(k): {"mean_entropy_drop": mean(v), "n": len(v)} for k, v in sorted(turn_drops.items())}


def cumulative_separation(conversations: list[dict]) -> dict:
    finals = [c["turns"][-1] for c in conversations if c["turns"]]
    real = [float(t["cumulative_log_bf"]) for t in finals if t["label"] == "real"]
    fake = [float(t["cumulative_log_bf"]) for t in finals if t["label"] == "fake"]
    labels = [1 if t["label"] == "real" else 0 for t in finals]
    scores = [float(t["cumulative_log_bf"]) for t in finals]
    return {
        "n_final_conversations": len(finals),
        "mean_cumulative_log_bf_real": mean(real),
        "mean_cumulative_log_bf_fake": mean(fake),
        "delta_real_minus_fake": (mean(real) - mean(fake)) if real and fake else None,
        "final_auc": roc_auc(labels, scores),
    }


def position_effects(turns: list[dict]) -> dict:
    by_item: dict[str, dict[int, list[float]]] = {}
    for t in turns:
        item = t["item_id"]
        if not item:
            continue
        by_item.setdefault(item, {}).setdefault(int(t["turn_index"]), []).append(float(t["prob_real"]))

    item_effects: list[float] = []
    for _, per_pos in by_item.items():
        means = [mean(vals) for _, vals in sorted(per_pos.items()) if vals]
        means = [m for m in means if m is not None]
        if len(means) >= 2:
            item_effects.append(max(means) - min(means))

    by_position: dict[int, list[float]] = {}
    for t in turns:
        by_position.setdefault(int(t["turn_index"]), []).append(float(t["prob_real"]))
    position_means = {str(k): mean(v) for k, v in sorted(by_position.items())}

    return {
        "mean_prob_by_position": position_means,
        "mean_item_position_range": mean(item_effects),
        "item_position_range_variance": variance(item_effects),
    }


def path_dependence(conversations: list[dict]) -> dict:
    groups: dict[tuple[str, ...], list[dict]] = {}
    for convo in conversations:
        ids = tuple(sorted([t["item_id"] for t in convo["turns"] if t.get("item_id")]))
        if len(ids) < 3:
            continue
        groups.setdefault(ids, []).append(convo)

    comparable = [g for g in groups.values() if len(g) > 1]
    diffs: list[float] = []
    for group in comparable:
        finals = [float(c["turns"][-1]["prob_real"]) for c in group if c["turns"]]
        if len(finals) > 1:
            diffs.append(max(finals) - min(finals))
    return {
        "num_comparable_item_sets": len(comparable),
        "mean_final_prob_range_within_same_set": mean(diffs),
        "max_final_prob_range_within_same_set": max(diffs) if diffs else None,
    }


def prior_sensitivity(conversations: list[dict], priors: list[float], reference_prior: float) -> dict:
    out: dict[str, dict] = {}
    for prior in priors:
        preds: list[dict] = []
        base = logit(prior)
        for convo in conversations:
            cumulative = 0.0
            for turn in convo["turns"]:
                cumulative += float(turn["log_bf"])
                p = sigmoid(base + cumulative)
                y = 1 if turn["label"] == "real" else 0
                preds.append({"p": p, "y": y})
        labels = [r["y"] for r in preds]
        probs = [r["p"] for r in preds]
        out[str(prior)] = {
            "n": len(preds),
            "mean_prob": mean(probs),
            "brier": brier_score(labels, probs),
            "auc": roc_auc(labels, probs),
            "reference_prior_for_evidence": reference_prior,
        }
    return out


def posterior_odds_accuracy(turns: list[dict]) -> dict:
    finals = [t for t in turns if int(t["turn_index"]) == int(t["num_turns"])]
    if not finals:
        return {"n": 0}
    mean_p = mean([float(t["prob_real"]) for t in finals]) or 0.5
    empirical = mean([1.0 if t["label"] == "real" else 0.0 for t in finals]) or 0.5
    mean_odds = mean_p / (1.0 - mean_p) if mean_p not in (0.0, 1.0) else None
    emp_odds = empirical / (1.0 - empirical) if empirical not in (0.0, 1.0) else None
    return {
        "n": len(finals),
        "mean_final_prob_real": mean_p,
        "empirical_real_rate": empirical,
        "mean_final_odds_real": mean_odds,
        "empirical_odds_real": emp_odds,
        "odds_ratio_model_over_empirical": (mean_odds / emp_odds) if (mean_odds and emp_odds) else None,
    }


def decision_utility(turns: list[dict], threshold: float, cost_fp: float, cost_fn: float) -> dict:
    labels = [1 if t["label"] == "real" else 0 for t in turns]
    probs = [float(t["prob_real"]) for t in turns]
    preds = [1 if p >= threshold else 0 for p in probs]
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    total_cost = cost_fp * fp + cost_fn * fn
    avg_cost = total_cost / len(turns) if turns else None
    return {
        "threshold": threshold,
        "cost_fp": cost_fp,
        "cost_fn": cost_fn,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total_cost": total_cost,
        "average_cost_per_decision": avg_cost,
    }


def evidence_by_turn(turns: list[dict]) -> dict:
    by_turn: dict[int, list[float]] = {}
    for t in turns:
        by_turn.setdefault(int(t["turn_index"]), []).append(float(t["log_bf"]))
    return {
        str(k): {"n": len(v), "mean_log_bf": mean(v), "mean_bf": mean([math.exp(x) for x in v])}
        for k, v in sorted(by_turn.items())
    }


def hierarchical_decomp(turns: list[dict]) -> dict:
    # Simple variance decomposition of prob_real means (not full Bayesian hierarchical model).
    probs = [float(t["prob_real"]) for t in turns]
    global_mean = mean(probs)
    if global_mean is None:
        return {}

    by_item: dict[str, list[float]] = {}
    by_turn: dict[str, list[float]] = {}
    by_convo: dict[str, list[float]] = {}
    for t in turns:
        by_item.setdefault(str(t.get("item_id", "")), []).append(float(t["prob_real"]))
        by_turn.setdefault(str(t.get("turn_index", "")), []).append(float(t["prob_real"]))
        by_convo.setdefault(str(t.get("conversation_id", "")), []).append(float(t["prob_real"]))

    item_means = [mean(v) for v in by_item.values() if v]
    turn_means = [mean(v) for v in by_turn.values() if v]
    convo_means = [mean(v) for v in by_convo.values() if v]

    return {
        "global_mean_prob_real": global_mean,
        "var_item_means": variance([x for x in item_means if x is not None]),
        "var_turn_means": variance([x for x in turn_means if x is not None]),
        "var_conversation_means": variance([x for x in convo_means if x is not None]),
    }


def update_dynamics(conversations: list[dict]) -> dict:
    updates: list[dict] = []
    for convo in conversations:
        turns = sorted(convo["turns"], key=lambda t: int(t["turn_index"]))
        for i in range(1, len(turns)):
            prev = turns[i - 1]
            cur = turns[i]
            delta_p = float(cur["prob_real"]) - float(prev["prob_real"])
            delta_log_odds = float(cur["log_odds"]) - float(prev["log_odds"])
            updates.append(
                {
                    "conversation_id": cur.get("conversation_id"),
                    "from_turn": int(prev["turn_index"]),
                    "to_turn": int(cur["turn_index"]),
                    "label": cur.get("label"),
                    "p_from": float(prev["prob_real"]),
                    "p_to": float(cur["prob_real"]),
                    "delta_p": delta_p,
                    "abs_delta_p": abs(delta_p),
                    "delta_log_odds": delta_log_odds,
                    "abs_delta_log_odds": abs(delta_log_odds),
                }
            )

    def summarize(rows: list[dict]) -> dict:
        if not rows:
            return {"n": 0}
        abs_dp = [float(r["abs_delta_p"]) for r in rows]
        abs_dlo = [float(r["abs_delta_log_odds"]) for r in rows]
        return {
            "n": len(rows),
            "mean_delta_p": mean([float(r["delta_p"]) for r in rows]),
            "mean_abs_delta_p": mean(abs_dp),
            "median_abs_delta_p": median(abs_dp),
            "mean_delta_log_odds": mean([float(r["delta_log_odds"]) for r in rows]),
            "mean_abs_delta_log_odds": mean(abs_dlo),
            "median_abs_delta_log_odds": median(abs_dlo),
        }

    def summarize_by_label(rows: list[dict]) -> dict:
        return {
            "all": summarize(rows),
            "real": summarize([r for r in rows if str(r.get("label", "")).lower() == "real"]),
            "fake": summarize([r for r in rows if str(r.get("label", "")).lower() == "fake"]),
        }

    biggest = max(updates, key=lambda r: float(r["abs_delta_log_odds"])) if updates else None
    updates_12 = [u for u in updates if u["from_turn"] == 1 and u["to_turn"] == 2]
    updates_23 = [u for u in updates if u["from_turn"] == 2 and u["to_turn"] == 3]
    return {
        "counts": {"n_updates": len(updates)},
        "avg_update_size": {
            "turn1_to_turn2": summarize_by_label(updates_12),
            "turn2_to_turn3": summarize_by_label(updates_23),
            "all_updates": summarize_by_label(updates),
        },
        "largest_single_update": biggest,
    }


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"[error] run dir does not exist: {run_dir}")
        return 1

    priors = [float(x.strip()) for x in args.prior_grid.split(",") if x.strip()]
    conversations = load_conversations(run_dir)
    if not conversations:
        print("[error] no usable conversations found")
        return 1

    enriched = add_bayes_fields(conversations, reference_prior=float(args.reference_prior))
    turns = flatten_turns(enriched)

    labels = [1 if t["label"] == "real" else 0 for t in turns]
    probs = [float(t["prob_real"]) for t in turns]

    summary = {
        "run_dir": str(run_dir),
        "n_conversations": len(enriched),
        "n_turns": len(turns),
        "overall": {
            "real_rate": mean([float(y) for y in labels]),
            "mean_prob_real": mean(probs),
            "brier": brier_score(labels, probs),
            "auc": roc_auc(labels, probs),
        },
        "evidence_by_turn": evidence_by_turn(turns),
        "calibration_by_turn": calibration_by_turn(turns),
        "information_gain_by_turn": information_gain(enriched, reference_prior=float(args.reference_prior)),
        "cumulative_evidence_separation": cumulative_separation(enriched),
        "update_dynamics": update_dynamics(enriched),
        "position_effects": position_effects(turns),
        "path_dependence": path_dependence(enriched),
        "prior_sensitivity": prior_sensitivity(
            enriched,
            priors=priors,
            reference_prior=float(args.reference_prior),
        ),
        "posterior_odds_accuracy": posterior_odds_accuracy(turns),
        "decision_utility": decision_utility(
            turns,
            threshold=float(args.threshold),
            cost_fp=float(args.cost_fp),
            cost_fn=float(args.cost_fn),
        ),
        "hierarchical_decomposition": hierarchical_decomp(turns),
    }

    out_path = run_dir / "bayes_analysis_summary.json"
    json_write(out_path, summary)

    print(f"Wrote: {out_path}")
    print(json.dumps(summary["overall"], indent=2, ensure_ascii=True))
    print(json.dumps(summary["update_dynamics"]["avg_update_size"], indent=2, ensure_ascii=True))
    print(json.dumps(summary["update_dynamics"]["largest_single_update"], indent=2, ensure_ascii=True))
    print(json.dumps(summary["calibration_by_turn"], indent=2, ensure_ascii=True))
    print(json.dumps(summary["cumulative_evidence_separation"], indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
