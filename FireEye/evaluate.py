"""Evaluation metrics for FireEye pipeline.

Computes classification accuracy, confusion matrix, and per-category
statistics from end-to-end test results.

Usage:
    python evaluate.py                  # Run evaluation on test_data/
    python evaluate.py --json           # Output JSON metrics
    python evaluate.py --heuristic-only # Use heuristic classifier (no LLM)
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config import settings
from app.models.schemas import RiskLevel
from app.pipeline import orchestrator, risk_classifier, yolo_detector
from app.storage import image_store

TEST_DATA = Path(__file__).resolve().parent / "test_data"

# Expected risk levels by category
EXPECTED = {
    "dangerous": {RiskLevel.medium, RiskLevel.high, RiskLevel.critical},
    "safe": {RiskLevel.safe, RiskLevel.low},
}


def evaluate(heuristic_only: bool = False, output_json: bool = False):
    results = []
    detection_stats = defaultdict(int)
    timing = {"yolo": [], "classify": [], "total": []}

    for category in ("dangerous", "safe"):
        category_dir = TEST_DATA / category
        if not category_dir.exists():
            continue

        images = sorted(category_dir.glob("*.jpg")) + sorted(category_dir.glob("*.png"))

        for img_path in images:
            image_id = uuid4()
            image_store.store_input_image(image_id, img_path.name, img_path.read_bytes())
            stored_path = image_store.get_input_image_path(image_id)

            t0 = time.time()
            try:
                # Stage 1: YOLO
                t_yolo = time.time()
                detections = yolo_detector.detect(str(stored_path))
                timing["yolo"].append(time.time() - t_yolo)

                for d in detections:
                    detection_stats[d.label] += 1

                # Stage 2: Classification
                t_cls = time.time()
                if heuristic_only:
                    risk = risk_classifier.classify_from_detections(detections)
                else:
                    risk = risk_classifier.classify_with_llm(str(stored_path), detections)
                timing["classify"].append(time.time() - t_cls)

                timing["total"].append(time.time() - t0)

                passed = risk.risk_level in EXPECTED[category]
                results.append({
                    "file": img_path.name,
                    "category": category,
                    "risk_level": risk.risk_level.value,
                    "confidence": risk.confidence,
                    "passed": passed,
                    "num_detections": len(detections),
                })
            except Exception as e:
                print(f"  SKIP {img_path.name}: {e}", file=sys.stderr)
                results.append({
                    "file": img_path.name,
                    "category": category,
                    "risk_level": "error",
                    "confidence": 0,
                    "passed": False,
                    "num_detections": 0,
                })
            finally:
                image_store.cleanup_image(image_id)

    # Compute metrics
    total = len(results)
    if total == 0:
        print("No test images found.")
        return

    passed = sum(1 for r in results if r["passed"])
    accuracy = passed / total

    # Per-category metrics
    cat_metrics = {}
    for cat in ("dangerous", "safe"):
        cat_results = [r for r in results if r["category"] == cat]
        if cat_results:
            cat_passed = sum(1 for r in cat_results if r["passed"])
            cat_metrics[cat] = {
                "total": len(cat_results),
                "correct": cat_passed,
                "accuracy": round(cat_passed / len(cat_results), 3),
            }

    # Confusion-style: what risk levels were assigned to each category
    confusion = defaultdict(lambda: defaultdict(int))
    for r in results:
        confusion[r["category"]][r["risk_level"]] += 1

    # False alarm rate: safe images classified as high/critical
    safe_results = [r for r in results if r["category"] == "safe"]
    false_alarms = sum(1 for r in safe_results
                       if r["risk_level"] in ("high", "critical"))
    false_alarm_rate = false_alarms / len(safe_results) if safe_results else 0

    # Miss rate: dangerous images classified as safe/low
    dangerous_results = [r for r in results if r["category"] == "dangerous"]
    misses = sum(1 for r in dangerous_results
                 if r["risk_level"] in ("safe", "low"))
    miss_rate = misses / len(dangerous_results) if dangerous_results else 0

    # Average timing
    avg_timing = {k: round(sum(v) / len(v), 3) if v else 0
                  for k, v in timing.items()}

    if heuristic_only:
        llm_label = "heuristic"
    elif settings.llm_backend == "local":
        llm_label = f"local/{settings.local_llm_model or 'unknown'}"
    else:
        llm_label = settings.llm_model

    metrics = {
        "model": settings.yolo_model_name,
        "llm": llm_label,
        "llm_backend": settings.llm_backend if not heuristic_only else "none",
        "total_images": total,
        "accuracy": round(accuracy, 3),
        "false_alarm_rate": round(false_alarm_rate, 3),
        "miss_rate": round(miss_rate, 3),
        "per_category": cat_metrics,
        "confusion": dict(confusion),
        "detection_counts": dict(detection_stats),
        "avg_time_s": avg_timing,
    }

    if output_json:
        print(json.dumps(metrics, indent=2))
    else:
        print("=" * 60)
        print("FireEye Evaluation Report")
        print("=" * 60)
        print(f"  YOLO model  : {metrics['model']}")
        print(f"  Classifier  : {metrics['llm']} ({metrics['llm_backend']})")
        print(f"  Images      : {total}")
        print(f"  Accuracy    : {accuracy:.1%}")
        print(f"  False alarm : {false_alarm_rate:.1%} (safe→high/critical)")
        print(f"  Miss rate   : {miss_rate:.1%} (dangerous→safe/low)")
        print()
        print("Per-category:")
        for cat, m in cat_metrics.items():
            print(f"  {cat:>10}: {m['correct']}/{m['total']} ({m['accuracy']:.1%})")
        print()
        print("Risk level distribution:")
        for cat in ("dangerous", "safe"):
            if cat in confusion:
                levels = ", ".join(f"{k}={v}" for k, v in
                                  sorted(confusion[cat].items()))
                print(f"  {cat:>10}: {levels}")
        print()
        print("Detection counts:")
        for label, count in sorted(detection_stats.items(),
                                   key=lambda x: -x[1]):
            print(f"  {label:>20}: {count}")
        print()
        print(f"Avg timing: YOLO={avg_timing['yolo']:.3f}s, "
              f"classify={avg_timing['classify']:.3f}s, "
              f"total={avg_timing['total']:.3f}s")
        print()

        # Per-image results
        for r in results:
            mark = "PASS" if r["passed"] else "FAIL"
            print(f"  [{mark}] {r['category']:>10} / {r['file']:<30} "
                  f"→ {r['risk_level']} ({r['confidence']:.2f})")

    return metrics


def save_results(metrics: dict):
    """Append evaluation results to a JSONL history file."""
    history_file = Path(__file__).resolve().parent / "eval_history.jsonl"
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **metrics,
    }
    with open(history_file, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"\nResults saved to {history_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FireEye evaluation")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--heuristic-only", action="store_true",
                        help="Use heuristic classifier only (no LLM)")
    parser.add_argument("--save", action="store_true",
                        help="Append results to eval_history.jsonl")
    parser.add_argument("--model", type=str, default=None,
                        help="Override YOLO model path (e.g. models/fireeye_yolo11n_v5.pt)")
    args = parser.parse_args()

    if args.model:
        # Override the model path before evaluation
        settings.yolo_model_name = args.model
        # Force model reload
        yolo_detector._model = None

    metrics = evaluate(heuristic_only=args.heuristic_only, output_json=args.json)
    if args.save and metrics:
        save_results(metrics)
