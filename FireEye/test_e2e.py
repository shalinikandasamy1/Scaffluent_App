"""End-to-end test for the FireEye pipeline.

Runs the full pipeline (YOLO → LLM risk classification → LLM agents)
on every image in test_data/{dangerous,safe}/ and prints results.
"""

import json
import sys
import time
from pathlib import Path
from uuid import uuid4

# Ensure the app package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config import settings
from app.models.schemas import RiskLevel
from app.pipeline import orchestrator
from app.storage import image_store

TEST_DATA = Path(__file__).resolve().parent / "test_data"

EXPECTED = {
    "dangerous": {RiskLevel.medium, RiskLevel.high, RiskLevel.critical},
    "safe": {RiskLevel.safe, RiskLevel.low},
}


def run_test():
    print("=" * 70)
    print("FireEye End-to-End Test")
    print("=" * 70)
    print(f"LLM model : {settings.llm_model}")
    print(f"YOLO model: {settings.yolo_model_name}")
    print(f"API key   : {'set' if settings.openrouter_api_key else 'MISSING'}")
    print()

    if not settings.openrouter_api_key:
        print("ERROR: FIREEYE_OPENROUTER_API_KEY not set in .env")
        sys.exit(1)

    results = []

    for category in ("dangerous", "safe"):
        category_dir = TEST_DATA / category
        if not category_dir.exists():
            print(f"SKIP: {category_dir} not found")
            continue

        images = sorted(category_dir.glob("*.jpg")) + sorted(category_dir.glob("*.png"))
        print(f"\n--- {category.upper()} ({len(images)} images) ---\n")

        for img_path in images:
            image_id = uuid4()
            # Copy image into the storage system
            image_store.store_input_image(image_id, img_path.name, img_path.read_bytes())

            print(f"  [{img_path.name}]")
            t0 = time.time()
            try:
                result = orchestrator.analyze_image(image_id)
                elapsed = time.time() - t0

                risk = result.risk_classification
                print(f"    YOLO detections : {len(result.detections)}")
                if result.detections:
                    top = result.detections[:5]
                    labels = ", ".join(f"{d.label}({d.confidence:.0%})" for d in top)
                    print(f"    Top detections  : {labels}")

                print(f"    Risk level      : {risk.risk_level.value} (conf={risk.confidence:.2f})")
                print(f"    Risk reason     : {risk.reason[:120]}")

                if result.present_assessment:
                    print(f"    Present summary : {result.present_assessment.summary[:120]}")
                    score = result.present_assessment.compliance_score
                    print(f"    Compliance score: {score:.0%} ({len(result.present_assessment.compliance_flags)} items)")
                    if result.present_assessment.compliance_flags:
                        for flag in result.present_assessment.compliance_flags:
                            print(f"    Compliance      : [{flag.status:>7}] {flag.item} - {flag.note[:80]}")
                    issues = result.present_assessment.compliance_issues
                    if issues:
                        print(f"    Issues          : {len(issues)} non-compliant item(s)")
                if result.future_prediction:
                    n = len(result.future_prediction.scenarios)
                    print(f"    Future scenarios: {n}, overall={result.future_prediction.overall_risk.value}")
                    print(f"    Recommendation  : {result.future_prediction.recommendation[:120]}")

                # Check if classification matches expected category
                passed = risk.risk_level in EXPECTED[category]
                status = "PASS" if passed else "FAIL"
                print(f"    Result          : {status} (expected {category}, got {risk.risk_level.value})")
                print(f"    Time            : {elapsed:.1f}s")

                results.append({
                    "file": img_path.name,
                    "category": category,
                    "risk_level": risk.risk_level.value,
                    "passed": passed,
                })

            except Exception as e:
                print(f"    ERROR: {e}")
                results.append({
                    "file": img_path.name,
                    "category": category,
                    "risk_level": "error",
                    "passed": False,
                })

            # Clean up stored files
            image_store.cleanup_image(image_id)
            print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    print(f"  Total : {total}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print()
    for r in results:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"  [{mark}] {r['category']:>10} / {r['file']:<30} → {r['risk_level']}")
    print()
    if failed:
        print(f"  {failed} test(s) did not match expected category.")
    else:
        print("  All tests passed!")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_test())
