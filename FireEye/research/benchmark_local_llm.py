#!/usr/bin/env python3
"""Benchmark local LLM (Ollama) vs cloud LLM (OpenRouter) for FireEye.

Runs risk classification on test images using both backends and compares
accuracy, speed, and cost.

Usage:
    python research/benchmark_local_llm.py
    python research/benchmark_local_llm.py --local-only
    python research/benchmark_local_llm.py --cloud-only
"""

import argparse
import base64
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

from openai import OpenAI

TEST_DATA = Path(__file__).resolve().parent.parent / "test_data"

EXPECTED = {
    "dangerous": {"medium", "high", "critical"},
    "safe": {"safe", "low"},
}

SYSTEM_PROMPT = """You are a fire safety expert analyzing construction site images.
Classify the risk level of the scene.
Respond ONLY with valid JSON (no markdown fences):
{"risk_level": "safe|low|medium|high|critical", "confidence": 0.0-1.0, "reason": "brief explanation"}"""


def classify_image(client: OpenAI, model: str, img_path: Path, detections_text: str) -> dict:
    """Classify a single image using the given client and model."""
    b64 = base64.b64encode(img_path.read_bytes()).decode()
    suffix = img_path.suffix.lower()
    mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"

    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": f"Analyze this image for fire safety risks. {detections_text}"},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
            ]}
        ],
        temperature=0.0,
    )
    elapsed = time.time() - t0
    content = resp.choices[0].message.content

    # Parse JSON from response (handle markdown fences)
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        result = {"risk_level": "error", "confidence": 0, "reason": f"Parse error: {text[:100]}"}

    result["elapsed_s"] = round(elapsed, 2)
    result["tokens"] = {
        "prompt": resp.usage.prompt_tokens if resp.usage else 0,
        "completion": resp.usage.completion_tokens if resp.usage else 0,
    }
    return result


def run_benchmark(client: OpenAI, model: str, label: str) -> dict:
    """Run the benchmark on all test images."""
    results = []
    total_time = 0

    for category in ("dangerous", "safe"):
        category_dir = TEST_DATA / category
        if not category_dir.exists():
            continue
        images = sorted(category_dir.glob("*.jpg")) + sorted(category_dir.glob("*.png"))
        for img_path in images:
            print(f"  [{label}] {category}/{img_path.name} ... ", end="", flush=True)
            try:
                result = classify_image(client, model, img_path, "")
                risk = result.get("risk_level", "error")
                passed = risk in EXPECTED[category]
                total_time += result["elapsed_s"]
                print(f"{risk} ({result['elapsed_s']:.1f}s) {'PASS' if passed else 'FAIL'}")
                results.append({
                    "file": img_path.name,
                    "category": category,
                    "risk_level": risk,
                    "confidence": result.get("confidence", 0),
                    "passed": passed,
                    "elapsed_s": result["elapsed_s"],
                    "prompt_tokens": result["tokens"]["prompt"],
                    "completion_tokens": result["tokens"]["completion"],
                })
            except Exception as e:
                print(f"ERROR: {e}")
                results.append({
                    "file": img_path.name,
                    "category": category,
                    "risk_level": "error",
                    "confidence": 0,
                    "passed": False,
                    "elapsed_s": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                })

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    accuracy = passed / total if total > 0 else 0
    avg_time = total_time / total if total > 0 else 0
    total_prompt = sum(r["prompt_tokens"] for r in results)
    total_completion = sum(r["completion_tokens"] for r in results)

    return {
        "label": label,
        "model": model,
        "total": total,
        "passed": passed,
        "accuracy": round(accuracy, 3),
        "avg_time_s": round(avg_time, 2),
        "total_time_s": round(total_time, 2),
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "results": results,
    }


def print_comparison(benchmarks: list[dict]) -> None:
    """Print a side-by-side comparison."""
    print("\n" + "=" * 70)
    print("BENCHMARK COMPARISON")
    print("=" * 70)

    for b in benchmarks:
        print(f"\n  {b['label']} ({b['model']})")
        print(f"    Accuracy:    {b['accuracy']:.1%} ({b['passed']}/{b['total']})")
        print(f"    Avg time:    {b['avg_time_s']:.2f}s per image")
        print(f"    Total time:  {b['total_time_s']:.1f}s")
        print(f"    Tokens:      {b['total_prompt_tokens']} prompt + {b['total_completion_tokens']} completion")

    # Per-image comparison if 2 benchmarks
    if len(benchmarks) == 2:
        a, b = benchmarks
        print(f"\n  Per-image comparison ({a['label']} vs {b['label']}):")
        print(f"  {'Image':<35} {'Cloud':>12} {'Local':>12} {'Match':>8}")
        print(f"  {'-'*67}")
        matches = 0
        for ra, rb in zip(a["results"], b["results"]):
            match = ra["risk_level"] == rb["risk_level"]
            if match:
                matches += 1
            print(f"  {ra['file']:<35} {ra['risk_level']:>12} {rb['risk_level']:>12} {'YES' if match else 'NO':>8}")
        print(f"\n  Agreement: {matches}/{len(a['results'])} ({matches/len(a['results']):.0%})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--cloud-only", action="store_true")
    parser.add_argument("--local-model", default="qwen2.5vl:7b")
    parser.add_argument("--local-url", default="http://localhost:11434")
    args = parser.parse_args()

    benchmarks = []

    if not args.local_only:
        from app.config import settings
        if settings.openrouter_api_key:
            cloud_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=settings.openrouter_api_key,
                timeout=120.0,
            )
            print("\n--- Cloud (OpenRouter) ---")
            benchmarks.append(run_benchmark(cloud_client, settings.llm_model, "Cloud"))
        else:
            print("Skipping cloud: no FIREEYE_OPENROUTER_API_KEY set")

    if not args.cloud_only:
        local_client = OpenAI(
            base_url=f"{args.local_url.rstrip('/')}/v1",
            api_key="ollama",
            timeout=300.0,
        )
        print(f"\n--- Local (Ollama: {args.local_model}) ---")
        benchmarks.append(run_benchmark(local_client, args.local_model, "Local"))

    if benchmarks:
        print_comparison(benchmarks)

        # Save results
        out = Path(__file__).resolve().parent / "benchmark_results.json"
        with open(out, "w") as f:
            json.dump([{k: v for k, v in b.items() if k != "results"} for b in benchmarks], f, indent=2)
        print(f"\nSummary saved to {out}")


if __name__ == "__main__":
    main()
