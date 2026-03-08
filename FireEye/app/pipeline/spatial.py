"""Spatial reasoning utilities for detection post-processing.

Computes geometric relationships between detected objects so the LLM
receives structured distance data rather than raw bounding boxes.
"""

from __future__ import annotations

from app.models.schemas import Detection


# Pairs where proximity matters for fire safety
_PROXIMITY_PAIRS = {
    # (ignition source, flammable) — order doesn't matter, we check both
    frozenset({"fire", "scaffold_net"}),
    frozenset({"fire", "tarpaulin"}),
    frozenset({"fire", "gas_cylinder"}),
    frozenset({"welding_sparks", "scaffold_net"}),
    frozenset({"welding_sparks", "tarpaulin"}),
    frozenset({"welding_sparks", "gas_cylinder"}),
    frozenset({"fire", "person"}),
    frozenset({"smoke", "person"}),
}


def bbox_center(det: Detection) -> tuple[float, float]:
    """Return the center point of a detection's bounding box."""
    cx = (det.bbox.x1 + det.bbox.x2) / 2
    cy = (det.bbox.y1 + det.bbox.y2) / 2
    return cx, cy


def bbox_area(det: Detection) -> float:
    """Return the area of a detection's bounding box in pixels."""
    return abs(det.bbox.x2 - det.bbox.x1) * abs(det.bbox.y2 - det.bbox.y1)


def compute_distances(detections: list[Detection]) -> list[dict]:
    """Compute center-to-center distances between all detection pairs.

    Returns a list of dicts with obj_a, obj_b, distance_px, and a flag
    indicating whether this pair is a safety-relevant proximity concern.
    """
    results = []
    for i, d1 in enumerate(detections):
        for j, d2 in enumerate(detections):
            if i >= j:
                continue
            cx1, cy1 = bbox_center(d1)
            cx2, cy2 = bbox_center(d2)
            dist_px = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5

            pair_key = frozenset({d1.label, d2.label})
            is_concern = pair_key in _PROXIMITY_PAIRS

            results.append({
                "obj_a": f"{d1.label} ({d1.confidence:.0%})",
                "obj_b": f"{d2.label} ({d2.confidence:.0%})",
                "distance_px": round(dist_px, 1),
                "safety_concern": is_concern,
            })

    # Sort: safety concerns first, then by distance
    results.sort(key=lambda r: (not r["safety_concern"], r["distance_px"]))
    return results


def estimate_scale(detections: list[Detection], image_height: int = 640) -> float | None:
    """Estimate a rough px-to-metre ratio using reference objects.

    Returns pixels per metre, or None if no reference objects found.
    Uses known real-world sizes:
      - hard_hat: ~0.3m tall
      - gas_cylinder: ~1.2m tall
      - person: ~1.7m tall
    """
    reference_sizes = {
        "hard_hat": 0.3,
        "gas_cylinder": 1.2,
        "person": 1.7,
    }

    estimates = []
    for det in detections:
        if det.label in reference_sizes:
            bbox_height = abs(det.bbox.y2 - det.bbox.y1)
            if bbox_height > 10:  # filter tiny detections
                px_per_m = bbox_height / reference_sizes[det.label]
                estimates.append(px_per_m)

    if not estimates:
        return None

    # Return median estimate for robustness
    estimates.sort()
    mid = len(estimates) // 2
    return estimates[mid]


def format_spatial_summary(
    detections: list[Detection],
    image_width: int = 640,
    image_height: int = 640,
) -> str:
    """Generate a human-readable spatial summary for LLM consumption."""
    if not detections:
        return "No objects detected."

    distances = compute_distances(detections)
    scale = estimate_scale(detections, image_height)

    lines = []

    # Safety-relevant proximity pairs
    concerns = [d for d in distances if d["safety_concern"]]
    if concerns:
        lines.append("PROXIMITY CONCERNS:")
        for c in concerns[:10]:  # limit to top 10
            dist_str = f"{c['distance_px']:.0f}px"
            if scale:
                metres = c["distance_px"] / scale
                dist_str += f" (~{metres:.1f}m)"
            lines.append(f"  {c['obj_a']} <-> {c['obj_b']}: {dist_str}")

    # Scale info
    if scale:
        lines.append(f"\nEstimated scale: {scale:.0f} px/m (from reference objects)")

    # Object counts by class
    label_counts: dict[str, int] = {}
    for d in detections:
        label_counts[d.label] = label_counts.get(d.label, 0) + 1
    lines.append(f"\nObject summary: {', '.join(f'{v}x {k}' for k, v in sorted(label_counts.items()))}")

    return "\n".join(lines)
