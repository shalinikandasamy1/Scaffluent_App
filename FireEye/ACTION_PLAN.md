# FireEye Enhancement Action Plan

> Compiled 2026-03-07 after cross-referencing the current codebase against the
> FYP Ground Truth booklet (HK construction-site fire safety regulations,
> FSD Circular Letter 2/2008, LegCo Q&A, good-practice infographics) and
> relevant academic/industry literature.

---

## Executive Summary

The current FireEye pipeline is a solid rapid-POC: YOLO11n detects generic objects,
then three LLM stages (risk classifier, present agent, future agent) reason about
fire-spread risk. However, the system was built without the domain-specific ground
truth that the booklet provides. The gap is twofold:

1. **Detection gap** -- YOLO11n is trained on COCO (80 generic classes). It cannot
   detect the construction-site-specific hazards the regulations care about (scaffold
   nets, fire extinguishers, exit signs, gas cylinders, hot-work screening, electrical
   panels, tarpaulins, etc.).

2. **Assessment gap** -- The LLM prompts reason about "fire spread physics" in
   general terms but are unaware of the specific HK regulatory framework (6m hot-work
   rule, 30m/80m water-relay thresholds, fire-retardant material standards, hot-work
   permit requirements, etc.).

Below is a prioritised action plan to close both gaps.

---

## Phase 1: Ground the System in Regulatory Knowledge (Low effort, high impact)

### 1.1 Inject HK Construction-Site Fire Safety Rules into LLM Prompts

**What:** Update the system prompts in `risk_classifier.py` and `llm_agents.py` to
include the key regulatory criteria from the ground truth booklet. The LLM already
receives the image and detections; giving it the right checklist will immediately
improve assessment quality.

**Specific rules to encode:**

| Rule | Source | How to encode |
|------|--------|---------------|
| Combustibles must be >= 6 m from hot work | FSD CL 2/2008 | Add to risk classifier & present agent prompts |
| Hot work requires a permit; area must be screened | FSD CL 2/2008 | Present agent should note presence/absence of screening |
| Fire extinguishers required on each floor & near each container | Construction Site (Safety) Regulations | Present agent checklist item |
| Emergency exits must be clear, marked, illuminated | Construction Site (Safety) Regulations | Present agent checklist item |
| Facade nets/tarpaulins must be fire-retardant (GB 5725-2009, BS 5867-2:2008 Type B, NFPA 701:2019) | Buildings Dept circular, APP-70 | Risk classifier escalation factor |
| Water relaying system required for buildings >30 m (mandatory closed-circuit >80 m) | FSD CL 2/2008 | Future agent: note if high-rise site lacks visible firefighting water supply |
| Dangerous goods (acetylene, oxygen) must be stored within exempted quantities | Construction Site (Safety) Regulations | Present agent: flag loose gas cylinders near ignition |
| Designated smoking areas required | Good practice guidelines | Present agent observation |

**Files to modify:**
- `app/pipeline/risk_classifier.py` -- add HK-specific calibration section to the system prompt
- `app/pipeline/llm_agents.py` -- add regulatory checklist to both present and future agent prompts

**Effort:** ~2 hours prompt engineering + testing.

### 1.2 Add a Regulatory Compliance Output Field

**What:** Extend `PresentAssessment` schema to include a `compliance_flags` field --
a structured list of regulatory items observed present/absent/unclear.

```python
class ComplianceFlag(BaseModel):
    item: str            # e.g. "fire_extinguisher_visible"
    status: str          # "present" | "absent" | "unclear"
    note: str            # e.g. "Red extinguisher visible near stairwell"

class PresentAssessment(BaseModel):
    summary: str
    hazards: list[str]
    distances: list[str]
    compliance_flags: list[ComplianceFlag]  # NEW
```

**Files to modify:** `app/models/schemas.py`, `app/pipeline/llm_agents.py`

**Effort:** ~1-2 hours.

### 1.3 Map the "Common Accidents" Checklist to Detection Targets

The ground truth booklet lists 7 common accident causes. Each maps to something
the system should try to detect or flag:

| Common Accident | Detection Target | Current Coverage |
|-----------------|-------------------|------------------|
| Sparks from hot works igniting combustibles | Welding/grinding + nearby wood/cardboard | Partial (YOLO may miss welding) |
| Electrical overload / short circuit | Exposed wiring, overloaded sockets | None |
| Improper flammable material storage | Gas cylinders, timber piles scattered | Partial |
| Unextinguished cigarette butts | Cigarette/smoking evidence | None |
| Lack of fire protection / maintenance | Missing extinguishers, blocked exits | None |
| Non-fire-retardant facade coverings | Scaffold net material type | None |
| Incomplete fire service installations | Missing hose reels, alarm panels | None |

This table should be encoded into the system documentation and used to prioritise
Phase 2 custom training.

---

## Phase 2: Upgrade Object Detection (Medium effort, high impact)

### 2.1 Train or Fine-tune a Fire-Safety-Specific YOLO Model

**Problem:** YOLO11n on COCO detects "person", "chair", "truck" etc. It does NOT
detect the objects that matter for construction-site fire safety: fire extinguishers,
gas cylinders, scaffold nets, exit signs, welding equipment, hose reels, smoke
detectors, electrical panels, tarpaulins, PPE (hard hats, hi-vis vests).

**Approach options (in order of recommendation):**

1. **Fine-tune YOLO11 on a domain-specific dataset.** Use a combination of:
   - [FASDD](https://essd.copernicus.org/preprints/essd-2023-73/) -- 120k+ fire/smoke images
   - [Roboflow fire datasets](https://universe.roboflow.com/search?q=class:fire) -- many community datasets with fire, smoke, extinguisher labels
   - [Construction-Hazard-Detection](https://github.com/yihong1120/Construction-Hazard-Detection) -- existing YOLO model for construction sites (helmets, vests, machinery)
   - Custom-labelled images from HK construction sites (team to collect & annotate)

2. **Use a larger YOLO variant** (yolo11s or yolo11m) to improve detection quality.
   The current nano model trades accuracy for speed; since the pipeline already
   takes 14-28s due to LLM calls, upgrading to a small/medium model adds negligible
   latency but meaningful detection quality.

3. **Multi-model ensemble**: Run both COCO-YOLO (for general objects) and a
   fire-safety-fine-tuned YOLO (for domain objects), then merge detections.

**Target classes for the custom model:**

| Priority | Class | Why |
|----------|-------|-----|
| P0 | fire, flame, smoke | Core fire detection |
| P0 | fire_extinguisher | Regulatory requirement (must be present) |
| P0 | gas_cylinder | Dangerous goods detection |
| P0 | welding_spark, grinding_spark | Hot work detection |
| P1 | scaffold_net, tarpaulin | Facade covering (fire-retardant check) |
| P1 | exit_sign, emergency_light | Escape route assessment |
| P1 | hose_reel, fire_alarm_panel | Fire service installations |
| P1 | hard_hat, safety_vest | PPE / site management quality indicator |
| P2 | electrical_panel, cable_tray | Electrical hazard detection |
| P2 | cigarette, ashtray | Smoking violation detection |
| P2 | no_smoking_sign, hot_work_permit | Compliance signage |

**Effort:** 2-4 weeks depending on dataset availability and annotation needs.

### 2.2 Add Spatial Reasoning Pre-processing

**What:** Before passing detections to the LLM, compute geometric relationships:

- Pairwise distances between all detected objects (in pixels and estimated metres)
- Overlap / containment checks (is the fire extinguisher behind an obstruction?)
- Proximity alerts (any flammable within 6m-equivalent of ignition source?)

**Implementation sketch:**

```python
def compute_distances(detections: list[Detection]) -> list[dict]:
    """Compute center-to-center distances between all detection pairs."""
    results = []
    for i, d1 in enumerate(detections):
        for j, d2 in enumerate(detections):
            if i >= j:
                continue
            cx1 = (d1.bbox.x1 + d1.bbox.x2) / 2
            cy1 = (d1.bbox.y1 + d1.bbox.y2) / 2
            cx2 = (d2.bbox.x1 + d2.bbox.x2) / 2
            cy2 = (d2.bbox.y1 + d2.bbox.y2) / 2
            dist_px = ((cx1-cx2)**2 + (cy1-cy2)**2) ** 0.5
            results.append({
                "obj_a": d1.label, "obj_b": d2.label,
                "distance_px": round(dist_px, 1),
            })
    return results
```

Pass these distances as structured data to the LLM alongside detections. This
removes the burden of spatial reasoning from the vision model.

**Files to modify:** new utility in `app/pipeline/`, update `orchestrator.py`

**Effort:** ~1 day.

### 2.3 Pixel-to-Metre Calibration

**Problem:** The 6m hot-work rule requires real-world distance estimation, but the
system only has pixel distances.

**Options:**
- **Reference object scaling**: If a hard hat (~30cm) or gas cylinder (~1.2m) is
  detected, use its pixel height to estimate a px/m ratio for the frame.
- **User-provided scale**: Allow the operator to input camera height / FOV when
  uploading, enabling approximate distance conversion.
- **Depth estimation model**: Use a monocular depth model (e.g., Depth Anything V2)
  to estimate relative depths, though this adds complexity.

**Recommendation:** Start with reference-object scaling (simplest), document the
uncertainty, and let the LLM reason about "approximately X metres" rather than
claiming precision.

**Effort:** ~2-3 days.

---

## Phase 3: Align Risk Levels with Regulatory Framework (Medium effort)

### 3.1 Redefine Risk Levels to Match Ground Truth Criteria

The ground truth booklet implies a binary controlled/uncontrolled framework. The
current 5-level scale (safe/low/medium/high/critical) is more granular but should
be explicitly mapped to the regulatory criteria:

| FireEye Level | Regulatory Mapping | Key Indicators |
|---------------|-------------------|----------------|
| **safe** | Fully compliant site, no ignition | No active flame, extinguishers present, exits clear, nets fire-retardant |
| **low** | Compliant site with controlled hot work | Hot work with permit, screening in place, extinguisher nearby, 6m clearance |
| **medium** | Minor compliance gaps OR elevated risk | Hot work without visible screening, OR combustibles within 6m but not adjacent, OR missing extinguisher on one floor |
| **high** | Significant compliance gaps OR active danger | Uncontrolled flame, gas cylinders near ignition, blocked exits, combustible facade nets near fire |
| **critical** | Active fire spread OR cascade risk | Fire spreading to facade nets, gas cylinder exposure, multiple simultaneous violations |

**Files to modify:** `app/pipeline/risk_classifier.py` (update prompt), documentation

### 3.2 Add a "Compliance Score" Parallel to Risk Level

Separate **fire spread risk** (physics-based, current system strength) from
**regulatory compliance** (checklist-based, ground truth booklet strength).

Output both:
```json
{
  "fire_spread_risk": "medium",
  "compliance_score": 0.4,
  "compliance_issues": [
    "No fire extinguisher visible",
    "Combustible materials within 6m of hot work",
    "Exit route not visible/potentially obstructed"
  ]
}
```

This dual output is more actionable for site managers: fire spread risk tells
them "how dangerous is this right now?", while compliance score tells them
"what are you doing wrong according to regulations?"

**Effort:** ~1-2 days (new schema + prompt updates).

---

## Phase 4: Testing & Validation Against Ground Truth (Ongoing)

### 4.1 Build a Ground-Truth-Aligned Test Dataset

**What:** Create or collect test images that specifically test the regulatory criteria:

| Test Case | Image Content | Expected Detection | Expected Risk |
|-----------|---------------|-------------------|---------------|
| GT-01 | Welder with screening, extinguisher nearby, 6m clearance | welding, screen, extinguisher | low |
| GT-02 | Welder WITHOUT screening, wood within 3m | welding, wood | high |
| GT-03 | Gas cylinders near open flame | gas_cylinder, flame | critical |
| GT-04 | Blocked emergency exit | exit_sign (obstructed) | high |
| GT-05 | Fire-retardant scaffold net (orange stripe) | scaffold_net | safe/low |
| GT-06 | Non-fire-retardant net near flame | scaffold_net, flame | critical |
| GT-07 | Fire extinguishers present, exits clear, no flame | extinguisher, exit_sign | safe |
| GT-08 | Overloaded electrical cables sparking | electrical_spark | high |
| GT-09 | Cigarette butts on ground near timber | cigarette, wood | medium |
| GT-10 | High-rise site with visible water relay system | water_relay, hose_reel | safe/low |

**Source images:** Team should photograph real HK construction sites (with
permission) or use stock construction site photos and manually annotate.

**Effort:** ~1-2 weeks for a meaningful dataset (50-100 images).

### 4.2 Quantitative Evaluation Metrics

Establish measurable benchmarks:

- **Detection mAP@0.5** for fire-safety-relevant classes (target: >0.6 for P0 classes)
- **Risk classification accuracy** against ground truth labels (target: >80% within 1 level)
- **Compliance flag accuracy** -- precision/recall for each compliance item
- **False alarm rate** -- percentage of safe scenes incorrectly classified as high/critical (target: <10%)
- **Miss rate** -- percentage of truly dangerous scenes classified as safe/low (target: <5%)

### 4.3 Use the "Good Practices" Infographic as Visual Ground Truth

The booklet's image9.png shows 6 illustrated panels of good vs. bad practices.
These are ideal visual references for what the system should detect:

1. Fire extinguisher in working order, accessible -- detect `fire_extinguisher`, flag `present`
2. Clear exit routes with directional signs -- detect `exit_sign`, assess route clearance
3. Hot work with screening and nearby extinguisher -- detect `welding`, `screen`, `extinguisher`
4. Gas cylinders properly stored with no-smoking sign -- detect `gas_cylinder`, `no_smoking_sign`
5. Fire drill / emergency procedures -- not directly detectable from single image
6. Electrical overloading (bad practice) -- detect `electrical_spark`, `cable_overload`

---

## Phase 5: System Hardening & Production Readiness

### 5.1 Heuristic Fallback Classifier

The existing `classify_from_detections()` heuristic is unused. Update it to
incorporate the regulatory rules as a deterministic fallback when the LLM is
unavailable or slow:

```python
def classify_from_detections_v2(detections, distances):
    # Gas cylinder within 3m of flame -> CRITICAL
    # Hot work without extinguisher nearby -> HIGH
    # Combustibles within 6m of ignition -> HIGH
    # Uncontrolled flame of any size -> MEDIUM minimum
    # All controls visible, no flame -> SAFE
    ...
```

### 5.2 Externalise Prompts and Thresholds

Move all LLM prompts to YAML/JSON config files to enable:
- Version tracking of prompt iterations
- A/B testing different prompt versions
- Non-developer team members editing prompts
- Regulatory updates without code changes

### 5.3 Add Audit Logging

For each analysis, record:
- LLM model version, temperature, token count, latency
- YOLO model version, confidence threshold
- Full prompt sent and response received
- Timestamp and image hash

This is essential for accountability in a safety-critical system.

### 5.4 Parallel Stage 3 Execution

Stages 3a (Present Agent) and 3b (Future Agent) are currently sequential but
independent until Future Agent needs Present Agent output. Consider running
3a first (it's faster), then 3b with 3a's output, or restructuring so they
can overlap.

---

## Phase 6: Future Research Directions

### 6.1 Video / Temporal Analysis

The ground truth implies ongoing monitoring. Extend to video streams:
- Process frames at intervals (e.g., 1 fps)
- Track changes over time (fire growing, people evacuating, extinguisher being used)
- Temporal risk escalation detection

### 6.2 Multi-Camera Fusion

Construction sites have multiple CCTV angles. Fuse detections across cameras
to build a site-wide risk map rather than per-image analysis.

### 6.3 Integration with Site Management Systems

Connect FireEye output to:
- Building Information Models (BIM) for spatial mapping
- Hot work permit databases (verify permit exists for detected welding)
- IoT sensors (smoke detectors, temperature sensors -- the "4S" from the booklet)

### 6.4 Registered Fire Engineer (RFE) Report Generation

The booklet references the proposed RFE scheme. FireEye could generate
structured reports aligned with RFE assessment formats, supporting (not
replacing) professional fire safety inspections.

---

## Priority Summary

| Phase | Effort | Impact | Recommendation |
|-------|--------|--------|----------------|
| 1.1 Inject regulations into prompts | 2 hours | High | Do immediately |
| 1.2 Compliance output field | 1-2 hours | High | Do immediately |
| 1.3 Map accident checklist | 1 hour | Medium | Do immediately |
| 2.1 Custom YOLO model | 2-4 weeks | Very High | Start dataset collection now |
| 2.2 Spatial reasoning | 1 day | High | Do this week |
| 2.3 Pixel-to-metre calibration | 2-3 days | Medium | Do after 2.2 |
| 3.1 Redefine risk levels | 1 day | High | Do with 1.1 |
| 3.2 Compliance score | 1-2 days | High | Do after 1.2 |
| 4.1 Ground truth test dataset | 1-2 weeks | Very High | Start collecting images now |
| 4.2 Evaluation metrics | 2-3 days | High | Define now, measure after 2.1 |
| 5.x System hardening | 1-2 weeks | Medium | After phases 1-3 |
| 6.x Future research | Ongoing | Variable | Plan after MVP validated |

---

## Key References

- [FSD Circular Letter No. 2/2008 (PDF)](https://www.hkfsd.gov.hk/eng/source/circular/2008_02.pdf) -- Primary regulatory instrument for fire protection at HK construction sites
- [LCQ8: Fire safety at construction sites (LegCo Q&A)](https://www.info.gov.hk/gia/general/202303/29/P2023032900359.htm) -- Government response on enforcement and statistics
- [Fire Protection Notice No. 13 (PDF)](https://www.hkfsd.gov.hk/eng/source/notices/Fire_Protection_Notice_No_13.pdf) -- FSD recommendations for construction site fire protection
- [HK Buildings Dept circular on fire-retardant facade materials](https://www.bd.gov.hk/doc/en/resources/codes-and-references/practice-notes-and-circular-letters/circular/CL_UFRPNSTPSFBCDRMW2023e.pdf) -- Standards for scaffold nets/tarpaulins
- [FASDD: 100k+ fire/smoke detection dataset](https://essd.copernicus.org/preprints/essd-2023-73/) -- Largest open fire detection dataset
- [Construction-Hazard-Detection (GitHub)](https://github.com/yihong1120/Construction-Hazard-Detection) -- YOLO model for construction site hazards
- [Roboflow fire detection datasets](https://universe.roboflow.com/search?q=class:fire) -- Community fire detection datasets
- [DSS-YOLO: improved fire detection on YOLOv8](https://www.nature.com/articles/s41598-025-93278-w) -- State-of-art lightweight fire detection
- [Fire Safety of Wood Construction (USDA)](https://www.fpl.fs.usda.gov/documnts/fplgtr/fplgtr282/chapter_18_fpl_gtr282.pdf) -- Radiant heat ignition thresholds
- [Establishing safety distances for wildland fires](https://www.sciencedirect.com/science/article/abs/pii/S0379711208000039) -- Distance/heat flux research
