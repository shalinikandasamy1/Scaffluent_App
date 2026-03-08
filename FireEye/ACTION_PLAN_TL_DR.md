I've reviewed the full FireEye codebase against the ground truth booklet and compiled an action plan (attached as PDF). Here's the TL;DR:

Two gaps, six phases.

*Gap 1 -- YOLO can't see what matters.* It's trained on COCO (chairs, dogs, trucks). It can't detect fire extinguishers, gas cylinders, scaffold nets, exit signs, or welding screening -- the things HK fire regulations actually check for.

*Gap 2 -- The LLM doesn't know HK rules.* Prompts talk generic fire physics. They don't know about the 6 m hot-work clearance rule, the 30 m / 80 m water-relay height thresholds, fire-retardant material standards, or what a compliant site looks like.

*What to do, in priority order:*

1. Paste HK regs into the LLM prompts + add compliance output (~1 day) -- High impact, instant quality boost, no model retraining
2. Train a fire-safety YOLO for extinguishers, gas cylinders, nets, signs + add spatial distance calc (2-4 wks) -- Highest impact, fixes root detection blind spot
3. Align risk levels to regulatory criteria + add a separate compliance score (2-3 days) -- High impact, makes output actionable for site managers
4. Build a test dataset from the booklet's checklist with real HK site photos (1-2 wks) -- High impact, without this we can't measure progress
5. Heuristic fallback, externalise prompts to YAML, audit logging (1-2 wks) -- Medium impact, production hardening
6. Video streams, multi-camera fusion, BIM integration (Future) -- Variable impact, research track

Start with Phase 1 today (prompt-only, no code beyond schema). Kick off dataset collection for Phases 2 & 4 in parallel.

Full details in the attached ACTION_PLAN.pdf.

---

## Progress Update (2026-03-08)

Overnight autonomous session completed 10 of ~14 actionable phases:

| Phase | Status |
|-------|--------|
| 1. HK regs in prompts + compliance output | **Done** (1.1, 1.2, 1.3) |
| 2. Custom YOLO + spatial reasoning | **Done** (12-class model, Run 5 mAP50=0.540, spatial.py) |
| 3. Risk level alignment + compliance score | **Done** (3.1, 3.2) |
| 4. Test dataset + evaluation metrics | **Partial** (12 real images, evaluate.py, 100% heuristic accuracy) |
| 5. Heuristic fallback, YAML prompts, audit logging | **Done** (5.1, 5.2, 5.3) |
| 6. Video/multi-camera/BIM | Not started (future research) |

Key deliverables:
- Fine-tuned `fireeye_yolo11n_v5.pt` (12 construction-site fire safety classes, mAP50=0.540)
- Externalized prompts in `prompts/*.yaml`
- Audit logging to `audit_logs/*.jsonl`
- Compliance score computed from regulatory flags
- Heuristic fallback classifier with HK regulatory rules
- `evaluate.py` with accuracy, false alarm rate, miss rate metrics
