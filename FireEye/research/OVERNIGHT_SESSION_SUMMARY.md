# Overnight Session Summary (2026-03-08, 01:30–09:00 HKT)

## What happened

### Dataset & Model Training
- **Dataset v3**: 6,112 images (5,195 train / 917 val) from D-Fire, PPE, synthetic, and weak-class augmentation
- **NMS label cleanup**: Removed 2,078 duplicate bounding boxes (56-81% of weak class labels) — single biggest quality improvement
- **Run 4 (best)**: mAP50=0.536, mAP50-95=0.332 (80 epochs, AdamW, RTX 3060)
- **SGD comparison**: Tesla P4 SGD run got mAP50=0.422 — AdamW confirmed ~25% better
- **Model deployed**: `models/fireeye_yolo11n_v4.pt` (5.5MB), configured in `.env`

### Action Plan Implementation (10 of 14 phases)
| Done | Phase | Description |
|------|-------|-------------|
| Y | 1.1 | HK regulatory rules injected into all LLM prompts |
| Y | 1.2 | ComplianceFlag schema + compliance_flags field |
| Y | 1.3 | Common accidents mapped to YOLO detection targets |
| Y | 2.1 | Custom 12-class YOLO model trained |
| Y | 2.2 | Spatial reasoning (pairwise distances, scale estimation) |
| Y | 3.1 | Risk levels aligned with HK regulatory framework |
| Y | 3.2 | Compliance score (0-1) computed from flags |
| Y | 5.1 | Heuristic fallback classifier with gas-near-fire=CRITICAL |
| Y | 5.2 | Prompts externalized to `prompts/*.yaml` |
| Y | 5.3 | Audit logging to `audit_logs/*.jsonl` |
| N | 4.1 | Ground-truth test dataset (need more real images) |
| Y | 4.2 | Evaluation metrics script (`evaluate.py`) |

### Testing
- **38 unit tests** passing (heuristic, spatial, compliance, prompt loader, audit, API smoke)
- **22 real test images** (12 dangerous, 10 safe) + 1 edge case — 95.5% heuristic accuracy
- `evaluate.py` with accuracy, false alarm rate, miss rate, timing, `--save` history tracking

### Infrastructure
- GPU power restored to 170W on .153
- OpenRouter client: retry with exponential backoff, 120s timeout
- FastAPI analysis endpoint: runs in thread pool (non-blocking)
- YOLO auto-detects GPU

## Key files changed/created

```
FireEye/
├── app/
│   ├── pipeline/
│   │   ├── spatial.py          [NEW] Pairwise distances, scale estimation
│   │   ├── risk_classifier.py  [MOD] HK rules, COMMON_ACCIDENTS, YAML prompts
│   │   ├── llm_agents.py       [MOD] Regulatory checklist, YAML prompts
│   │   └── orchestrator.py     [MOD] Audit logging integration
│   ├── models/schemas.py       [MOD] ComplianceFlag, compliance_score, compliance_issues
│   ├── services/
│   │   ├── audit.py            [NEW] JSONL audit logging
│   │   ├── prompt_loader.py    [NEW] YAML prompt loading
│   │   └── openrouter_client.py [MOD] Retry + backoff + timeout
│   ├── config.py               [MOD] yolo_device config
│   └── routers/analysis.py     [MOD] asyncio.to_thread
├── prompts/
│   ├── risk_classifier.yaml    [NEW] Risk classifier prompts
│   ├── present_agent.yaml      [NEW] Present agent prompts
│   └── future_agent.yaml       [NEW] Future agent prompts
├── tests/
│   ├── test_heuristic.py       [NEW] 24 heuristic/spatial/compliance tests
│   ├── test_services.py        [NEW] 10 prompt loader + audit tests
│   ├── test_api.py             [NEW] 4 API smoke tests
│   └── conftest.py             [NEW] Shared fixtures
├── evaluate.py                 [NEW] Evaluation metrics with --save history
├── research/
│   ├── train_run5.py           [NEW] Run 5 config (imgsz=800, copy_paste=0.3)
│   └── DATASET_GENERATION_REPORT.md [MOD] Full overnight report
├── models/fireeye_yolo11n_v4.pt [NEW] Best model weights
└── .env                        [MOD] Fine-tuned model path + threshold
```

## Branch: `research/dataset-generation-overnight` (30+ commits)

## What to do next

1. **Merge to main** when ready: `git merge research/dataset-generation-overnight`
2. **Update NiceGUI dashboard** (on `feature/fireeye-nicegui-dashboard`) to show:
   - Compliance score gauge/badge
   - Compliance flags table
   - Audit log viewer
3. **Run 5 training**: In progress — imgsz=800, copy_paste=0.3, mixup=0.15. Compare with `python research/compare_per_class.py run4 run5`
4. **Collect more test images**: Real HK construction site photos for Phase 4.1
5. **Test with LLM**: Run `python evaluate.py` (needs API key) to test full pipeline accuracy
