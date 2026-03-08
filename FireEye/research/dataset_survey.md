# FireEye Dataset Survey

**Date:** 2026-03-08
**Purpose:** Identify public datasets for expanding FireEye's object detection training data.
**Target classes:** fire, smoke, fire extinguisher, gas cylinder, welding sparks, scaffold net, tarpaulin, exit sign, hose reel, electrical panel, hard hat, safety vest, cigarette.

---

## Summary

| # | Dataset | Images | Relevant Classes | Format | License | Relevance |
|---|---------|--------|-----------------|--------|---------|-----------|
| 1 | FASDD | ~120,000 | fire, smoke | COCO/YOLO/VOC/TFRecord | Academic (open-access) | 4/5 |
| 2 | D-Fire | 21,527 | fire, smoke | YOLO | CC0 (public domain) | 5/5 |
| 3 | SODA | 19,846 | helmet, vest, person, scaffold, fence, rebar, ebox | YOLO/COCO | Academic | 4/5 |
| 4 | SH17 | 8,099 | 17 PPE classes (hard hat, vest, goggles, gloves, etc.) | YOLO | CC BY-NC-SA 4.0 | 4/5 |
| 5 | CylinDeRS | 7,060 | gas cylinder (+ material/size/orientation attributes) | Roboflow (multi-format) | CC BY 4.0 | 5/5 |
| 6 | Construction-PPE (Ultralytics) | ~1,416 | helmet, vest, gloves, boots, goggles (+ missing variants) | YOLO | AGPL-3.0 | 3/5 |
| 7 | Roboflow Construction Site Safety | 717 | hard hat, safety vest, person | Multi-format | Roboflow (open source) | 3/5 |
| 8 | FireExtinguisher (Roboflow) | ~3,300 | fire extinguisher, fire blanket, fire exit, alarm activator, sounders | Multi-format | Roboflow (open source) | 5/5 |
| 9 | Fire Safety Equipment (Bayer et al.) | 841 | fire extinguisher, emergency call point, smoke detector, fire safety blanket | YOLO | Academic | 4/5 |
| 10 | DeepQuestAI Fire-Smoke-Dataset | ~3,000 | fire, smoke, neutral | Classification | MIT | 3/5 |
| 11 | FireNet Dataset | ~500+ (frames) | fire, non-fire | Classification | Open | 2/5 |
| 12 | FireRescue (UAV) | 15,980 | fire truck, firefighter, flames, smoke + 4 more | YOLO | Academic | 3/5 |
| 13 | MOCS | 41,668 | worker, tower crane, vehicles (13 classes) | COCO (bbox + seg masks) | Academic | 2/5 |
| 14 | Emergency Exit Signs (Roboflow) | 482 | left exit, right exit, straight exit, bidirectional exit | Multi-format | Roboflow (open source) | 4/5 |
| 15 | Smoker YOLO / Cigarette (Roboflow) | ~4,127 | cigarette, smoking person | Multi-format | Roboflow (open source) | 4/5 |
| 16 | Welding Sparks (Roboflow) | ~100-300 | welding spark | Multi-format | Roboflow (open source) | 3/5 |
| 17 | Construction-Hazard-Detection (yihong1120) | varies | hardhat, mask, no-hardhat, no-mask, no-safety-vest, person, safety cone, safety vest, machinery, vehicle | YOLO | Open source | 3/5 |
| 18 | Kaggle Smoke-Fire-Detection-YOLO | varies | fire, smoke | YOLO | Kaggle (open) | 3/5 |
| 19 | FASDD_CV COCO Split (Kaggle) | subset of FASDD | fire, smoke | COCO | Open | 4/5 |
| 20 | Pictor-PPE | ~1,472 | hard hat, safety vest | YOLO/TF | Academic | 2/5 |

---

## Detailed Dataset Profiles

### 1. FASDD (Flame And Smoke Detection Dataset)

- **URL:** https://www.scidb.cn/en/detail?dataSetId=ce9c9400b44148e1b0a749f5c3eb0bda
- **Paper:** https://www.tandfonline.com/doi/full/10.1080/10095020.2024.2347922
- **Kaggle subset:** https://www.kaggle.com/datasets/yuulind/fasdd-cv-coco
- **Size:** ~120,000 images across 3 sub-datasets (FASDD_CV, FASDD_UAV, FASDD_RS)
- **Classes:** fire, smoke
- **Format:** COCO, YOLO, VOC, TFRecord (4 annotation configs provided)
- **Platforms:** Ground-based cameras, UAV/drone, satellite/remote sensing
- **Variations:** Day/night, indoor/outdoor, near/far, multiple viewing angles
- **License:** Academic open-access
- **Relevance:** 4/5 -- Largest fire/smoke dataset available. The CV subset is most relevant for construction site scenarios. Remote sensing subsets less useful.
- **Notes:** Download via SciDB (Chinese academic data platform). FASDD_CV COCO split also available on Kaggle.

### 2. D-Fire

- **URL:** https://github.com/gaiasd/DFireDataset
- **Paper:** Venancio et al., "An automatic fire detection system based on deep convolutional neural networks," Neural Computing and Applications, 2022.
- **Size:** 21,527 images, 26,557 bounding boxes
- **Classes:** fire, smoke (image-level: fire, smoke, fire+smoke, none)
- **Format:** YOLO (normalized coordinates)
- **License:** CC0 (public domain) -- unrestricted use
- **Relevance:** 5/5 -- Large, well-annotated, YOLO-ready, permissive license. Ideal starting point for fire/smoke classes.
- **Notes:** Includes train/val/test split. Conversion utility provided (yolo2pixel).

### 3. SODA (Site Object Detection dAtaset)

- **URL:** https://linjiarui.net/en/publications/2022-07-24-soda-large-scale-open-site-object-detection-dataset-for-deep-learning-in-construction
- **Paper:** https://arxiv.org/abs/2202.09554
- **Size:** 19,846 images, 286,201 annotated objects
- **Classes (15):** helmet, vest, wood, board, person, fence, rebar, hook, brick, cutter, ebox (electrical box), handcart, hopper, scaffold, slogan
- **Format:** YOLO/COCO
- **License:** Academic (check paper for terms)
- **Relevance:** 4/5 -- Directly relevant for hard hat, safety vest, scaffold, and electrical box classes. The "ebox" class could serve as a proxy for electrical panel detection.
- **Notes:** Collected from real construction sites across different conditions and phases. Backup on Baidu Cloud.

### 4. SH17 (Safety and Human PPE Dataset)

- **URL (GitHub):** https://github.com/ahmadmughees/SH17dataset
- **URL (Kaggle):** https://www.kaggle.com/datasets/mugheesahmad/sh17-dataset-for-ppe-detection
- **URL (Zenodo):** https://zenodo.org/records/12659325
- **Size:** 8,099 images, 75,994 instances
- **Classes (17):** Includes hard hat, safety vest, goggles, gloves, ear protection, face shield, full-body harness, safety shoes, and corresponding "missing" variants
- **Format:** YOLO
- **License:** CC BY-NC-SA 4.0 (non-commercial)
- **Relevance:** 4/5 -- Excellent for hard hat and safety vest classes. Manufacturing/industrial environments similar to construction.
- **Notes:** Non-commercial license limits deployment options.

### 5. CylinDeRS (Gas Cylinder Detection)

- **URL:** https://universe.roboflow.com/klearchos-stavrothanasopoulos-konstantinos-gkountakos-6jwgj/cylinders-iaq6n
- **Paper:** https://www.mdpi.com/1424-8220/25/4/1016
- **Size:** 7,060 images, 25,269 instances
- **Split:** 4,915 train / 1,434 val / 711 test
- **Classes:** gas cylinder (with attribute classification for material, size, orientation)
- **Format:** Roboflow (export to YOLO/COCO/VOC etc.)
- **License:** CC BY 4.0
- **Relevance:** 5/5 -- Directly matches the gas cylinder class. Large, well-annotated, permissive license.
- **Notes:** Real-world scenes with challenging environments. Best available resource for gas cylinders.

### 6. Construction-PPE (Ultralytics)

- **URL:** https://docs.ultralytics.com/datasets/detect/construction-ppe/
- **Size:** ~1,416 images (1,132 train / 143 val / 141 test)
- **Classes (11):** helmet, no-helmet, vest, no-vest, gloves, no-gloves, boots, no-boots, goggles, no-goggles, person
- **Format:** Ultralytics YOLO
- **License:** AGPL-3.0
- **Relevance:** 3/5 -- Useful for PPE classes but small. Direct YOLO integration via Ultralytics hub.

### 7. Roboflow Construction Site Safety

- **URL:** https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety
- **Size:** 717 images
- **Classes:** hard hat, safety vest, person
- **Format:** Multi-format export (YOLO, COCO, VOC, etc.)
- **License:** Open source (Roboflow)
- **Relevance:** 3/5 -- Small but curated. Good supplement.

### 8. FireExtinguisher (Roboflow)

- **URL:** https://universe.roboflow.com/fire-extinguisher/fireextinguisher-z5atr
- **Size:** ~3,300 images
- **Classes (8):** Alarm_Activator, Fire_Blanket, Fire_Exit, Fire_Extinguisher, Fire_Suppression_Signage, Flashing_Light_Orbs, Sounders, extinguisher
- **Format:** Multi-format (YOLO, COCO, VOC, TFRecord, etc.)
- **License:** Roboflow (open source)
- **Relevance:** 5/5 -- Directly covers fire extinguisher, fire exit sign, fire blanket, and alarm classes. Excellent multi-class fire safety equipment dataset.
- **Notes:** Best available resource for fire safety equipment detection.

### 9. Fire Safety Equipment (Bayer et al., TUM)

- **URL:** https://mediatum.ub.tum.de/doc/1688375/tcu83fom5wjmacmp6gxdvtosv.Bayer%20et%20Al.%202022.pdf
- **Size:** 841 images (758 Dataset_L / 590 Dataset_S after splits)
- **Classes (4):** fire extinguisher, emergency call point, smoke detector, fire safety blanket
- **Format:** YOLO
- **License:** Academic
- **Relevance:** 4/5 -- Small but precisely annotated fire safety equipment dataset.

### 10. DeepQuestAI Fire-Smoke-Dataset

- **URL:** https://github.com/DeepQuestAI/Fire-Smoke-Dataset
- **Size:** ~3,000 images
- **Classes (3):** fire, smoke, neutral
- **Format:** Classification (folder-based)
- **License:** MIT
- **Relevance:** 3/5 -- Classification format (not bounding box). Useful for pre-training or augmenting fire/smoke data.

### 11. FireNet

- **URL:** https://github.com/OlafenwaMoses/FireNET
- **Paper:** https://arxiv.org/pdf/1905.11922
- **Size:** ~500+ images/frames (gathered from Flickr, Google, and other datasets)
- **Classes:** fire, non-fire
- **Format:** Classification
- **License:** Open
- **Relevance:** 2/5 -- Small and classification-only. Primarily useful as supplementary data.

### 12. FireRescue (UAV-Based)

- **URL:** https://arxiv.org/abs/2512.24622
- **Size:** 15,980 images, 32,000 bounding boxes
- **Classes (8):** Emergency Rescue Fire Truck, Water Tanker Fire Truck, Firefighter, Flames, Smoke, and 3 more
- **Format:** YOLO (associated with FRS-YOLO model)
- **License:** Academic (Dec 2025)
- **Relevance:** 3/5 -- UAV perspective useful for construction site overhead monitoring. Flame and smoke classes directly relevant.

### 13. MOCS (Moving Objects in Construction Sites)

- **URL:** https://competitions.codalab.org/competitions/32605
- **Paper:** https://www.sciencedirect.com/science/article/abs/pii/S0926580520310621
- **Size:** 41,668 images, 222,861 instances
- **Classes (13):** Worker, Tower crane, Hanging hook, Vehicle crane, Roller, Bulldozer, Excavator, Truck, Loader, Pump truck, Concrete transport mixer, Pile driver, Other vehicle
- **Format:** COCO (bbox + segmentation masks)
- **License:** Academic
- **Relevance:** 2/5 -- Focused on machinery/vehicles, not directly on fire safety. "Worker" class may help.

### 14. Emergency Exit Signs (Roboflow)

- **URL:** https://universe.roboflow.com/emergency-exit-signs/emergency-exit-signs
- **Size:** 482 images
- **Classes:** Left Exit, Right Exit, Straight Exit, Bidirectional Exit (and variants)
- **Format:** Multi-format (YOLO, COCO, etc.)
- **License:** Roboflow (open source)
- **Relevance:** 4/5 -- Directly covers exit sign class. Small but targeted.

### 15. Cigarette / Smoker YOLO (Roboflow)

- **URL:** https://universe.roboflow.com/yolo-pdvpx/cigarette-h2p1m
- **Alt URL:** https://universe.roboflow.com/cigaretteple-7m0hn/smoker-yolo/dataset/2
- **Size:** ~4,127 images (combined across projects)
- **Classes:** cigarette, smoking person
- **Format:** Multi-format (YOLO, COCO, etc.)
- **License:** Roboflow (open source)
- **Relevance:** 4/5 -- Directly covers cigarette detection. Small target detection is challenging but these provide a starting point.

### 16. Welding Sparks (Roboflow)

- **URL:** https://universe.roboflow.com/factory-bnvql/welding_sparks
- **Paper:** https://www.mdpi.com/1424-8220/23/15/6826
- **Size:** ~100-300 images (estimated from research; 100 images + 300 color masks in the academic paper)
- **Classes:** welding spark
- **Format:** Multi-format
- **License:** Roboflow / Academic
- **Relevance:** 3/5 -- Very small. Best available for welding sparks but will likely need augmentation from our own 12 welding videos.
- **Notes:** The academic paper by PMC used contour detection + CNN; limited public dataset availability. This is the weakest class in terms of public data.

### 17. Construction-Hazard-Detection (yihong1120)

- **URL (GitHub):** https://github.com/yihong1120/Construction-Hazard-Detection
- **URL (Roboflow):** https://universe.roboflow.com/object-detection-qn97p/construction-hazard-detection
- **Size:** Varies (based on Roboflow Construction Site Safety + extended annotations)
- **Classes (11):** Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person, Safety Cone, Safety Vest, Machinery, Utility Pole, Vehicle
- **Format:** YOLO
- **License:** Open source
- **Relevance:** 3/5 -- Good PPE classes, but mostly overlaps with other datasets. The extended annotations add value.

### 18. Kaggle Smoke-Fire-Detection-YOLO

- **URL:** https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo
- **Size:** Varies
- **Classes:** fire, smoke
- **Format:** YOLO
- **License:** Kaggle (check individual)
- **Relevance:** 3/5 -- YOLO-ready fire/smoke data. Useful as supplement to D-Fire/FASDD.

### 19. Pictor-PPE

- **URL:** https://github.com/ciber-lab/pictor-ppe
- **Size:** ~1,472 images (774 crowd-sourced + 698 web-mined)
- **Classes:** hard hat, safety vest, person
- **Format:** TF / YOLO
- **License:** Academic
- **Relevance:** 2/5 -- Small. Already covered by larger PPE datasets.

---

## Coverage Analysis by Target Class

| Target Class | Best Dataset(s) | Total Available Images (approx.) | Gap Assessment |
|-------------|-----------------|--------------------------------|----------------|
| **Fire** | D-Fire, FASDD, DeepQuestAI | 140,000+ | Well covered |
| **Smoke** | D-Fire, FASDD, DeepQuestAI | 140,000+ | Well covered |
| **Fire extinguisher** | FireExtinguisher (Roboflow), Bayer et al. | ~4,100 | Adequately covered |
| **Gas cylinder** | CylinDeRS | ~7,060 | Well covered |
| **Welding sparks** | Welding Sparks (Roboflow), academic paper | ~300 | MAJOR GAP -- need custom annotation |
| **Scaffold net** | None found | 0 | CRITICAL GAP -- no public dataset exists |
| **Tarpaulin** | None found | 0 | CRITICAL GAP -- no public dataset exists |
| **Exit sign** | Emergency Exit Signs (Roboflow), FireExtinguisher | ~800 | Partially covered, may need augmentation |
| **Hose reel** | None found | 0 | CRITICAL GAP -- no public dataset exists |
| **Electrical panel** | SODA ("ebox" class) | ~19,846 (subset) | Partially covered via proxy class |
| **Hard hat** | SODA, SH17, Construction-PPE, Roboflow | 30,000+ | Well covered |
| **Safety vest** | SODA, SH17, Construction-PPE, Roboflow | 30,000+ | Well covered |
| **Cigarette** | Cigarette/Smoker YOLO (Roboflow) | ~4,127 | Adequately covered |

---

## Hong Kong-Specific Resources

No HK-specific construction site fire safety datasets were found. Related HK resources:

1. **HK PolyU Synthetic Construction Equipment Dataset** -- Synthetic image generation for construction equipment detection (not fire safety). Academic paper only.
   - URL: https://research.polyu.edu.hk/en/publications/synthetic-image-dataset-development-for-vision-based-construction/

2. **CIM-WV Dataset** -- 2,000 images of Hong Kong high-rise building window views. Not directly relevant but demonstrates HK urban scene data availability.
   - URL: https://link.springer.com/article/10.1007/s44212-024-00039-7

3. **Our own dataset** -- 29 real photos from 2 HK construction sites + 12 welding videos remain the only HK-specific construction fire safety data.

---

## Recommendations

### Immediate Actions (High Impact, Low Effort)

1. **Download D-Fire** (CC0 license, YOLO format, 21.5k images) -- primary fire/smoke training data.
2. **Download CylinDeRS** from Roboflow (CC BY 4.0, 7k images) -- gas cylinder detection.
3. **Download FireExtinguisher dataset** from Roboflow (~3.3k images) -- fire extinguisher + fire exit + fire blanket.
4. **Download SODA** (~20k images) -- hard hat, safety vest, scaffold, electrical box.
5. **Download Emergency Exit Signs** from Roboflow (482 images) -- exit sign detection.
6. **Download Cigarette dataset** from Roboflow (~4k images) -- cigarette detection.

### Medium-Term Actions (Require More Effort)

7. **Download FASDD_CV** from Kaggle or SciDB -- supplement fire/smoke with diverse scenarios.
8. **Download SH17** from Kaggle/Zenodo -- comprehensive PPE detection (note: non-commercial license).
9. **Extract frames from our 12 welding videos** and annotate welding sparks to supplement the tiny Roboflow welding sparks dataset.
10. **Merge and harmonize** class labels across datasets into a unified YOLO training config.

### Custom Annotation Required (Critical Gaps)

11. **Scaffold net / tarpaulin:** No public datasets exist. Must collect and annotate images manually. Consider:
    - Web scraping construction site images showing scaffold nets/tarpaulins
    - Using our 29 HK site photos as seed data
    - Synthetic data generation using diffusion models
12. **Hose reel:** No public dataset. Similar approach as scaffold net -- collect and annotate.
13. **Welding sparks:** Very limited public data (~300 images). Prioritize frame extraction from our 12 welding videos with manual annotation.

### Dataset Combination Strategy

A practical training pipeline would merge:
- D-Fire + FASDD_CV for fire/smoke (~140k images)
- CylinDeRS for gas cylinders (~7k images)
- SODA + SH17 for hard hat/vest/scaffold (~28k images)
- FireExtinguisher (Roboflow) for fire safety equipment (~3.3k images)
- Emergency Exit Signs for exit signs (~500 images)
- Cigarette datasets for cigarettes (~4k images)
- Custom-annotated data for scaffold net, tarpaulin, hose reel, welding sparks

**Estimated total from public sources:** ~180,000+ images (before deduplication)
**Classes still requiring custom data collection:** scaffold net, tarpaulin, hose reel, welding sparks (partial)

---

## Key URLs Reference

- D-Fire: https://github.com/gaiasd/DFireDataset
- FASDD: https://www.scidb.cn/en/detail?dataSetId=ce9c9400b44148e1b0a749f5c3eb0bda
- FASDD_CV (Kaggle): https://www.kaggle.com/datasets/yuulind/fasdd-cv-coco
- SODA: https://arxiv.org/abs/2202.09554
- SH17: https://github.com/ahmadmughees/SH17dataset
- CylinDeRS: https://universe.roboflow.com/klearchos-stavrothanasopoulos-konstantinos-gkountakos-6jwgj/cylinders-iaq6n
- Construction-PPE: https://docs.ultralytics.com/datasets/detect/construction-ppe/
- FireExtinguisher: https://universe.roboflow.com/fire-extinguisher/fireextinguisher-z5atr
- Emergency Exit Signs: https://universe.roboflow.com/emergency-exit-signs/emergency-exit-signs
- Cigarette: https://universe.roboflow.com/yolo-pdvpx/cigarette-h2p1m
- Welding Sparks: https://universe.roboflow.com/factory-bnvql/welding_sparks
- Construction-Hazard-Detection: https://github.com/yihong1120/Construction-Hazard-Detection
- DeepQuestAI Fire-Smoke: https://github.com/DeepQuestAI/Fire-Smoke-Dataset
- FireNet: https://github.com/OlafenwaMoses/FireNET
- MOCS: https://competitions.codalab.org/competitions/32605
- Pictor-PPE: https://github.com/ciber-lab/pictor-ppe
- Bayer et al. (TUM): https://mediatum.ub.tum.de/doc/1688375/tcu83fom5wjmacmp6gxdvtosv.Bayer%20et%20Al.%202022.pdf
