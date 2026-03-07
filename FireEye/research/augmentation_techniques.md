# Data Augmentation Techniques for FireEye YOLO Training

**Context:** 29 annotated images + video frames for detecting fire, smoke, fire extinguishers,
gas cylinders, welding sparks, scaffold nets, exit signs, etc. on Hong Kong construction sites.
The dataset is heavily skewed toward high-severity fire scenes; low/medium severity is underrepresented.

**Research date:** 2026-03-08

---

## Table of Contents

1. [Classical Augmentation for Object Detection](#1-classical-augmentation-for-object-detection)
2. [Advanced Augmentation](#2-advanced-augmentation)
3. [Generative Augmentation](#3-generative-augmentation)
4. [Few-Shot and Self-Training](#4-few-shot-and-self-training)
5. [Severity Variation Strategies (FireEye-Specific)](#5-severity-variation-strategies-fireeye-specific)
6. [Recent Literature on Fire Detection Augmentation](#6-recent-literature-on-fire-detection-augmentation-2024-2026)
7. [Recommended Pipeline for FireEye](#7-recommended-pipeline-for-fireeye)

---

## 1. Classical Augmentation for Object Detection

### 1.1 Geometric Transforms

| Transform | Effect | Fire/Safety Notes |
|-----------|--------|-------------------|
| **Horizontal flip** | Doubles effective dataset. Safe for most objects. | Fire, smoke, equipment are orientation-agnostic. Always enable. |
| **Vertical flip** | Less natural but useful for overhead/drone views. | Use sparingly; construction scenes have a strong gravity prior. |
| **Rotation (small, +/-15 deg)** | Simulates camera tilt. | Keep small to avoid unrealistic fire orientations. |
| **Random scale (0.5x-1.5x)** | Simulates distance variation. | Critical: fire extinguishers and exit signs appear at many scales. |
| **Random crop** | Forces model to detect partially visible objects. | Helps with edge-of-frame scenarios common on CCTV. |
| **Perspective / affine transform** | Simulates different camera angles. | Useful for construction sites with varied camera mounts. |
| **Shear** | Mild geometric distortion. | Keep magnitude low (<10 deg) to maintain realism. |

**YOLO-specific note:** Ultralytics YOLO applies many of these at training time via built-in
hyperparameters (`degrees`, `translate`, `scale`, `shear`, `perspective`, `flipud`, `fliplr`).
These are configured in the training YAML or via CLI args.

### 1.2 Photometric Transforms

| Transform | Effect | Fire/Safety Notes |
|-----------|--------|-------------------|
| **Brightness (+/-)** | Simulates day/night, indoor/outdoor. | Essential: construction sites operate in varying light. |
| **Contrast** | Enhances or flattens features. | Smoke in low-contrast scenes is a key failure mode. |
| **Saturation** | Varies color intensity. | Fire color varies from pale yellow to deep orange/red. |
| **Hue shift** | Small shifts simulate different flame temperatures. | Keep small (+/-15 deg) to avoid unrealistic colors. |
| **Gaussian noise** | Simulates sensor noise, compression artifacts. | Helps with low-quality CCTV feeds common on sites. |
| **Gaussian blur** | Simulates out-of-focus or motion blur. | Smoke is inherently blurry; helps model learn soft edges. |
| **CLAHE** | Adaptive histogram equalization. | Improves detail in shadowed construction areas. |
| **Random rain/fog overlay** | Simulates weather conditions. | HK has humid/rainy climate; important for outdoor scenes. |

**YOLO built-in params:** `hsv_h`, `hsv_s`, `hsv_v` control hue/saturation/value jitter.

### 1.3 Mosaic Augmentation (YOLOv4+)

Mosaic stitches **4 training images into one** by resizing each to a quadrant and taking a
random crop of the composite. Benefits:

- Increases object density per training image (sees 4x more objects per batch element).
- Forces the model to handle objects at multiple scales and partial visibility.
- Reduces reliance on large batch sizes (each mosaic image is effectively a mini-batch).
- Controlled via `mosaic` hyperparameter (probability, default 1.0 in Ultralytics).

**For small datasets this is extremely valuable** because it combinatorially creates O(n^4)
unique training images from n source images. With 29 images, that is potentially ~700k unique
mosaic combinations.

**Mosaic-9 variant:** Stitches 9 images instead of 4, further increasing density. Available
in some YOLOv5/v8 implementations.

**Important:** Mosaic is typically disabled in the last N epochs (default: last 10) to let the
model fine-tune on single, unperturbed images for final performance.

### 1.4 Albumentations Library

[Albumentations](https://albumentations.ai/) is the gold-standard augmentation library for
object detection. Key capabilities:

- **Bounding-box-aware transforms:** All spatial transforms automatically adjust bounding
  boxes. Supports YOLO format (`[x_center, y_center, width, height]` normalized).
- **Pipeline composition:** `A.Compose([...], bbox_params=A.BboxParams(format='yolo'))`.
- **70+ transforms** including geometric, photometric, weather simulation, blur, noise, etc.
- **Ultralytics integration:** Ultralytics YOLO natively integrates with Albumentations.
  Custom pipelines can be injected into the training loop.
- **Performance:** Written in NumPy/OpenCV, very fast. Supports multiprocessing data loaders.

**Example pipeline for FireEye:**

```python
import albumentations as A

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=30, p=0.5),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.GaussianBlur(blur_limit=(3, 7), p=0.2),
    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.2),
    A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, p=0.1),
    A.CLAHE(clip_limit=4.0, p=0.2),
    A.Perspective(scale=(0.02, 0.05), p=0.2),
    A.RandomScale(scale_limit=0.3, p=0.3),
    A.PadIfNeeded(min_height=640, min_width=640, border_mode=0),
    A.RandomCrop(height=640, width=640, p=1.0),
], bbox_params=A.BboxParams(format='yolo', min_visibility=0.3, label_fields=['class_labels']))
```

### 1.5 Offline vs. Online Augmentation

| Approach | Pros | Cons |
|----------|------|------|
| **Online** (during training) | Infinite variation; no disk cost | Slower training per epoch |
| **Offline** (pre-generate) | Inspect quality; faster training | Disk-heavy; fixed variation |

**Recommendation for 29 images:** Use **both**. Offline-generate a 5-10x expanded dataset
with careful quality control, then apply online augmentation on top during training.

---

## 2. Advanced Augmentation

### 2.1 Copy-Paste Augmentation

The **Simple Copy-Paste** method (Ghiasi et al., CVPR 2021) is one of the strongest
augmentation strategies for detection and segmentation with small datasets:

1. **Extract** object instances from labeled images (using bounding boxes or masks).
2. **Paste** them onto other background images at random locations and scales.
3. **Update** bounding box annotations accordingly.

**Why it is especially powerful for FireEye:**

- Can paste fire extinguishers, gas cylinders, exit signs onto clean background scenes.
- Can paste fire/smoke onto non-fire construction images to create synthetic fire scenarios.
- Can vary the number of pasted objects to simulate different scene densities.
- With only 15 images of a new category, studies show models can reach 95% accuracy.

**Implementation options:**
- [conradry/copy-paste-aug](https://github.com/conradry/copy-paste-aug) -- ready-to-use library.
- [RocketFlash/CAP_augmentation](https://github.com/RocketFlash/CAP_augmentation) -- Cut and Paste augmentation.
- Custom implementation: extract objects via bounding box crop, alpha-blend onto backgrounds.

**Context-Aware Copy-Paste (CACP, 2024):** Newer variant that uses background classifiers to
ensure pasted objects appear in contextually plausible locations (e.g., fire extinguisher on a
wall, not floating in the sky).

**Key considerations:**
- Paste with slight alpha blending at edges to avoid hard boundaries.
- Apply small random transforms (rotation, scale, brightness) to pasted objects.
- Restrict paste regions to plausible locations in the scene.

### 2.2 CutMix

CutMix cuts a rectangular region from one image and pastes it onto another:

- Controlled via `cutmix` hyperparameter in Ultralytics (probability).
- Creates realistic partial-occlusion scenarios.
- Less powerful than copy-paste for fire detection because it does not preserve semantic
  coherence (a random rectangle from a fire image pasted onto a scaffold image may not
  be meaningful).

**Verdict:** Use as supplementary augmentation, not primary.

### 2.3 MixUp

MixUp blends two images together (weighted average, typically 50/50):

- Controlled via `mixup` hyperparameter in Ultralytics.
- Creates ghost-like overlays; less realistic but forces the model to learn from mixed features.
- Lambda sampled from Beta(32, 32) distribution in YOLOX.

**Verdict for fire detection:** Moderate usefulness. Smoke is semi-transparent, so MixUp
coincidentally simulates light-smoke overlays. Set probability low (~0.1-0.2).

### 2.4 Style Transfer for Domain Variation

Neural style transfer can transform images to simulate:

- Different times of day (daytime scene to nighttime).
- Different weather conditions (clear to foggy/rainy).
- Different camera qualities (high-res DSLR to grainy CCTV).

**Practical approach:** Use fast neural style transfer (e.g., Johnson et al.) or simpler
photometric domain randomization rather than full artistic style transfer.

### 2.5 Domain Randomization

Randomize aspects of the scene that are not relevant to the detection task:

- Background texture replacement.
- Random color shifts to non-object regions.
- Random overlay of construction-site-typical clutter (scaffolding patterns, netting, etc.).

**For FireEye:** Most useful when combined with copy-paste -- paste fire/safety objects onto
heavily randomized construction backgrounds.

---

## 3. Generative Augmentation

### 3.1 Inpainting for Object Addition/Removal

**Stable Diffusion inpainting** can add or remove objects within a scene while preserving
context:

**Adding objects (positive examples):**
- Mask a wall area -> inpaint a fire extinguisher.
- Mask an open area -> inpaint a gas cylinder.
- Mask part of a scene -> inpaint fire/smoke of varying intensity.

**Removing objects (negative examples):**
- Mask an existing fire extinguisher -> inpaint the background to create a "missing
  equipment" scene.
- Remove fire/smoke from a fire scene to create "before" version.

**Recent results (2024-2025):**
- Segmentation-guided diffusion inpainting improved mAP50 from 0.579 to 0.647 in forestry
  detection tasks (a ~12% improvement).
- Progressive augmentation in 10% increments up to 200% additional data showed consistent
  gains.
- Stable Diffusion + ControlNet can maintain structural consistency of the scene while
  adding objects.

**Tools:**
- Stable Diffusion WebUI (AUTOMATIC1111 or ComfyUI) with inpainting mode.
- ControlNet for structural guidance (edge maps, depth maps).
- Text prompts like "fire extinguisher mounted on wall", "small flame on welding torch",
  "thin smoke haze in construction area".

**Caution -- domain gap:** Synthetic images may have subtle artifacts that the model
overfits to. Mitigations:
- Mix synthetic and real data (no more than 50-70% synthetic).
- Apply additional augmentation to synthetic images to break artifacts.
- Validate on real-only test set.

### 3.2 Background Replacement / Scene Composition

Preserve labeled foreground objects while replacing the background:

1. **Segment** the foreground (fire, equipment) using SAM (Segment Anything Model).
2. **Replace** background with different construction site images or generated backgrounds.
3. **Composite** with proper lighting adjustment.

This is effectively an advanced form of copy-paste that produces more realistic results by
using segmentation models rather than bounding-box crops.

**X-Paste (ICML 2023):** Scales copy-paste augmentation by using CLIP-retrieved or
diffusion-generated instances, removing the need for manually segmented source objects.

### 3.3 Text-to-Image Generation

Use text-to-image diffusion models to generate entirely synthetic training images:

- Prompt: "CCTV footage of a small fire near gas cylinders on a Hong Kong construction site"
- Prompt: "fire extinguisher mounted on scaffolding at a building site, daytime"
- Prompt: "welding sparks flying near scaffold netting on a construction site"

**Workflow:**
1. Generate candidate images with Stable Diffusion / SDXL / Flux.
2. Manually verify and annotate (or use a pretrained detector for pre-annotation).
3. Add to training set.

**Limitation:** Generated images may lack the specific visual characteristics of HK
construction sites. Fine-tuning the diffusion model on real site photos (even without
annotations) via DreamBooth or LoRA can improve domain specificity.

---

## 4. Few-Shot and Self-Training

### 4.1 Transfer Learning Strategy

**Starting point matters enormously for small datasets.**

| Pretrained Model | Recommendation |
|-----------------|----------------|
| **YOLOv8/v11 COCO-pretrained** | Best starting point. COCO includes "fire hydrant" but not fire/smoke. Still provides excellent low-level and mid-level features. |
| **YOLOv8/v11 Objects365-pretrained** | Objects365 has more diverse categories. May transfer better. |
| **YOLO-World (open-vocabulary)** | Can zero-shot detect "fire", "smoke", "fire extinguisher" without fine-tuning. Use as pseudo-label generator (see 4.3). |
| **Fine-tune from FASDD-pretrained** | If a FASDD-trained checkpoint is available, this provides fire/smoke domain knowledge. |

**Freezing strategy for small datasets:**
1. Freeze backbone, train only head (first 5-10 epochs).
2. Unfreeze backbone with low learning rate (1/10th of head LR).
3. This prevents catastrophic forgetting of pretrained features.

### 4.2 Semi-Supervised Learning with Pseudo-Labels

The **teacher-student** paradigm for semi-supervised object detection:

1. **Teacher model:** Train on the 29 labeled images.
2. **Generate pseudo-labels:** Run the teacher on a large pool of unlabeled construction
   site images (scraped from CCTV footage, stock photos, web images).
3. **Filter:** Keep only high-confidence detections (confidence > 0.7-0.8).
4. **Student model:** Train on labeled + pseudo-labeled data.
5. **Iterate:** The student becomes the next teacher. Repeat 2-3 rounds.

**YOLO implementations:**
- [Efficient Teacher](https://github.com/AlibabaResearch/efficientteacher) -- Alibaba's
  semi-supervised framework for YOLO series.
- Manual pseudo-labeling loop with Ultralytics `model.predict()` + confidence filtering.

**Expected gains:** Studies show 1-3 mAP points improvement on COCO with semi-supervised
methods. For small datasets the relative gain is much larger.

**Practical approach for FireEye:**
1. Collect 500-1000 unlabeled construction site images from CCTV feeds or web scraping.
2. Train initial YOLO on 29 labeled images with heavy augmentation.
3. Predict on unlabeled set, manually verify top-confidence predictions.
4. Add verified pseudo-labels to training set.
5. Retrain and repeat.

### 4.3 YOLO-World for Zero-Shot Pre-Annotation

YOLO-World is an open-vocabulary detector that can detect objects from text descriptions:

1. Run YOLO-World with prompts: "fire", "smoke", "fire extinguisher", "gas cylinder",
   "welding sparks", "scaffold net", "exit sign".
2. Use detections as pseudo-labels on unlabeled images.
3. Manually verify and correct.
4. Use corrected labels for training a standard YOLOv8/v11.

This is faster than annotating from scratch and can rapidly expand the labeled dataset.

### 4.4 Active Learning

Rather than randomly selecting images to label, use **active learning** to select the most
informative images:

1. Train initial model on 29 images.
2. Run inference on large unlabeled pool.
3. Select images where the model is most uncertain (low confidence, conflicting predictions).
4. Manually label those images (highest value-per-label).
5. Retrain and repeat.

---

## 5. Severity Variation Strategies (FireEye-Specific)

### 5.1 The Severity Problem

Current dataset is dominated by high-severity fire scenes (large flames, heavy smoke). The
model needs to detect:

- **Low severity:** Small sparks, smoldering, thin haze, minor welding residue.
- **Medium severity:** Growing flames, spreading smoke, visible but contained fire.
- **High severity:** Large fires, dense smoke, structural involvement (already well-represented).

### 5.2 Creating Severity Variation from Existing Images

**Reducing fire/smoke intensity (high -> low/medium):**

| Technique | Implementation |
|-----------|---------------|
| **Opacity reduction** | Alpha-blend fire/smoke region with original background (50-80% transparency simulates thinner smoke). |
| **Scale reduction** | Crop fire region, resize smaller, paste back (simulates earlier-stage fire). |
| **Color shift** | Shift flame color toward yellow/white (lower intensity) or make smoke lighter/more transparent. |
| **Partial masking** | Mask portions of fire/smoke region and inpaint background (simulates partially contained fire). |
| **Gaussian blur on fire region** | Heavy blur on flame area simulates heat haze / early-stage smoldering rather than open flame. |

**Adding fire/smoke to clean scenes (none -> low/medium):**

| Technique | Implementation |
|-----------|---------------|
| **Semi-transparent smoke overlay** | Use PNG smoke assets with alpha channel; vary opacity 10-40% for light haze. |
| **Small flame insertion** | Crop small flame from high-severity image; paste at reduced scale near welding/hot-work areas. |
| **Spark generation** | Programmatically generate small bright points (welding sparks) using drawing operations. |
| **Diffusion inpainting** | Prompt: "thin smoke haze in construction area" / "small flame on metal surface". |

### 5.3 Simulating Before/During/After Scenarios

**Before (normal operation):**
- Remove fire/smoke from existing fire images via inpainting.
- These become negative examples showing normal construction with no hazards.
- Still annotate fire extinguishers, gas cylinders, exit signs present in the scene.

**During (active incident):**
- Original high-severity images.
- Synthetically generated low/medium severity versions.
- Copy-paste fire/smoke onto "before" scenes at varying intensities.

**After (post-incident):**
- Darken/blacken areas where fire was (scorch marks).
- Reduce flame visibility, increase smoke density (smoldering).
- These are rare but important for incident documentation.

### 5.4 Equipment Detection Augmentation

Fire extinguishers, gas cylinders, exit signs, and scaffold nets are relatively rigid objects.
Best augmented via:

1. **Copy-paste** at various scales and positions throughout construction backgrounds.
2. **Rotation** (extinguishers can be wall-mounted at any angle).
3. **Partial occlusion** (paste scaffolding/netting partially over equipment).
4. **Web scraping** for additional labeled or easy-to-label images of these objects.
5. **Color variation** (extinguishers come in red, yellow, green; gas cylinders vary by gas type).

---

## 6. Recent Literature on Fire Detection Augmentation (2024-2026)

### Key Papers and Resources

1. **FASDD (Flame And Smoke Detection Dataset):** Over 120,000 heterogeneous images across
   three sub-datasets (CV, UAV, Remote Sensing). Published June 2024 in Earth System Science
   Data. Provides a large benchmark for fire detection and could serve as a source of
   copy-paste objects.
   - Source: [ESSD preprint](https://essd.copernicus.org/preprints/essd-2023-73/)

2. **AI-Assisted Fire Risk Target Detection for Under-Construction Nuclear Power Plants (2025):**
   Addresses fire detection in construction environments specifically, using synthetic fire and
   smoke assets blended into base images to generate 1,580 augmented samples. Achieved high
   detection accuracy (95.6% fire, 92% smoke).
   - Source: [MDPI Fire 2025](https://www.mdpi.com/2571-6255/9/3/115)

3. **Early Fire and Smoke Detection: A Comprehensive Review (2025):** Reviews models,
   datasets, and challenges across fire detection approaches including CNNs, YOLO, and
   Faster R-CNN.
   - Source: [MDPI Applied Sciences](https://www.mdpi.com/2076-3417/15/18/10255)

4. **Generative AI and Simulation-Based Data Augmentation (2025):** Demonstrates diffusion-
   based inpainting for low-data environments, achieving mAP50 improvements from 0.579 to
   0.647 over real-only training.
   - Source: [MDPI Forests](https://www.mdpi.com/1999-4907/17/3/302)

5. **Data Augmentation for Object Detection via Controllable Diffusion Models (WACV 2024):**
   Uses diffusion models with spatial and textual control to generate augmented training
   data for object detection.
   - Source: [WACV 2024 paper](https://openaccess.thecvf.com/content/WACV2024/papers/Fang_Data_Augmentation_for_Object_Detection_via_Controllable_Diffusion_Models_WACV_2024_paper.pdf)

6. **Advances in Diffusion Models for Image Data Augmentation (2025):** Comprehensive review
   of diffusion-based augmentation methods, evaluation metrics, and future directions.
   - Source: [Springer AI Review](https://link.springer.com/article/10.1007/s10462-025-11116-x)

7. **Semi-Supervised Object Detection: A Survey (2024):** Covers teacher-student architectures,
   pseudo-labeling, and consistency regularization methods from CNNs to Transformers.
   - Source: [arXiv 2407.08460](https://arxiv.org/html/2407.08460v1)

### Available Public Datasets for Transfer / Copy-Paste Source

| Dataset | Size | Objects | Use for FireEye |
|---------|------|---------|-----------------|
| FASDD | 120k+ images | Fire, smoke | Copy-paste fire/smoke instances; pretrain then fine-tune |
| DeepFire | 35k+ images | Fire, smoke | Additional fire/smoke training data |
| COCO | 330k images | 80 categories (incl. fire hydrant) | General pretrained backbone |
| Objects365 | 2M images | 365 categories | Stronger pretrained backbone |
| Roboflow fire datasets | Various | Fire, smoke, extinguisher | Search Roboflow Universe for specific object classes |

---

## 7. Recommended Pipeline for FireEye

Given the constraints (29 images + video frames, HK construction site domain, multiple object
classes with severity variation), here is a prioritized augmentation strategy:

### Phase 1: Immediate (Offline Data Expansion)

**Goal: 29 images -> 300-500 images**

1. **Extract video frames** at varying intervals to capture temporal diversity.
2. **Copy-paste augmentation:** Extract fire/smoke/equipment instances from labeled images.
   Paste onto clean construction site backgrounds at varying scales and positions.
   - Priority: paste fire extinguishers and gas cylinders onto many backgrounds.
   - Paste fire/smoke at reduced opacity/scale for low-severity variants.
3. **Albumentations offline pipeline:** Generate 5-10 variants per image with geometric +
   photometric transforms. Visually inspect all generated images.
4. **FASDD mining:** Download FASDD dataset, find images most similar to construction
   sites, and either use directly or extract fire/smoke instances for copy-paste.

### Phase 2: Generative Expansion

**Goal: 500 -> 1000-2000 images**

5. **Stable Diffusion inpainting:**
   - Add fire/smoke of varying severity to clean construction scenes.
   - Remove fire/smoke from fire scenes to create negative examples.
   - Add/reposition safety equipment.
6. **Text-to-image generation** of construction fire scenarios (with manual annotation).
7. **Manual quality review** of all synthetic images. Discard unrealistic ones.

### Phase 3: Semi-Supervised Expansion

**Goal: Leverage unlabeled data**

8. **Collect unlabeled construction site images** (500-1000).
9. **Train initial model** on Phase 1+2 data with heavy online augmentation.
10. **YOLO-World pre-annotation** on unlabeled pool.
11. **Manual verification** of pseudo-labels.
12. **Retrain** with expanded labeled set.

### Phase 4: Training Configuration

**Online augmentation during YOLO training:**

```yaml
# Ultralytics training hyperparameters
mosaic: 1.0          # Mosaic probability (essential for small datasets)
mixup: 0.15          # MixUp probability (moderate, helps with smoke)
copy_paste: 0.3      # Copy-paste probability
hsv_h: 0.015         # Hue augmentation
hsv_s: 0.7           # Saturation augmentation
hsv_v: 0.4           # Value (brightness) augmentation
degrees: 10.0        # Rotation degrees
translate: 0.1       # Translation fraction
scale: 0.5           # Scale augmentation (+/- 50%)
shear: 5.0           # Shear degrees
perspective: 0.0005  # Perspective distortion
flipud: 0.1          # Vertical flip probability
fliplr: 0.5          # Horizontal flip probability
close_mosaic: 10     # Disable mosaic for last 10 epochs
```

**Transfer learning setup:**

```bash
# Start from COCO-pretrained YOLOv8m (medium -- good balance for small datasets)
yolo detect train model=yolov8m.pt data=fireeye.yaml epochs=200 \
    patience=30 batch=8 imgsz=640 freeze=10 \
    lr0=0.001 lrf=0.01 warmup_epochs=5
```

- Use `freeze=10` to freeze backbone for initial epochs.
- Use `patience=30` for early stopping.
- Use `batch=8` (small batch for small dataset -- avoids overfitting).
- `imgsz=640` is standard; consider 1280 if GPU memory allows (better for small objects).

---

## Summary of Expected Impact

| Technique | Effort | Expected Impact | Priority |
|-----------|--------|-----------------|----------|
| Mosaic (built-in) | Zero (default on) | High | Must-have |
| Albumentations pipeline | Low | High | Must-have |
| Copy-paste augmentation | Medium | Very High | Must-have |
| Transfer learning from COCO | Zero | High | Must-have |
| Severity variation (opacity/scale) | Medium | High (fills dataset gap) | High |
| FASDD dataset mining | Medium | High | High |
| Diffusion inpainting | Medium-High | Medium-High | Medium |
| Semi-supervised pseudo-labels | High | Medium-High | Medium |
| Text-to-image generation | High | Medium (domain gap risk) | Low-Medium |
| Style transfer / domain randomization | Medium | Low-Medium | Low |

**Bottom line:** For a 29-image dataset, the combination of (1) COCO-pretrained YOLO +
(2) aggressive mosaic/Albumentations + (3) copy-paste augmentation + (4) mining FASDD for
additional fire/smoke instances can realistically push effective training set size to 1000+
images and produce a usable detector. Generative methods and semi-supervised learning provide
further gains but with diminishing returns and higher effort.

---

## Sources

- [Ultralytics YOLO Data Augmentation Docs](https://docs.ultralytics.com/guides/yolo-data-augmentation/)
- [Data Augmentation in YOLOv4 (Roboflow)](https://blog.roboflow.com/yolov4-data-augmentation/)
- [Albumentations Bounding Box Augmentation](https://albumentations.ai/docs/3-basic-usage/bounding-boxes-augmentations/)
- [Ultralytics Albumentations Integration](https://docs.ultralytics.com/integrations/albumentations/)
- [Simple Copy-Paste is a Strong Data Augmentation Method (CVPR 2021)](https://arxiv.org/abs/2012.07177)
- [Context-Aware Copy-Paste (CACP, 2024)](https://arxiv.org/html/2407.08151v2)
- [FASDD: 100,000-level Flame and Smoke Detection Dataset](https://essd.copernicus.org/preprints/essd-2023-73/)
- [Early Fire and Smoke Detection: A Comprehensive Review (2025)](https://www.mdpi.com/2076-3417/15/18/10255)
- [AI-Assisted Fire Risk Detection for Under-Construction Nuclear Power Plants (2025)](https://www.mdpi.com/2571-6255/9/3/115)
- [Generative AI Data Augmentation for Low-Data Environments (2025)](https://www.mdpi.com/1999-4907/17/3/302)
- [Data Augmentation via Controllable Diffusion Models (WACV 2024)](https://openaccess.thecvf.com/content/WACV2024/papers/Fang_Data_Augmentation_for_Object_Detection_via_Controllable_Diffusion_Models_WACV_2024_paper.pdf)
- [Diffusion Models for Image Augmentation Review (2025)](https://link.springer.com/article/10.1007/s10462-025-11116-x)
- [Semi-Supervised Object Detection Survey (2024)](https://arxiv.org/html/2407.08460v1)
- [Efficient Teacher (Alibaba Semi-Supervised YOLO)](https://github.com/AlibabaResearch/efficientteacher)
- [YOLOX Mosaic and Mixup Explanation](https://gmongaras.medium.com/yolox-explanation-mosaic-and-mixup-for-data-augmentation-3839465a3adf)
- [Mosaic Data Augmentation (Analytics Vidhya)](https://www.analyticsvidhya.com/blog/2023/12/mosaic-data-augmentation/)
- [Open Flame and Smoke Detection Dataset (2024)](https://www.tandfonline.com/doi/full/10.1080/10095020.2024.2347922)
- [Advancements in Small-Object Detection 2023-2025](https://www.mdpi.com/2076-3417/15/22/11882)
