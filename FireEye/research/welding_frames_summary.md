# Welding Video Frame Extraction Summary

**Date:** 2026-03-08
**Source:** `/home/evnchn/Scaffluent_App/Images dataset/welding/`
**Output:** `/home/evnchn/Scaffluent_App/FireEye/research/welding_frames/`
**Total frames extracted:** 96

## Extraction Parameters

- Videos under 5 seconds: extracted at 2 fps
- Videos 5 seconds or longer: extracted at 1 fps
- Output format: JPEG

## Per-Video Statistics

| Video | Duration (s) | Resolution | FPS Used | Frames |
|-------|-------------|------------|----------|--------|
| welding 1 | 10.73 | 480x854 | 1 | 11 |
| welding 2 | 8.48 | 640x360 | 1 | 8 |
| welding 3 | 4.73 | 608x1080 | 2 | 9 |
| welding 4 | 5.27 | 608x1080 | 1 | 5 |
| welding 5 | 5.23 | 608x1080 | 1 | 5 |
| welding 6 | 5.37 | 1280x720 | 1 | 5 |
| welding 7 | 5.47 | 608x1080 | 1 | 5 |
| welding 8 | 4.48 | 608x1080 | 2 | 9 |
| welding 9 | 4.77 | 608x1080 | 2 | 10 |
| welding 10 | 4.73 | 608x1080 | 2 | 9 |
| welding 11 | 4.77 | 608x1080 | 2 | 10 |
| welding 12 | 4.77 | 608x1080 | 2 | 10 |

## Resolution Breakdown

- **608x1080** (portrait): 10 videos (welding 3-5, 7-12) -- likely phone-recorded vertical video
- **480x854** (portrait): 1 video (welding 1)
- **640x360** (landscape): 1 video (welding 2)
- **1280x720** (landscape): 1 video (welding 6)

## Visual Content Analysis

Analysis based on per-frame pixel statistics (average brightness, bright pixel percentage, warm/red-dominant pixel percentage):

### High Spark/Flame Visibility (best for hot-works detection training)

- **Welding 1** (11 frames): Strong warm pixel presence (40-69% warm pixels in mid-frames), brightness peaks around frames 3-7 suggesting active welding arc with visible sparks. Brightness drops toward the end (frames 10-11) indicating welding stopped.
- **Welding 7** (5 frames): Consistently high warm pixels (~50%) and brightness (~145) across all frames. Suggests sustained, well-lit welding activity with constant spark/glow throughout.
- **Welding 9** (10 frames): Dramatic warm pixel ramp from 18% to 86%, indicating a welding sequence that intensifies. Strong orange/red dominance in later frames -- excellent for capturing active hot-work progression.
- **Welding 10** (9 frames): Warm pixels climb from 18% to 68%, showing a similar arc-intensification pattern. Good variety from setup to active welding.
- **Welding 12** (10 frames): Warm pixels range 1-71%, capturing the full lifecycle from pre-welding to peak activity to cooldown.

### Moderate Spark/Flame Visibility

- **Welding 3** (9 frames): High warm pixels in frame 1 (65%) dropping to ~20%, with moderate bright pixels. May show welding aftermath or lower-intensity work.
- **Welding 8** (9 frames): Highly variable -- frame 3 has 95% bright pixels (likely a flash/arc burst), frame 6 is very dark (brightness 51) with 37% warm pixels. Shows dramatic lighting changes typical of arc welding.
- **Welding 11** (10 frames): Moderate warm pixels (13-21%) with gradual increase. Subtler welding activity, possibly at greater distance or with less intense sparks.

### Low Spark/Flame Visibility (still useful as context)

- **Welding 2** (8 frames): Very low warm pixels (6-10%), neutral color balance. Likely shows welding from a distance or with protective shielding reducing visible sparks.
- **Welding 4** (5 frames): Very low warm pixels (~2%), higher file sizes suggesting detailed scenes. May show industrial welding setup with less visible arc.
- **Welding 5** (5 frames): Similar profile to welding 4 -- low warm pixels, moderate brightness. Industrial welding with limited direct spark visibility.
- **Welding 6** (5 frames): Low warm pixels (1-8%), highest resolution (1280x720). Shows welding environment with occasional warm flashes.

## Assessment of Training Usefulness

### Strengths
- **96 total frames** provide a reasonable starting set for hot-works detection
- **Good variety of welding stages**: pre-welding setup, active arc, peak sparks, cooldown/aftermath
- **Range of spark intensities**: from barely visible to frame-filling bright arcs (welding 8 frame 3: 95% bright pixels)
- **Multiple perspectives**: both portrait and landscape orientations, varying distances from the welding activity
- **Temporal progression**: several videos capture the full welding lifecycle, useful for understanding how hot-works scenes evolve
- **Real-world conditions**: varying lighting, backgrounds, and welding types

### Limitations
- **Small dataset**: 96 frames is a starting point but insufficient alone for robust model training
- **Resolution inconsistency**: most videos are 608x1080 portrait; only 2 are landscape, limiting viewpoint diversity
- **Short clips**: most videos are under 6 seconds, limiting the number of distinct frames per video
- **No annotations**: frames lack bounding boxes or labels for specific hazards (sparks, arc, molten material, etc.)
- **Unknown welding types**: cannot determine from pixel analysis alone whether these show MIG, TIG, stick welding, or cutting -- different types present different fire hazard profiles

### Recommendations
- Use these frames as positive examples for the "hot works" fire hazard category in FireEye
- Frames with high warm-pixel percentages (welding 1, 7, 9, 10, 12) are best candidates for spark/flame detection examples
- The high-variability frames (welding 8) are valuable for testing robustness against sudden brightness changes
- Consider augmentation (rotation, brightness adjustment, cropping) to expand the effective dataset size
- Supplement with additional welding imagery from other sources for better generalization
