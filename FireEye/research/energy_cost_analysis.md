# Local vs Cloud LLM Inference: Energy Cost Analysis

## Assumptions
- Electricity: HKD 1.2/kWh (USD 0.154/kWh)
- Cloud API: ~USD 0.001/image (OpenRouter, Gemini 3 Flash)
- Local model: Qwen2.5-VL-7B (the tested local VLM candidate)

## 1. Cost Per Image — Local Inference (Electricity Only)

| GPU         | Power (W) | Time/image (s) | Energy (Wh) | Cost/image (USD) |
|-------------|-----------|-----------------|-------------|-------------------|
| RTX 3060    | 170       | 11.5            | 0.543       | 0.0000837         |
| RTX 3060    | 115       | 11.5*           | 0.367       | 0.0000566         |
| Tesla P4    | 75        | 20.0            | 0.417       | 0.0000642         |

*At 115W the GPU may throttle slightly, but 7B inference is rarely fully compute-bound.

## 2. Cloud vs Local — Cost Ratio

Cloud API at $0.001/image is **12-18x more expensive** than local electricity alone:
- RTX 3060 170W: cloud is 11.9x costlier
- RTX 3060 115W: cloud is 17.7x costlier
- Tesla P4 75W:  cloud is 15.6x costlier

## 3. Break-Even Analysis (Electricity vs Cloud)

On pure per-image cost, local is cheaper from **image #1**. However, local has
fixed costs: hardware depreciation and idle power. Accounting for those:

| Factor                    | RTX 3060          | Tesla P4 (used)   |
|---------------------------|-------------------|-------------------|
| GPU cost (approx)         | USD 300           | USD 60            |
| Useful life (years)       | 4                 | 2                 |
| Daily depreciation        | $0.205/day        | $0.082/day        |
| Break-even images/day     | ~224/day          | ~88/day           |

Formula: depreciation / (cloud_cost - local_cost) = break-even volume.
At 224+ images/day on RTX 3060, local wins even including hardware cost.

## 4. Monthly Electricity Cost (Local Inference)

Using RTX 3060 at 170W, 11.5s/image:

| Images/day | Daily energy (Wh) | Monthly cost (USD) |
|------------|--------------------|--------------------|
| 100        | 54.3               | $0.25              |
| 500        | 271.5              | $1.25              |
| 1,000      | 543.1              | $2.51              |

These costs are negligible. Even 1,000 images/day costs ~$2.50/month in electricity.
The same volume via cloud API would cost $30/month.

## 5. The Sweet Spot

| Volume (images/day) | Cloud (USD/mo) | Local electricity | Local + depreciation | Winner     |
|----------------------|----------------|-------------------|----------------------|------------|
| 50                   | $1.50          | $0.13             | $6.28 (3060)         | Cloud      |
| 100                  | $3.00          | $0.25             | $6.40                | Cloud      |
| 250                  | $7.50          | $0.63             | $6.78                | ~Tied      |
| 500                  | $15.00         | $1.25             | $7.40                | **Local**  |
| 1,000                | $30.00         | $2.51             | $8.66                | **Local**  |

The crossover is around **250 images/day** for RTX 3060 (or ~90/day for Tesla P4).

## Recommendation

For FireEye's current use case (demo/evaluation, likely <100 images/day), **cloud API
is simpler and cost-competitive**. The operational advantages (no model loading, no VRAM
contention, better model quality from Gemini 3 Flash vs 7B local) outweigh the small
electricity savings.

**Switch to local when**: sustained volume exceeds ~250 images/day, or if API latency,
privacy requirements, or network reliability become concerns. At that point the Tesla P4
is the most cost-efficient option given its low power draw and $60 acquisition cost.
