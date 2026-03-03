#!/usr/bin/env /usr/bin/python3
"""Generate fire escalation frame sequences using OpenRouter image APIs.

Strategy:
  - Frame 1: Generate a calm base scene with FLUX.2 Klein ($0.014/img)
  - Frames 2-N: Iteratively edit using Gemini 3.1 Flash Image, progressively
    adding fire escalation. Multi-turn editing preserves scene composition.

Output structure:
  openrouter_frames/{scene_id}/frame_01.png  ... frame_10.png

5 scenarios (8-10 frames each):
  1. campfire_spread       - Campfire spreading beyond stone ring
  2. welding_sparks        - Welding sparks igniting scaffolding
  3. agricultural_burn     - Agricultural burn jumping firebreak
  4. bbq_grease_fire       - BBQ grease fire spreading to furniture
  5. fireplace_embers      - Fireplace embers on carpet

Budget target: <$5 total for ~50 images.
"""

import base64
import json
import os
import sys
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
FIREEYE_DIR = SCRIPT_DIR.parent.parent  # FireEye/
OUTPUT_DIR = SCRIPT_DIR / "openrouter_frames"

API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Models
FLUX_MODEL = "black-forest-labs/flux.2-klein-4b"
GEMINI_MODEL = "google/gemini-3.1-flash-image-preview"

# Costs (approximate)
FLUX_COST = 0.014   # per image
GEMINI_COST = 0.07  # per edit (average of $0.04-$0.10)

# Rate limiting
REQUEST_DELAY = 2.0  # seconds between requests to avoid rate limits


def load_api_key() -> str:
    """Load API key from FireEye/.env file."""
    env_path = FIREEYE_DIR / ".env"
    if not env_path.exists():
        print(f"ERROR: {env_path} not found")
        sys.exit(1)
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("FIREEYE_OPENROUTER_API_KEY="):
                key = line.split("=", 1)[1].strip()
                if key and key != "sk-or-v1-your-key-here":
                    return key
    print("ERROR: FIREEYE_OPENROUTER_API_KEY not set in .env")
    sys.exit(1)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

class CostTracker:
    """Track API spending."""
    def __init__(self):
        self.flux_calls = 0
        self.gemini_calls = 0

    @property
    def total_cost(self) -> float:
        return (self.flux_calls * FLUX_COST) + (self.gemini_calls * GEMINI_COST)

    def report(self) -> str:
        return (
            f"  FLUX.2 Klein calls : {self.flux_calls} x ${FLUX_COST:.3f} = ${self.flux_calls * FLUX_COST:.3f}\n"
            f"  Gemini Flash edits : {self.gemini_calls} x ~${GEMINI_COST:.3f} = ~${self.gemini_calls * GEMINI_COST:.3f}\n"
            f"  Estimated total    : ~${self.total_cost:.2f}"
        )


tracker = CostTracker()


def generate_image(api_key: str, prompt: str, retries: int = 3) -> bytes:
    """Generate an image from text prompt using FLUX.2 Klein.

    Returns raw PNG bytes.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": FLUX_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "modalities": ["image"],
        "image_config": {"aspect_ratio": "16:9"},
    }

    for attempt in range(retries):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=120)
            if resp.status_code == 429:
                wait = min(30, 5 * (attempt + 1))
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            image_url = data["choices"][0]["message"]["images"][0]["image_url"]["url"]
            # Extract base64 data
            b64_data = image_url.split(",", 1)[1]
            tracker.flux_calls += 1
            return base64.b64decode(b64_data)
        except Exception as e:
            if attempt < retries - 1:
                wait = 5 * (attempt + 1)
                print(f"    Error: {e}. Retrying in {wait}s... (attempt {attempt+1}/{retries})")
                time.sleep(wait)
            else:
                raise RuntimeError(f"Failed to generate image after {retries} attempts: {e}")


def edit_image(api_key: str, image_bytes: bytes, instruction: str, retries: int = 3) -> bytes:
    """Edit an image using Gemini 3.1 Flash Image via OpenRouter.

    Takes existing image bytes + text instruction, returns edited PNG bytes.
    """
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GEMINI_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                    {
                        "type": "text",
                        "text": f"Edit this image: {instruction}",
                    },
                ],
            }
        ],
        "modalities": ["image", "text"],
    }

    for attempt in range(retries):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=120)
            if resp.status_code == 429:
                wait = min(30, 5 * (attempt + 1))
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()

            # Try to get image from response
            choices = data.get("choices", [])
            if not choices:
                raise RuntimeError(f"No choices in response: {json.dumps(data)[:300]}")

            message = choices[0].get("message", {})

            # Check images array first
            images = message.get("images", [])
            if images:
                image_url = images[0].get("image_url", {}).get("url", "")
                if image_url:
                    b64_data = image_url.split(",", 1)[1]
                    tracker.gemini_calls += 1
                    return base64.b64decode(b64_data)

            # Check content array (alternative response format)
            content = message.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "image_url":
                            url = part.get("image_url", {}).get("url", "")
                            if url:
                                b64_data = url.split(",", 1)[1]
                                tracker.gemini_calls += 1
                                return base64.b64decode(b64_data)

            raise RuntimeError(f"No image in response. Message keys: {list(message.keys())}. Content snippet: {str(content)[:200]}")

        except Exception as e:
            if attempt < retries - 1:
                wait = 5 * (attempt + 1)
                print(f"    Error: {e}. Retrying in {wait}s... (attempt {attempt+1}/{retries})")
                time.sleep(wait)
            else:
                raise RuntimeError(f"Failed to edit image after {retries} attempts: {e}")


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

SCENARIOS = {
    "campfire_spread": {
        "title": "Campfire Spreading Beyond Stone Ring",
        "base_prompt": (
            "Photorealistic wide shot of an evening campfire scene in a forest clearing. "
            "A neat stone ring contains a small, well-controlled campfire with orange flames. "
            "Pine trees surround the clearing, dry leaves and twigs scattered on the ground "
            "near the stone ring. A couple of camping chairs and a small tent visible in "
            "background. Warm evening light, slightly overcast sky. High detail, 16:9 aspect."
        ),
        "edit_steps": [
            "The campfire flames are slightly larger and more lively, a few sparks rising. The fire is still within the stone ring but more energetic.",
            "A burning ember has landed just outside the stone ring on some dry leaves. The main campfire is bigger, flames reaching a bit higher. Very slight wisps of smoke from the ember spot.",
            "The dry leaves outside the stone ring have caught fire - a small patch (about 6 inches) is burning. The main campfire is larger with higher flames. A slight breeze is visible from the flame direction.",
            "The ground fire outside the stone ring has spread to about 2 feet. Flames lick at dry grass and twigs on the ground. Smoke is more visible. The campfire itself is roaring.",
            "The ground fire has reached the base of nearby brush and low branches. Flames are now about 1-2 feet high outside the ring. Heavy smoke rises. The original campfire is blazing.",
            "Low-hanging pine branches have caught fire. The ground fire has spread to about 6 feet from the stone ring. Multiple areas burning. Thick smoke. The scene is significantly more dangerous.",
            "A small pine tree is engulfed in flames. The ground fire has spread across much of the clearing. The tent is starting to catch sparks. Very heavy smoke, orange glow illuminating the scene.",
            "Multiple trees are on fire. The entire area around the campsite is ablaze. The tent is burning. A wall of fire and thick smoke fills much of the scene. The original stone ring is barely visible amidst the inferno.",
        ],
    },
    "welding_sparks": {
        "title": "Welding Sparks Igniting Scaffolding",
        "base_prompt": (
            "Photorealistic wide shot of an indoor construction site. Metal scaffolding "
            "with wooden planks, a welder in protective gear working on a steel beam at "
            "mid-height. Bright welding arc visible. Sawdust and wood shavings on the "
            "concrete floor below. Plastic sheeting draped over parts of the scaffolding. "
            "Industrial overhead lighting. Construction materials stacked nearby. 16:9 aspect."
        ),
        "edit_steps": [
            "Welding sparks are showering down more intensely from the welder's work. A few bright sparks land on the wooden plank below, some bounce off the scaffolding. Everything else unchanged.",
            "A small smoldering spot has appeared on the wooden scaffolding plank where sparks landed. A thin wisp of smoke rises from it. The welder continues working above, unaware.",
            "The smoldering spot on the wooden plank has grown to a small flame (about 3 inches). Smoke is more visible. Some sawdust near the flame is beginning to glow.",
            "The wooden plank is now clearly on fire with flames about 6-8 inches high. Sawdust on the floor below has caught some sparks and is smoldering. Smoke is filling the mid-level of the scaffolding.",
            "The fire has spread to the plastic sheeting on the scaffolding, which is melting and dripping burning drops. The wooden plank fire is larger. Heavy dark smoke. The welder has stopped and is backing away.",
            "Multiple wooden planks on the scaffolding are burning. The plastic sheeting is fully ablaze creating toxic black smoke. Burning debris falls to the floor. Construction materials below are starting to catch fire.",
            "The entire scaffolding section is engulfed in flames. The floor below has multiple fire spots from falling burning debris. Thick black smoke fills the upper half of the indoor space. The structure is clearly compromised.",
            "A massive indoor fire with the scaffolding structure collapsing in flames. Nearly everything visible is burning - wood, plastic, construction materials. Dense smoke fills most of the frame. Orange and red glow dominates the scene.",
        ],
    },
    "agricultural_burn": {
        "title": "Agricultural Burn Jumping Firebreak",
        "base_prompt": (
            "Photorealistic aerial-angle wide shot of an agricultural field during a "
            "controlled burn. A neat firebreak (bare dirt strip about 10 feet wide) "
            "separates a burning section of crop stubble on the left from unburned dry "
            "golden wheat field on the right. Low controlled flames on the left side. "
            "Clear blue sky, late afternoon sun. A tractor visible in the distance. "
            "Flat farmland stretching to the horizon. 16:9 aspect."
        ),
        "edit_steps": [
            "The controlled burn on the left is slightly more active, flames a bit taller. A gentle wind is blowing from left to right, visible from the smoke direction. The firebreak is still clear and intact.",
            "Wind has picked up - smoke is blowing strongly to the right across the firebreak. The flames on the left are taller and more aggressive. A few glowing embers are visible in the air above the firebreak.",
            "Burning embers have landed on the right side of the firebreak in the dry wheat. Two small spot fires (each about 1 foot) have started in the unburned wheat field. Smoke streams across the firebreak.",
            "The spot fires in the wheat field have grown to about 4-5 feet in diameter each. New embers continue to fly across. The original burn on the left is now very intense with tall flames. Wind-driven smoke fills the sky.",
            "The spot fires in the wheat field have merged into a larger fire line about 20 feet wide. Flames in the dry wheat are chest-high and moving with the wind. The firebreak has been completely bypassed. Heavy smoke.",
            "The wheat field fire has spread dramatically - a wall of flame stretches across the field. Flames are 6-8 feet tall, driven by wind. Both sides of the firebreak are now fully burning. The tractor is retreating.",
            "The entire visible wheat field is engulfed in wind-driven fire. A massive front of flames moves across the landscape. Thick columns of smoke rise hundreds of feet. The scene shows an uncontrolled wildfire.",
            "An apocalyptic agricultural wildfire. The entire landscape is burning with towering flames. Dense smoke fills most of the sky, turning it orange-brown. The fire front stretches to the horizon. Complete loss of control.",
        ],
    },
    "bbq_grease_fire": {
        "title": "BBQ Grease Fire Spreading to Furniture",
        "base_prompt": (
            "Photorealistic wide shot of a suburban backyard patio during a summer "
            "barbecue. A charcoal grill with its lid open showing glowing coals and "
            "some sausages cooking. Nearby: a wooden patio table with a checkered "
            "tablecloth, paper plates and napkins, a bottle of lighter fluid on the "
            "table edge. Wooden fence behind. Cushioned patio chairs. Green lawn. "
            "Afternoon sunlight. 16:9 aspect."
        ),
        "edit_steps": [
            "Grease from the cooking sausages has dripped onto the hot coals causing a flare-up. Flames shoot up about 12 inches above the grill grate. A small amount of smoke rises. Everything else is calm.",
            "The grease fire on the grill has intensified - flames are about 18 inches high, licking above the grill edges. Some grease has dripped down the side of the grill and is burning. Light smoke drifts toward the nearby table.",
            "The fire from the grill has grown larger. Burning grease has dripped onto the wooden deck/patio beneath the grill, starting a small fire on the ground. The tablecloth edge nearest the grill is beginning to brown and curl.",
            "The tablecloth has caught fire from heat radiation. Flames climb up the paper napkins on the table. The ground fire under the grill is growing. The lighter fluid bottle on the table is dangerously close to the spreading flames.",
            "The table is significantly on fire - tablecloth, paper plates, and napkins burning. The lighter fluid bottle has caught fire or exploded, intensifying the table fire dramatically. Flames are 3-4 feet high.",
            "The fire has spread to the cushioned patio chair nearest the table. The chair cushions are burning intensely with dark smoke. The wooden table is fully engulfed. Flames reach toward the wooden fence.",
            "The wooden fence has caught fire. Multiple pieces of patio furniture are burning. The original grill fire is lost in the larger blaze. Heavy smoke. Flames are 6+ feet high along the fence line.",
            "A major structure fire - the fence is fully ablaze, all patio furniture burning, flames threatening the house siding/overhang visible in background. Thick black smoke. The entire patio area is an inferno.",
        ],
    },
    "fireplace_embers": {
        "title": "Fireplace Embers on Carpet",
        "base_prompt": (
            "Photorealistic wide shot of a cozy living room interior at night. A brick "
            "fireplace with a low, crackling fire and glowing embers. No fireplace screen. "
            "A thick cream-colored wool rug extends from the fireplace hearth into the "
            "room. A leather armchair sits nearby with a throw blanket draped over it. "
            "Bookshelves line one wall. Warm lamplight. Holiday decorations on the mantle "
            "including a dry pine garland. 16:9 aspect."
        ),
        "edit_steps": [
            "A single glowing ember has popped out of the fireplace and landed on the cream rug about 6 inches from the hearth edge. The ember glows bright orange on the rug. A tiny wisp of smoke rises. The fireplace fire continues crackling.",
            "The ember on the rug has created a small blackened burn mark. A second ember has popped out. The rug fibers nearest the first ember are starting to smolder with a thin trail of smoke. The fire in the fireplace is still going.",
            "The rug has a visible smoldering area about the size of a fist. Thin smoke rises steadily. The rug fibers are blackening and curling. A small orange glow is visible in the smoldering spot. The room is getting slightly hazy.",
            "The smoldering rug has produced a small open flame about 3-4 inches high. The burn area has grown to about a foot in diameter. Smoke is more substantial now. The nearby armchair leg is close to the creeping fire.",
            "The rug fire has grown to about 2 feet across with flames 8-10 inches high. The fire has reached the base of the leather armchair. The throw blanket hanging off the chair is starting to catch. Room is noticeably smoky.",
            "The armchair is on fire - the throw blanket and cushion are burning. Rug fire continues to spread across the floor. Flames are now 2-3 feet high from the chair. Smoke fills the upper portion of the room. The heat is affecting nearby items.",
            "The fire has spread to the bookshelf - books and wood shelving are catching fire. The dry pine garland on the mantle has ignited. Multiple points of fire around the room. Heavy smoke, visibility reduced. Flames reaching the ceiling.",
            "The living room is fully engulfed. The bookshelf is a wall of flame, the ceiling is on fire, furniture is burning. Dense black smoke fills most of the room. Only the orange glow of widespread fire is visible through the smoke.",
        ],
    },
}


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_scenario(api_key: str, scene_id: str, scenario: dict) -> bool:
    """Generate all frames for a single scenario.

    Returns True if all frames were generated successfully.
    """
    scene_dir = OUTPUT_DIR / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    title = scenario["title"]
    base_prompt = scenario["base_prompt"]
    edit_steps = scenario["edit_steps"]
    total_frames = 1 + len(edit_steps)  # frame 1 (base) + edit frames

    print(f"\n{'='*60}")
    print(f"Scene: {title}")
    print(f"Frames: {total_frames} | Dir: {scene_dir.relative_to(SCRIPT_DIR)}")
    print(f"{'='*60}")

    # --- Frame 1: Generate base image with FLUX ---
    frame_path = scene_dir / "frame_01.png"
    if frame_path.exists():
        print(f"  [1/{total_frames}] frame_01.png already exists, loading...")
        current_image = frame_path.read_bytes()
    else:
        print(f"  [1/{total_frames}] Generating base scene with FLUX.2 Klein...")
        try:
            current_image = generate_image(api_key, base_prompt)
            frame_path.write_bytes(current_image)
            print(f"  [1/{total_frames}] Saved frame_01.png ({len(current_image):,} bytes)")
        except Exception as e:
            print(f"  [1/{total_frames}] FAILED: {e}")
            return False
        time.sleep(REQUEST_DELAY)

    # --- Frames 2-N: Edit progressively with Gemini ---
    for i, instruction in enumerate(edit_steps, start=2):
        frame_num = f"{i:02d}"
        frame_path = scene_dir / f"frame_{frame_num}.png"

        if frame_path.exists():
            print(f"  [{i}/{total_frames}] frame_{frame_num}.png already exists, loading...")
            current_image = frame_path.read_bytes()
            continue

        print(f"  [{i}/{total_frames}] Editing: {instruction[:70]}...")
        try:
            edited = edit_image(api_key, current_image, instruction)
            frame_path.write_bytes(edited)
            current_image = edited
            print(f"  [{i}/{total_frames}] Saved frame_{frame_num}.png ({len(edited):,} bytes)")
        except Exception as e:
            print(f"  [{i}/{total_frames}] FAILED: {e}")
            # On failure, try one more time with the base image instead of the chain
            print(f"  [{i}/{total_frames}] Attempting fallback: generating fresh with FLUX...")
            try:
                fallback_prompt = (
                    f"{base_prompt} "
                    f"However, the scene has changed dramatically: {instruction}"
                )
                fallback = generate_image(api_key, fallback_prompt)
                frame_path.write_bytes(fallback)
                current_image = fallback
                print(f"  [{i}/{total_frames}] Fallback saved frame_{frame_num}.png")
            except Exception as e2:
                print(f"  [{i}/{total_frames}] Fallback also FAILED: {e2}")
                return False
        time.sleep(REQUEST_DELAY)

    return True


def main():
    print("=" * 60)
    print("FireEye Escalation Frame Generator")
    print("Using OpenRouter API (FLUX.2 Klein + Gemini 3.1 Flash Image)")
    print("=" * 60)

    api_key = load_api_key()
    print(f"API key: ...{api_key[-8:]}")
    print(f"Output:  {OUTPUT_DIR}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for scene_id, scenario in SCENARIOS.items():
        success = generate_scenario(api_key, scene_id, scenario)
        results[scene_id] = success

    # --- Summary ---
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)

    for scene_id, success in results.items():
        status = "OK" if success else "PARTIAL/FAILED"
        scene_dir = OUTPUT_DIR / scene_id
        n_frames = len(list(scene_dir.glob("frame_*.png"))) if scene_dir.exists() else 0
        print(f"  {scene_id:<25} {status:<15} ({n_frames} frames)")

    print()
    print("Cost breakdown:")
    print(tracker.report())
    print()

    total_frames = sum(
        len(list((OUTPUT_DIR / sid).glob("frame_*.png")))
        for sid in SCENARIOS
        if (OUTPUT_DIR / sid).exists()
    )
    print(f"Total frames generated: {total_frames}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
