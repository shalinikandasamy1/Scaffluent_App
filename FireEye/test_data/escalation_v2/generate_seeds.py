#!/usr/bin/env python3
"""Generate seed image pairs for FLF2V escalation video generation.

Stage 1: Generate 'calm start' images with FLUX.2 Klein ($0.014/img)
Stage 2: Edit start images into 'fire end' variants using Gemini Flash Image ($0.04/img)
Estimated total cost: ~$0.27 for 5 scene pairs
"""

import os
import sys
import json
import base64
import requests
from pathlib import Path

# Load API key
env_path = Path(__file__).parent.parent.parent / ".env"
api_key = None
with open(env_path) as f:
    for line in f:
        if line.startswith("FIREEYE_OPENROUTER_API_KEY="):
            api_key = line.strip().split("=", 1)[1]
            break

if not api_key:
    print("ERROR: FIREEYE_OPENROUTER_API_KEY not found in .env")
    sys.exit(1)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

OUTPUT_DIR = Path(__file__).parent / "seed_images"
OUTPUT_DIR.mkdir(exist_ok=True)

# Scene definitions: (id, start_prompt, end_prompt)
SCENES = [
    (
        "01_campfire",
        "A small controlled campfire inside a stone ring in a clearing, surrounded by dry grass and scattered leaves. Evening light, calm atmosphere, no wind. Wide shot from about 3 meters away.",
        "The same campfire has spread beyond the stone ring, flames catching the dry grass and leaves around it. Fire spreading outward in all directions, smoke rising. Emergency situation, wide shot.",
    ),
    (
        "02_welding",
        "A welder in protective gear working on a wooden scaffolding structure inside a workshop. Small bright sparks flying from the welding point. Tools and wood pieces scattered nearby. Industrial interior, overhead lighting.",
        "The wooden scaffolding has caught fire from welding sparks. Flames climbing up the wood structure, the welder stepping back in alarm. Smoke filling the workshop ceiling. Same camera angle, emergency situation.",
    ),
    (
        "03_agri_burn",
        "A controlled agricultural burn in a flat field with a clear dirt firebreak. Low flames burning crop stubble in one section, unburned field on the other side of the break. Rural landscape, daytime, slight breeze.",
        "The agricultural burn has jumped the firebreak. Flames spreading across both sides of the dirt path into the unburned field and toward trees at the edge. Thick smoke, growing wildfire, same wide shot.",
    ),
    (
        "04_bbq",
        "A backyard BBQ scene with a charcoal grill cooking hamburgers on a wooden deck. Patio furniture, cushions, and a tablecloth nearby. Suburban backyard, afternoon sunlight, normal grilling.",
        "The BBQ grill has a grease fire that has spread to the nearby tablecloth and deck furniture cushions. Flames engulfing the patio area, chairs on fire. Same backyard scene, emergency situation.",
    ),
    (
        "05_fireplace",
        "A cozy living room with a stone fireplace, logs burning gently inside. Plush carpet in front of the hearth, wooden floor, a blanket draped on a nearby armchair. Warm interior lighting, peaceful evening.",
        "Embers and sparks from the fireplace have scattered onto the carpet and blanket. Small flames spreading on the carpet near the hearth, smoke rising. The armchair blanket starting to smolder. Same living room, alarming situation.",
    ),
]


def generate_image(prompt, model="black-forest-labs/flux.2-klein-4b", aspect_ratio="16:9"):
    """Generate an image using OpenRouter API."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "modalities": ["image"],
        "image_config": {"aspect_ratio": aspect_ratio},
    }

    print(f"  Generating with {model}...")
    resp = requests.post(OPENROUTER_URL, headers=HEADERS, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    # Extract base64 image from response
    choices = data.get("choices", [])
    if not choices:
        print(f"  ERROR: No choices in response: {json.dumps(data, indent=2)[:500]}")
        return None

    message = choices[0].get("message", {})

    # Check for images array
    images = message.get("images", [])
    if images:
        url = images[0].get("image_url", {}).get("url", "")
        if url.startswith("data:image/"):
            # Extract base64 data after the comma
            b64_data = url.split(",", 1)[1]
            return base64.b64decode(b64_data)

    # Check content for base64 image
    content = message.get("content", "")
    if content and content.startswith("data:image/"):
        b64_data = content.split(",", 1)[1]
        return base64.b64decode(b64_data)

    print(f"  ERROR: Could not extract image from response")
    print(f"  Response keys: {list(data.keys())}")
    print(f"  Message keys: {list(message.keys())}")
    print(f"  Content preview: {str(content)[:200]}")
    return None


def edit_image_with_gemini(source_image_bytes, edit_prompt):
    """Edit an image using Gemini Flash Image via OpenRouter."""
    # Encode source image as base64
    b64_source = base64.b64encode(source_image_bytes).decode("utf-8")

    payload = {
        "model": "google/gemini-2.5-flash-image-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_source}"},
                    },
                    {
                        "type": "text",
                        "text": f"Edit this image: {edit_prompt}. Keep the same camera angle, composition, and scene layout. Only add the fire/emergency elements described.",
                    },
                ],
            }
        ],
        "modalities": ["image", "text"],
        "image_config": {"aspect_ratio": "16:9"},
    }

    print(f"  Editing with Gemini Flash Image...")
    resp = requests.post(OPENROUTER_URL, headers=HEADERS, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    choices = data.get("choices", [])
    if not choices:
        print(f"  ERROR: No choices in response: {json.dumps(data, indent=2)[:500]}")
        return None

    message = choices[0].get("message", {})
    images = message.get("images", [])
    if images:
        url = images[0].get("image_url", {}).get("url", "")
        if url.startswith("data:image/"):
            b64_data = url.split(",", 1)[1]
            return base64.b64decode(b64_data)

    content = message.get("content", "")
    if content and "data:image/" in content:
        # Try to extract base64 from content
        for part in content.split("data:image/"):
            if ";base64," in part:
                b64_data = part.split(";base64,", 1)[1].split('"')[0].split("'")[0]
                return base64.b64decode(b64_data)

    print(f"  ERROR: Could not extract edited image")
    print(f"  Message keys: {list(message.keys())}")
    print(f"  Content preview: {str(content)[:300]}")
    return None


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode in ("start", "all"):
        print("=== Stage 1: Generating START images (FLUX.2 Klein) ===")
        print(f"Cost: ~$0.014 × {len(SCENES)} = ~${0.014 * len(SCENES):.3f}")
        print()

        for scene_id, start_prompt, _ in SCENES:
            out_path = OUTPUT_DIR / f"{scene_id}_start.png"
            if out_path.exists():
                print(f"[SKIP] {scene_id}_start.png already exists")
                continue

            print(f"[GEN] {scene_id}_start.png")
            img_data = generate_image(start_prompt)
            if img_data:
                out_path.write_bytes(img_data)
                print(f"  Saved: {out_path} ({len(img_data)} bytes)")
            else:
                print(f"  FAILED: {scene_id}_start.png")
            print()

    if mode in ("end", "all"):
        print("=== Stage 2: Editing START images into END images (Gemini 3.1 Flash Image) ===")
        print(f"Estimated cost: ~$0.04-0.10 × {len(SCENES)} scenes")
        print()

        for scene_id, _, end_prompt in SCENES:
            start_path = OUTPUT_DIR / f"{scene_id}_start.png"
            out_path = OUTPUT_DIR / f"{scene_id}_end.png"

            if out_path.exists():
                print(f"[SKIP] {scene_id}_end.png already exists")
                continue

            if not start_path.exists():
                print(f"[ERROR] {scene_id}_start.png not found, cannot edit")
                continue

            # Edit the start image with Gemini 3.1 Flash Image
            print(f"[EDIT] {scene_id}_end.png")
            source_bytes = start_path.read_bytes()
            b64_source = base64.b64encode(source_bytes).decode("utf-8")

            edit_payload = {
                "model": "google/gemini-3.1-flash-image-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64_source}"},
                            },
                            {
                                "type": "text",
                                "text": (
                                    f"Edit this image: {end_prompt} "
                                    "Keep the exact same camera angle, composition, and scene layout. "
                                    "Only add the fire/emergency elements described."
                                ),
                            },
                        ],
                    }
                ],
                "modalities": ["image", "text"],
            }

            print(f"  Editing with google/gemini-3.1-flash-image-preview...")
            resp = requests.post(OPENROUTER_URL, headers=HEADERS, json=edit_payload, timeout=120)
            if resp.status_code != 200:
                print(f"  ERROR: HTTP {resp.status_code}: {resp.text[:300]}")
                continue

            data = resp.json()
            message = data.get("choices", [{}])[0].get("message", {})
            images = message.get("images", [])
            img_data = None
            if images:
                url = images[0].get("image_url", {}).get("url", "")
                if url.startswith("data:image/"):
                    b64_data = url.split(",", 1)[1]
                    img_data = base64.b64decode(b64_data)

            if img_data:
                out_path.write_bytes(img_data)
                print(f"  Saved: {out_path} ({len(img_data)} bytes)")
            else:
                print(f"  FAILED: {scene_id}_end.png")
            print()

    # Summary
    print("=== Summary ===")
    for scene_id, _, _ in SCENES:
        start_exists = (OUTPUT_DIR / f"{scene_id}_start.png").exists()
        end_exists = (OUTPUT_DIR / f"{scene_id}_end.png").exists()
        status = "READY" if start_exists and end_exists else "INCOMPLETE"
        print(f"  {scene_id}: start={'OK' if start_exists else 'MISSING'} end={'OK' if end_exists else 'MISSING'} [{status}]")


if __name__ == "__main__":
    main()
