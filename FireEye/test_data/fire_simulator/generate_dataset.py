#!/usr/bin/env python3
"""
Fire Simulation Dataset Generator — CLI Tool

Generates frame sequences of fire spreading in various scenes.
Each sequence is saved as a series of PNG images suitable for
FireEye fire detection analysis.

Usage:
    python3 generate_dataset.py --scene indoor --frames 30 --output output/indoor_01
    python3 generate_dataset.py --scene outdoor --frames 30 --seed 123 --wind-speed 0.5
    python3 generate_dataset.py --scene urban --frames 30 --cell-size 8
    python3 generate_dataset.py --all  # Generate all default scenarios

Parameters are fully configurable for deterministic, reproducible output.
"""

import argparse
import os
import sys
import json
import time

# Ensure imports work regardless of how this script is invoked
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from simulator import (
    FireSimulator, CellState, FuelType,
    create_scene, SCENE_CREATORS
)
from renderer import render_sequence, render_frame, FireRenderer


def generate_sequence(scene_type: str, output_dir: str,
                      num_frames: int = 30, seed: int = 42,
                      cell_size: int = 6, wind_direction: float = None,
                      wind_speed: float = None, fuel_density: float = None,
                      width: int = None, height: int = None,
                      steps_per_frame: int = 1,
                      show_glow: bool = True,
                      ignition_point: tuple = None):
    """Generate a single fire simulation sequence."""

    # Build kwargs for scene creation
    kwargs = {'seed': seed}
    if wind_direction is not None:
        kwargs['wind_direction'] = wind_direction
    if wind_speed is not None:
        kwargs['wind_speed'] = wind_speed
    if fuel_density is not None:
        kwargs['fuel_density'] = fuel_density
    if ignition_point is not None:
        kwargs['ignition_point'] = ignition_point

    # Scene dimensions
    defaults = {
        'indoor': (80, 60),
        'outdoor': (100, 80),
        'urban': (100, 80),
    }
    default_w, default_h = defaults.get(scene_type, (80, 60))
    kwargs['width'] = width or default_w
    kwargs['height'] = height or default_h

    print(f"  Creating {scene_type} scene (seed={seed}, "
          f"{kwargs['width']}x{kwargs['height']})...")
    sim = create_scene(scene_type, **kwargs)

    stats_before = sim.get_stats()
    print(f"  Initial state: {stats_before['fuel']} fuel cells, "
          f"{stats_before['burning']} burning")

    print(f"  Wind: {sim.wind.direction:.0f} deg, speed={sim.wind.speed:.2f}")
    print(f"  Rendering {num_frames} frames (steps_per_frame={steps_per_frame})...")

    t0 = time.time()
    label = scene_type.upper()
    paths = render_sequence(
        sim, output_dir, num_frames=num_frames,
        cell_size=cell_size, label=label,
        show_hud=True, show_glow=show_glow,
        steps_per_frame=steps_per_frame
    )
    elapsed = time.time() - t0

    stats_after = sim.get_stats()
    print(f"  Done in {elapsed:.1f}s. Final: {stats_after['burning']} burning, "
          f"{stats_after['burned']} burned, {stats_after['fuel']} fuel remaining")
    print(f"  Output: {output_dir}/ ({len(paths)} frames)")

    # Save metadata
    metadata = {
        'scene_type': scene_type,
        'seed': seed,
        'width': kwargs['width'],
        'height': kwargs['height'],
        'num_frames': num_frames,
        'cell_size': cell_size,
        'steps_per_frame': steps_per_frame,
        'wind_direction': sim.wind.direction,
        'wind_speed': sim.wind.speed,
        'stats_initial': stats_before,
        'stats_final': stats_after,
        'frames': [os.path.basename(p) for p in paths],
    }
    meta_path = os.path.join(output_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return paths


def generate_all_defaults(base_dir: str, num_frames: int = 30,
                          cell_size: int = 6):
    """Generate default scenarios for all scene types."""
    scenarios = [
        {
            'name': 'indoor_room_fire',
            'scene_type': 'indoor',
            'seed': 42,
            'description': 'Room fire starting near furniture, minimal wind',
            'wind_speed': 0.05,
            'wind_direction': 90.0,
            'steps_per_frame': 2,
        },
        {
            'name': 'outdoor_wildfire',
            'scene_type': 'outdoor',
            'seed': 137,
            'description': 'Wildfire in field with moderate wind from the west',
            'wind_speed': 0.55,
            'wind_direction': 270.0,
            'fuel_density': 0.7,
            'steps_per_frame': 1,
        },
        {
            'name': 'urban_building_fire',
            'scene_type': 'urban',
            'seed': 256,
            'description': 'Fire in an urban building spreading to nearby structures',
            'wind_speed': 0.25,
            'wind_direction': 180.0,
            'steps_per_frame': 2,
        },
    ]

    all_paths = []
    for i, scenario in enumerate(scenarios):
        name = scenario.pop('name')
        desc = scenario.pop('description')
        scene_type = scenario.pop('scene_type')

        print(f"\n{'='*60}")
        print(f"Scenario {i+1}/{len(scenarios)}: {name}")
        print(f"  {desc}")
        print(f"{'='*60}")

        output_dir = os.path.join(base_dir, name)
        paths = generate_sequence(
            scene_type=scene_type,
            output_dir=output_dir,
            num_frames=num_frames,
            cell_size=cell_size,
            **scenario,
        )
        all_paths.extend(paths)

    return all_paths


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic fire simulation frame sequences',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --all                          Generate all default scenarios
  %(prog)s --scene indoor --frames 30     Generate indoor fire sequence
  %(prog)s --scene outdoor --seed 99      Custom outdoor with specific seed
  %(prog)s --scene urban --wind-speed 0.6 --wind-direction 180
        """
    )

    parser.add_argument('--all', action='store_true',
                        help='Generate all default scenarios')
    parser.add_argument('--scene', type=str, choices=list(SCENE_CREATORS.keys()),
                        help='Scene type to generate')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: output/<scene_type>)')
    parser.add_argument('--frames', type=int, default=30,
                        help='Number of frames to generate (default: 30)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--cell-size', type=int, default=6,
                        help='Pixels per grid cell (default: 6)')
    parser.add_argument('--wind-direction', type=float, default=None,
                        help='Wind direction in degrees (0=N, 90=E, 180=S, 270=W)')
    parser.add_argument('--wind-speed', type=float, default=None,
                        help='Wind speed 0.0-1.0 (default: random per scene)')
    parser.add_argument('--fuel-density', type=float, default=None,
                        help='Fuel density 0.0-1.0 (outdoor scenes)')
    parser.add_argument('--width', type=int, default=None,
                        help='Grid width (default: scene-dependent)')
    parser.add_argument('--height', type=int, default=None,
                        help='Grid height (default: scene-dependent)')
    parser.add_argument('--steps-per-frame', type=int, default=1,
                        help='Simulation steps between rendered frames (default: 1)')
    parser.add_argument('--no-glow', action='store_true',
                        help='Disable fire glow effects (faster rendering)')

    args = parser.parse_args()

    # Determine base output directory
    base_dir = os.path.join(SCRIPT_DIR, 'output')

    if args.all:
        print("Generating all default fire simulation scenarios...")
        print(f"Output directory: {base_dir}")
        paths = generate_all_defaults(base_dir, num_frames=args.frames,
                                      cell_size=args.cell_size)
        print(f"\n{'='*60}")
        print(f"Generated {len(paths)} total frames across all scenarios.")
        print(f"Output: {base_dir}/")

    elif args.scene:
        output_dir = args.output or os.path.join(base_dir, args.scene)
        print(f"Generating {args.scene} fire simulation...")
        print(f"Output directory: {output_dir}")
        paths = generate_sequence(
            scene_type=args.scene,
            output_dir=output_dir,
            num_frames=args.frames,
            seed=args.seed,
            cell_size=args.cell_size,
            wind_direction=args.wind_direction,
            wind_speed=args.wind_speed,
            fuel_density=args.fuel_density,
            width=args.width,
            height=args.height,
            steps_per_frame=args.steps_per_frame,
            show_glow=not args.no_glow,
        )
        print(f"\nGenerated {len(paths)} frames.")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
