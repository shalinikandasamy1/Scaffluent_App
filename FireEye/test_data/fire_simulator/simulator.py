"""
Fire Simulation Engine — Cellular Automaton Model

A parametric, deterministic fire spread simulator that models:
- Cell states: EMPTY, FUEL, BURNING, BURNED, SMOKE, WALL
- Wind-driven fire spread with configurable direction and speed
- Multiple fuel types with different burn rates and ignitability
- Scene templates (indoor, outdoor, urban)
- Smoke generation and drift

The simulation runs on a 2D grid where each cell tracks:
- state: current cell state
- fuel_level: remaining fuel (0.0 - 1.0)
- burn_timer: how long the cell has been burning
- temperature: radiated heat affecting nearby cells
- fuel_type: what kind of material (grass, wood, fabric, concrete)
"""

import random
import math
from enum import IntEnum
from typing import List, Tuple, Optional, Dict


class CellState(IntEnum):
    EMPTY = 0       # Nothing — bare ground, air
    FUEL = 1        # Unburned combustible material
    BURNING = 2     # Actively on fire
    BURNED = 3      # Consumed — ash/char remains
    SMOKE = 4       # Smoke particle (drifts with wind)
    WALL = 5        # Non-combustible structure
    EMBER = 6       # Glowing embers (post-burn, can reignite)


class FuelType(IntEnum):
    NONE = 0
    GRASS = 1       # Burns fast, low fuel
    WOOD = 2        # Burns moderately, high fuel
    FABRIC = 3      # Burns fast, moderate fuel
    CONCRETE = 4    # Does not burn (structural)
    BRUSH = 5       # Burns fast, moderate fuel (outdoor)


# Fuel type properties: (ignitability, burn_rate, max_fuel, smoke_output)
# ignitability: probability multiplier for catching fire
# burn_rate: how fast fuel is consumed per tick
# max_fuel: maximum fuel level
# smoke_output: how much smoke is generated while burning
FUEL_PROPERTIES = {
    FuelType.NONE:     (0.0, 0.0, 0.0, 0.0),
    FuelType.GRASS:    (0.8, 0.15, 0.4, 0.3),
    FuelType.WOOD:     (0.4, 0.05, 1.0, 0.6),
    FuelType.FABRIC:   (0.9, 0.12, 0.5, 0.8),
    FuelType.CONCRETE: (0.0, 0.0, 0.0, 0.0),
    FuelType.BRUSH:    (0.7, 0.10, 0.6, 0.5),
}


class Cell:
    """Represents a single cell in the fire simulation grid."""
    __slots__ = ['state', 'fuel_level', 'fuel_type', 'burn_timer',
                 'temperature', 'smoke_density', 'elevation']

    def __init__(self, state=CellState.EMPTY, fuel_type=FuelType.NONE,
                 fuel_level=0.0, elevation=0.0):
        self.state = state
        self.fuel_type = fuel_type
        self.fuel_level = fuel_level
        self.burn_timer = 0.0
        self.temperature = 0.0   # Ambient = 0, fire ~1.0
        self.smoke_density = 0.0
        self.elevation = elevation

    def copy(self):
        c = Cell(self.state, self.fuel_type, self.fuel_level, self.elevation)
        c.burn_timer = self.burn_timer
        c.temperature = self.temperature
        c.smoke_density = self.smoke_density
        return c


class WindModel:
    """Wind direction and speed affecting fire spread and smoke drift."""

    def __init__(self, direction: float = 0.0, speed: float = 0.0):
        """
        direction: angle in degrees (0=North/up, 90=East/right, etc.)
        speed: 0.0 (calm) to 1.0 (strong wind)
        """
        self.direction = direction
        self.speed = speed
        # Precompute directional components (dx, dy in grid coords)
        rad = math.radians(direction)
        self.dx = math.sin(rad)  # East component
        self.dy = -math.cos(rad)  # North is negative Y in grid coords

    def spread_probability_modifier(self, from_r, from_c, to_r, to_c):
        """
        Calculate how wind affects fire spread probability from one cell
        to another. Returns a multiplier (>1 means wind pushes fire that way).
        """
        if self.speed < 0.01:
            return 1.0

        # Direction from source to target
        dr = to_r - from_r
        dc = to_c - from_c
        dist = math.sqrt(dr * dr + dc * dc)
        if dist < 0.01:
            return 1.0

        # Dot product: how aligned is this spread direction with wind?
        alignment = (dc * self.dx + dr * self.dy) / dist
        # alignment: -1 (against wind) to +1 (with wind)

        # Wind multiplier: 1.0 at calm, up to 3x with wind, down to 0.2x against
        modifier = 1.0 + alignment * self.speed * 2.0
        return max(0.1, modifier)


class FireSimulator:
    """
    Core fire simulation using cellular automaton rules.

    The simulation proceeds in discrete timesteps. At each step:
    1. Burning cells radiate heat to neighbors
    2. Fuel cells adjacent to fire may ignite (probability-based)
    3. Burning cells consume fuel and may burn out
    4. Smoke is generated and drifts with wind
    5. Embers cool down
    """

    def __init__(self, width: int, height: int, seed: int = 42):
        self.width = width
        self.height = height
        self.rng = random.Random(seed)
        self.grid: List[List[Cell]] = [
            [Cell() for _ in range(width)] for _ in range(height)
        ]
        self.wind = WindModel(0.0, 0.0)
        self.timestep = 0
        self.history: List[List[List[Cell]]] = []

        # Simulation parameters (tunable)
        self.base_ignition_prob = 0.35    # Base chance of fuel catching fire
        self.heat_radius = 2.0            # How far heat radiates
        self.ember_duration = 5           # How long embers glow
        self.smoke_decay = 0.08           # How fast smoke fades per tick
        self.smoke_generation = 0.4       # Smoke per burning tick
        self.heat_decay = 0.15            # Temperature cooling per tick

    def set_wind(self, direction: float, speed: float):
        """Set wind conditions. Direction in degrees, speed 0.0-1.0."""
        self.wind = WindModel(direction, speed)

    def set_cell(self, row: int, col: int, state: CellState,
                 fuel_type: FuelType = FuelType.NONE, fuel_level: float = -1.0):
        """Set a cell's state and fuel properties."""
        if 0 <= row < self.height and 0 <= col < self.width:
            cell = self.grid[row][col]
            cell.state = state
            cell.fuel_type = fuel_type
            if fuel_level < 0:
                # Use default for fuel type
                cell.fuel_level = FUEL_PROPERTIES[fuel_type][2]
            else:
                cell.fuel_level = fuel_level
            if state == CellState.BURNING:
                cell.temperature = 1.0

    def ignite(self, row: int, col: int):
        """Start a fire at the specified cell."""
        cell = self.grid[row][col]
        if cell.state == CellState.FUEL:
            cell.state = CellState.BURNING
            cell.temperature = 1.0
            cell.burn_timer = 0.0

    def fill_rect(self, r1: int, c1: int, r2: int, c2: int,
                  state: CellState, fuel_type: FuelType = FuelType.NONE):
        """Fill a rectangular region with a given state."""
        for r in range(max(0, r1), min(self.height, r2 + 1)):
            for c in range(max(0, c1), min(self.width, c2 + 1)):
                self.set_cell(r, c, state, fuel_type)

    def fill_circle(self, cr: int, cc: int, radius: float,
                    state: CellState, fuel_type: FuelType = FuelType.NONE):
        """Fill a circular region."""
        for r in range(max(0, int(cr - radius)), min(self.height, int(cr + radius) + 1)):
            for c in range(max(0, int(cc - radius)), min(self.width, int(cc + radius) + 1)):
                if math.sqrt((r - cr) ** 2 + (c - cc) ** 2) <= radius:
                    self.set_cell(r, c, state, fuel_type)

    def scatter_fuel(self, r1: int, c1: int, r2: int, c2: int,
                     fuel_type: FuelType, density: float = 0.7):
        """Scatter fuel cells randomly in a region with given density."""
        for r in range(max(0, r1), min(self.height, r2 + 1)):
            for c in range(max(0, c1), min(self.width, c2 + 1)):
                if self.grid[r][c].state == CellState.EMPTY:
                    if self.rng.random() < density:
                        self.set_cell(r, c, CellState.FUEL, fuel_type)

    def _get_neighbors(self, row: int, col: int, radius: int = 1):
        """Get neighboring cells within given radius (Moore neighborhood)."""
        neighbors = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    dist = math.sqrt(dr * dr + dc * dc)
                    neighbors.append((nr, nc, dist))
        return neighbors

    def step(self):
        """Advance simulation by one timestep."""
        # Create a snapshot for the next state
        new_grid = [[self.grid[r][c].copy() for c in range(self.width)]
                     for r in range(self.height)]

        for r in range(self.height):
            for c in range(self.width):
                cell = self.grid[r][c]
                new_cell = new_grid[r][c]

                if cell.state == CellState.BURNING:
                    self._update_burning(r, c, cell, new_cell, new_grid)
                elif cell.state == CellState.FUEL:
                    self._update_fuel(r, c, cell, new_cell)
                elif cell.state == CellState.EMBER:
                    self._update_ember(r, c, cell, new_cell)

                # Smoke dynamics for all cells
                self._update_smoke(r, c, cell, new_cell, new_grid)

                # Temperature decay
                if new_cell.state != CellState.BURNING:
                    new_cell.temperature = max(0.0, new_cell.temperature - self.heat_decay)

        self.grid = new_grid
        self.timestep += 1

    def _update_burning(self, r, c, cell, new_cell, new_grid):
        """Update a burning cell: consume fuel, generate smoke, possibly burn out."""
        props = FUEL_PROPERTIES[cell.fuel_type]
        burn_rate = props[1]
        smoke_output = props[3]

        # Consume fuel
        new_cell.fuel_level -= burn_rate
        new_cell.burn_timer += 1
        new_cell.temperature = 1.0

        # Generate smoke
        new_cell.smoke_density = min(1.0, new_cell.smoke_density + smoke_output * self.smoke_generation)

        # Radiate heat to neighbors
        for nr, nc, dist in self._get_neighbors(r, c, radius=2):
            if dist <= self.heat_radius:
                heat_contribution = 0.3 / (dist * dist)
                # Wind pushes heat
                wind_mod = self.wind.spread_probability_modifier(r, c, nr, nc)
                new_grid[nr][nc].temperature = min(
                    1.0,
                    new_grid[nr][nc].temperature + heat_contribution * wind_mod
                )

        # Check if fuel is exhausted
        if new_cell.fuel_level <= 0:
            new_cell.fuel_level = 0.0
            new_cell.state = CellState.EMBER
            new_cell.burn_timer = 0
            new_cell.temperature = 0.6

    def _update_fuel(self, r, c, cell, new_cell):
        """Update a fuel cell: check if it should ignite from nearby fire."""
        props = FUEL_PROPERTIES[cell.fuel_type]
        ignitability = props[0]

        if ignitability <= 0:
            return

        # Check for nearby burning cells
        total_ignition_pressure = 0.0
        for nr, nc, dist in self._get_neighbors(r, c, radius=2):
            neighbor = self.grid[nr][nc]
            if neighbor.state == CellState.BURNING:
                # Base ignition from proximity
                proximity_factor = 1.0 / (dist * dist)
                # Wind effect
                wind_mod = self.wind.spread_probability_modifier(nr, nc, r, c)
                total_ignition_pressure += proximity_factor * wind_mod

            elif neighbor.state == CellState.EMBER and dist <= 1.5:
                # Embers can also ignite (weakly)
                total_ignition_pressure += 0.1 / dist

        # Also factor in ambient temperature (from heat radiation)
        total_ignition_pressure += cell.temperature * 0.5

        # Calculate final ignition probability
        ignition_prob = self.base_ignition_prob * ignitability * total_ignition_pressure
        ignition_prob = min(0.95, ignition_prob)  # Cap at 95%

        if self.rng.random() < ignition_prob:
            new_cell.state = CellState.BURNING
            new_cell.temperature = 1.0
            new_cell.burn_timer = 0

    def _update_ember(self, r, c, cell, new_cell):
        """Update ember cell: cool down and eventually become burned."""
        new_cell.burn_timer += 1
        new_cell.temperature = max(0.0, new_cell.temperature - 0.08)
        new_cell.smoke_density = max(0.0, new_cell.smoke_density - 0.03)

        if new_cell.burn_timer >= self.ember_duration:
            new_cell.state = CellState.BURNED
            new_cell.temperature = 0.0

    def _update_smoke(self, r, c, cell, new_cell, new_grid):
        """Update smoke: decay and drift with wind."""
        if new_cell.smoke_density > 0:
            new_cell.smoke_density = max(0.0, new_cell.smoke_density - self.smoke_decay)

            # Drift smoke with wind
            if self.wind.speed > 0.05:
                drift_r = int(round(r + self.wind.dy * 0.5))
                drift_c = int(round(c + self.wind.dx * 0.5))
                if 0 <= drift_r < self.height and 0 <= drift_c < self.width:
                    transfer = new_cell.smoke_density * self.wind.speed * 0.3
                    new_grid[drift_r][drift_c].smoke_density = min(
                        1.0,
                        new_grid[drift_r][drift_c].smoke_density + transfer
                    )

    def snapshot(self) -> List[List[Cell]]:
        """Return a deep copy of the current grid state."""
        return [[self.grid[r][c].copy() for c in range(self.width)]
                for r in range(self.height)]

    def save_snapshot(self):
        """Save the current state to history."""
        self.history.append(self.snapshot())

    def get_stats(self) -> Dict:
        """Return statistics about the current simulation state."""
        counts = {s: 0 for s in CellState}
        total_fuel = 0.0
        total_temp = 0.0
        for r in range(self.height):
            for c in range(self.width):
                cell = self.grid[r][c]
                counts[cell.state] += 1
                total_fuel += cell.fuel_level
                total_temp += cell.temperature
        return {
            'timestep': self.timestep,
            'empty': counts[CellState.EMPTY],
            'fuel': counts[CellState.FUEL],
            'burning': counts[CellState.BURNING],
            'burned': counts[CellState.BURNED],
            'smoke': counts.get(CellState.SMOKE, 0),
            'ember': counts[CellState.EMBER],
            'wall': counts[CellState.WALL],
            'total_fuel': round(total_fuel, 2),
            'avg_temperature': round(total_temp / (self.width * self.height), 4),
        }


# ── Scene Generators ─────────────────────────────────────────────────

def create_indoor_scene(width: int = 80, height: int = 60,
                        seed: int = 42, **kwargs) -> FireSimulator:
    """
    Indoor room scene: walls, furniture (wood/fabric), door openings.
    Fire typically starts near a piece of furniture or electrical source.
    """
    sim = FireSimulator(width, height, seed=seed)
    rng = sim.rng

    # Room walls
    sim.fill_rect(0, 0, 0, width - 1, CellState.WALL, FuelType.CONCRETE)    # Top
    sim.fill_rect(height - 1, 0, height - 1, width - 1, CellState.WALL, FuelType.CONCRETE)  # Bottom
    sim.fill_rect(0, 0, height - 1, 0, CellState.WALL, FuelType.CONCRETE)    # Left
    sim.fill_rect(0, width - 1, height - 1, width - 1, CellState.WALL, FuelType.CONCRETE)  # Right

    # Wooden floor — moderate fuel across most of the room (acts as connective tissue)
    sim.scatter_fuel(2, 2, height - 3, width - 3, FuelType.WOOD, density=0.35)

    # Carpet / rug areas (fabric — highly flammable, creates fire highways)
    num_rugs = rng.randint(2, 4)
    for _ in range(num_rugs):
        rw = rng.randint(8, 18)
        rh = rng.randint(5, 10)
        rr = rng.randint(3, height - rh - 3)
        rc = rng.randint(3, width - rw - 3)
        sim.fill_rect(rr, rc, rr + rh, rc + rw, CellState.FUEL, FuelType.FABRIC)

    # Furniture pieces (randomly placed)
    num_furniture = kwargs.get('num_furniture', 7)
    furniture_positions = []
    for _ in range(num_furniture):
        fw = rng.randint(3, 8)
        fh = rng.randint(3, 6)
        fr = rng.randint(3, height - fh - 3)
        fc = rng.randint(3, width - fw - 3)

        # Alternate between wood and fabric furniture
        ftype = rng.choice([FuelType.WOOD, FuelType.FABRIC])
        sim.fill_rect(fr, fc, fr + fh, fc + fw, CellState.FUEL, ftype)
        furniture_positions.append((fr + fh // 2, fc + fw // 2))

    # Internal wall / divider
    if kwargs.get('has_divider', True):
        wall_c = width // 2
        gap_r = rng.randint(height // 4, 3 * height // 4)
        for r in range(2, height - 2):
            if abs(r - gap_r) > 3:  # Leave a doorway gap
                sim.set_cell(r, wall_c, CellState.WALL, FuelType.CONCRETE)

    # Set ignition point — near a piece of furniture
    ignition = kwargs.get('ignition_point', None)
    if ignition:
        sim.ignite(ignition[0], ignition[1])
    elif furniture_positions:
        ir, ic = rng.choice(furniture_positions)
        # Offset slightly from center of furniture
        sim.ignite(ir + rng.randint(-1, 1), ic + rng.randint(-1, 1))

    # Wind (indoor — minimal, but can represent ventilation)
    wind_dir = kwargs.get('wind_direction', rng.uniform(0, 360))
    wind_speed = kwargs.get('wind_speed', rng.uniform(0.0, 0.15))
    sim.set_wind(wind_dir, wind_speed)

    return sim


def create_outdoor_scene(width: int = 100, height: int = 80,
                         seed: int = 42, **kwargs) -> FireSimulator:
    """
    Outdoor field/forest scene: grass, trees (brush), clearings.
    Fire typically starts at a point and spreads with wind.
    """
    sim = FireSimulator(width, height, seed=seed)
    rng = sim.rng

    fuel_density = kwargs.get('fuel_density', 0.75)

    # Base layer: grass across most of the field
    sim.scatter_fuel(0, 0, height - 1, width - 1, FuelType.GRASS, density=fuel_density)

    # Tree clusters (brush/wood)
    num_clusters = kwargs.get('num_tree_clusters', 6)
    for _ in range(num_clusters):
        cx = rng.randint(5, width - 6)
        cy = rng.randint(5, height - 6)
        radius = rng.uniform(3, 8)
        tree_type = rng.choice([FuelType.WOOD, FuelType.BRUSH])
        sim.fill_circle(cy, cx, radius, CellState.FUEL, tree_type)

    # Clearings / firebreaks (empty patches)
    num_clearings = kwargs.get('num_clearings', 3)
    for _ in range(num_clearings):
        cx = rng.randint(10, width - 11)
        cy = rng.randint(10, height - 11)
        radius = rng.uniform(2, 5)
        sim.fill_circle(cy, cx, radius, CellState.EMPTY)

    # A dirt road (firebreak)
    if kwargs.get('has_road', True):
        road_r = rng.randint(height // 3, 2 * height // 3)
        for c in range(width):
            for dr in range(-1, 2):
                r = road_r + dr
                if 0 <= r < height:
                    sim.set_cell(r, c, CellState.EMPTY, FuelType.NONE)

    # Ignition
    ignition = kwargs.get('ignition_point', None)
    if ignition:
        sim.ignite(ignition[0], ignition[1])
    else:
        # Random ignition in a grassy area
        for _ in range(100):
            ir = rng.randint(5, height - 6)
            ic = rng.randint(5, width - 6)
            if sim.grid[ir][ic].state == CellState.FUEL:
                sim.ignite(ir, ic)
                break

    # Wind (outdoor — can be strong)
    wind_dir = kwargs.get('wind_direction', rng.uniform(0, 360))
    wind_speed = kwargs.get('wind_speed', rng.uniform(0.2, 0.7))
    sim.set_wind(wind_dir, wind_speed)

    return sim


def create_urban_scene(width: int = 100, height: int = 80,
                       seed: int = 42, **kwargs) -> FireSimulator:
    """
    Urban/building exterior: concrete structures, wooden fixtures,
    windows (openings), vegetation nearby.
    """
    sim = FireSimulator(width, height, seed=seed)
    rng = sim.rng

    # Ground level — some grass/vegetation
    sim.scatter_fuel(0, 0, height - 1, width - 1, FuelType.GRASS, density=0.2)

    # Buildings (concrete walls with interior fuel)
    num_buildings = kwargs.get('num_buildings', 3)
    buildings = []
    for i in range(num_buildings):
        bw = rng.randint(15, 25)
        bh = rng.randint(12, 20)
        br = rng.randint(2, height - bh - 3)
        bc = rng.randint(2, width - bw - 3)

        # Concrete shell
        sim.fill_rect(br, bc, br + bh, bc + bw, CellState.WALL, FuelType.CONCRETE)

        # Interior (wood/fabric)
        sim.fill_rect(br + 2, bc + 2, br + bh - 2, bc + bw - 2, CellState.FUEL, FuelType.WOOD)

        # Windows (openings in walls) — on all sides
        for _ in range(rng.randint(2, 5)):
            side = rng.randint(0, 3)
            if side == 0:  # Top
                wc = rng.randint(bc + 3, bc + bw - 3)
                sim.set_cell(br, wc, CellState.EMPTY, FuelType.NONE)
                sim.set_cell(br, wc + 1, CellState.EMPTY, FuelType.NONE)
            elif side == 1:  # Bottom
                wc = rng.randint(bc + 3, bc + bw - 3)
                sim.set_cell(br + bh, wc, CellState.EMPTY, FuelType.NONE)
                sim.set_cell(br + bh, wc + 1, CellState.EMPTY, FuelType.NONE)
            elif side == 2:  # Left
                wr = rng.randint(br + 3, br + bh - 3)
                sim.set_cell(wr, bc, CellState.EMPTY, FuelType.NONE)
                sim.set_cell(wr + 1, bc, CellState.EMPTY, FuelType.NONE)
            else:  # Right
                wr = rng.randint(br + 3, br + bh - 3)
                sim.set_cell(wr, bc + bw, CellState.EMPTY, FuelType.NONE)
                sim.set_cell(wr + 1, bc + bw, CellState.EMPTY, FuelType.NONE)

        # Door
        door_c = rng.randint(bc + 3, bc + bw - 4)
        sim.set_cell(br + bh, door_c, CellState.EMPTY)
        sim.set_cell(br + bh, door_c + 1, CellState.EMPTY)
        sim.set_cell(br + bh, door_c + 2, CellState.EMPTY)

        buildings.append((br, bc, bh, bw))

    # Trees along streets
    for _ in range(rng.randint(5, 10)):
        tx = rng.randint(2, width - 3)
        ty = rng.randint(2, height - 3)
        if sim.grid[ty][tx].state == CellState.EMPTY:
            sim.fill_circle(ty, tx, rng.uniform(1.5, 3.0), CellState.FUEL, FuelType.BRUSH)

    # Ignition — inside one building
    ignition = kwargs.get('ignition_point', None)
    if ignition:
        sim.ignite(ignition[0], ignition[1])
    elif buildings:
        br, bc, bh, bw = rng.choice(buildings)
        sim.ignite(br + bh // 2, bc + bw // 2)

    # Wind
    wind_dir = kwargs.get('wind_direction', rng.uniform(0, 360))
    wind_speed = kwargs.get('wind_speed', rng.uniform(0.1, 0.4))
    sim.set_wind(wind_dir, wind_speed)

    return sim


# Scene factory
SCENE_CREATORS = {
    'indoor': create_indoor_scene,
    'outdoor': create_outdoor_scene,
    'urban': create_urban_scene,
}


def create_scene(scene_type: str, **kwargs) -> FireSimulator:
    """Create a scene by type name."""
    if scene_type not in SCENE_CREATORS:
        raise ValueError(f"Unknown scene type '{scene_type}'. "
                         f"Available: {list(SCENE_CREATORS.keys())}")
    return SCENE_CREATORS[scene_type](**kwargs)
