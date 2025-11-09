#!/usr/bin/env python3
"""
Script to simulate Fabrax world generation and count average ore spawns.

Additionally, if Craftax (non‑Fabrax) modules are available, also sample Craftax
worlds to verify the presence of the down staircase (ladder) on Floor 0 and
report simple statistics for it. This is helpful to sanity‑check ladder spawning.
"""

import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from flax import struct
from typing import Dict, List
import sys
import os

# Add the Craftax path to import the modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'Craftax'))

from craftax.fabrax.world_gen import generate_world
from craftax.fabrax.constants import BlockType

# Optional Craftax (non‑Fabrax) imports for ladder stats
_HAS_CRAFTAX = False
try:
    from craftax.craftax.craftax.world_gen.world_gen import generate_world as c_generate_world
    from craftax.craftax.craftax.craftax_state import EnvParams as CEnvParams, StaticEnvParams as CStaticEnvParams
    from craftax.craftax.craftax.constants import ItemType as CItemType
    _HAS_CRAFTAX = True
except Exception:
    _HAS_CRAFTAX = False


@struct.dataclass
class SimParams:
    """Parameters for world generation simulation"""
    max_timesteps: int = 10000
    day_length: int = 300
    always_diamond: bool = False
    zombie_health: int = 5
    cow_health: int = 3
    skeleton_health: int = 3
    mob_despawn_distance: int = 14
    spawn_cow_chance: float = 0.1
    spawn_zombie_base_chance: float = 0.02
    spawn_zombie_night_chance: float = 0.1
    spawn_skeleton_chance: float = 0.05
    fractal_noise_angles: tuple = (None, None, None, None)


@struct.dataclass
class StaticParams:
    """Static parameters for world generation"""
    map_size: tuple = (64, 64)
    max_zombies: int = 3
    max_cows: int = 3
    max_growing_plants: int = 10
    max_skeletons: int = 2
    max_arrows: int = 3


def count_ores_in_map(world_map: jnp.ndarray) -> Dict[str, int]:
    """Count the number of each ore type in the generated map"""
    ore_counts = {}

    # Count each block type
    ore_counts['coal'] = jnp.sum(world_map == BlockType.COAL.value).item()
    ore_counts['iron'] = jnp.sum(world_map == BlockType.IRON.value).item()
    ore_counts['diamond'] = jnp.sum(world_map == BlockType.DIAMOND.value).item()
    ore_counts['copper'] = jnp.sum(world_map == BlockType.COPPER.value).item()
    ore_counts['tin'] = jnp.sum(world_map == BlockType.TIN.value).item()
    ore_counts['limestone'] = jnp.sum(world_map == BlockType.LIMESTONE.value).item()
    ore_counts['sand'] = jnp.sum(world_map == BlockType.SAND.value).item()
    ore_counts['clay'] = jnp.sum(world_map == BlockType.CLAY.value).item()
    ore_counts['stone'] = jnp.sum(world_map == BlockType.STONE.value).item()
    ore_counts['grass'] = jnp.sum(world_map == BlockType.GRASS.value).item()
    ore_counts['water'] = jnp.sum(world_map == BlockType.WATER.value).item()
    ore_counts['tree'] = jnp.sum(world_map == BlockType.TREE.value).item()

    return ore_counts


def simulate_world_generation(num_simulations: int = 100, map_size: tuple = (64, 64)) -> Dict[str, List[float]]:
    """Simulate world generation multiple times and collect ore counts.

    If Craftax is available, also collect down‑ladder counts from Craftax Floor 0.
    """

    static_params = StaticParams(map_size=map_size)
    all_counts = {
        'coal': [], 'iron': [], 'diamond': [], 'copper': [], 'tin': [],
        'limestone': [], 'sand': [], 'clay': [], 'stone': [], 'grass': [],
        'water': [], 'tree': []
    }
    craftax_down_ladder_counts: List[int] = []

    print(f"Simulating {num_simulations} world generations with map size {map_size}...")

    for i in range(num_simulations):
        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{num_simulations} simulations")

        # Generate random parameters for this simulation
        rng = jax.random.PRNGKey(i + 1000)  # Offset to avoid potential issues with seed 0

        # Use default parameters like the environment does
        params = SimParams(
            always_diamond=True  # Use natural diamond generation
        )

        # Generate Fabrax world for ore/terrain counts
        state = generate_world(rng, params, static_params)

        # Count ores in this world
        ore_counts = count_ores_in_map(state.map)

        # Add to our collection
        for ore_type, count in ore_counts.items():
            all_counts[ore_type].append(count)

        # Optionally: sample Craftax world to check down‑ladder on Floor 0
        if _HAS_CRAFTAX:
            rng_c = jax.random.PRNGKey(10_000 + i)
            c_state = c_generate_world(rng_c, CEnvParams(), CStaticEnvParams(map_size=map_size))
            # Count ladder down tiles on Floor 0 using item_map and ItemType
            # item_map shape: (levels, H, W); Floor 0 is index 0.
            down_count = int(jnp.sum(c_state.item_map[0] == CItemType.LADDER_DOWN.value))
            craftax_down_ladder_counts.append(down_count)

    # Attach Craftax ladder stats if available
    if _HAS_CRAFTAX:
        all_counts['craftax_down_ladder_count'] = craftax_down_ladder_counts

    return all_counts


def calculate_statistics(counts: List[float]) -> Dict[str, float]:
    """Calculate statistics for a list of counts"""
    counts_array = np.array(counts)
    return {
        'mean': np.mean(counts_array),
        'std': np.std(counts_array),
        'min': np.min(counts_array),
        'max': np.max(counts_array),
        'median': np.median(counts_array)
    }


def print_results(all_counts: Dict[str, List[float]], map_size: tuple):
    """Print formatted results, including optional Craftax down‑ladder stats."""
    total_blocks = map_size[0] * map_size[1]

    print(f"\n{'='*60}")
    print(f"ORE DISTRIBUTION ANALYSIS ({map_size[0]}x{map_size[1]} map, {total_blocks} total blocks)")
    print(f"{'='*60}")

    # Group by category
    ores = ['coal', 'iron', 'diamond', 'copper', 'tin', 'limestone']
    terrain = ['stone', 'grass', 'water', 'sand', 'clay', 'tree']

    print(f"\n{'ORES AND MINERALS':<20} {'Count':<8} {'%':<6} {'Std':<6} {'Min':<4} {'Max':<4}")
    print("-" * 60)

    for ore in ores:
        stats = calculate_statistics(all_counts[ore])
        percentage = (stats['mean'] / total_blocks) * 100
        print(f"{ore.upper():<20} {stats['mean']:<8.1f} {percentage:<6.2f} {stats['std']:<6.1f} {stats['min']:<4.0f} {stats['max']:<4.0f}")

    print(f"\n{'TERRAIN BLOCKS':<20} {'Count':<8} {'%':<6} {'Std':<6} {'Min':<4} {'Max':<4}")
    print("-" * 60)

    for terrain_type in terrain:
        stats = calculate_statistics(all_counts[terrain_type])
        percentage = (stats['mean'] / total_blocks) * 100
        print(f"{terrain_type.upper():<20} {stats['mean']:<8.1f} {percentage:<6.2f} {stats['std']:<6.1f} {stats['min']:<4.0f} {stats['max']:<4.0f}")

    # Calculate spawn rates based on stone blocks
    stone_stats = calculate_statistics(all_counts['stone'])
    print(f"\n{'ORE SPAWN RATES (% of stone blocks)'}")
    print("-" * 40)

    for ore in ['coal', 'iron', 'copper', 'tin', 'limestone']:
        if ore in all_counts:
            ore_stats = calculate_statistics(all_counts[ore])
            if stone_stats['mean'] > 0:
                spawn_rate = (ore_stats['mean'] / stone_stats['mean']) * 100
                print(f"{ore.upper():<12}: {spawn_rate:.2f}%")

    # Optional Craftax down‑ladder stats (Floor 0)
    if 'craftax_down_ladder_count' in all_counts and all_counts['craftax_down_ladder_count']:
        ladder_stats = calculate_statistics(all_counts['craftax_down_ladder_count'])
        print(f"\n{'CRAFtAX FLOOR 0 DOWN-LADDER'}")
        print("-" * 40)
        print(f"mean count: {ladder_stats['mean']:.2f} | std: {ladder_stats['std']:.2f} | min: {ladder_stats['min']:.0f} | max: {ladder_stats['max']:.0f}")
    else:
        print(f"\nCraftax ladder stats: unavailable (Craftax modules not found in environment)")


def main():
    """Main function"""
    # Set random seed for reproducibility
    jax.config.update('jax_platform_name', 'cpu')  # Force CPU for consistency

    # Default parameters
    num_simulations = 50
    map_size = (64, 64)

    # Parse command line arguments if provided
    if len(sys.argv) > 1:
        num_simulations = int(sys.argv[1])
    if len(sys.argv) > 2:
        size = int(sys.argv[2])
        map_size = (size, size)

    # Run simulation
    all_counts = simulate_world_generation(num_simulations, map_size)
    print(all_counts)

    # Print results
    print_results(all_counts, map_size)

    print(f"\n{'='*60}")
    print("NOTES:")
    print("- Percentages are relative to total map blocks")
    print("- Spawn rates show ore frequency within stone blocks")
    print("- Clay spawns as a percentage of sand blocks (30% conversion)")
    print("- Diamond requires mountain > 0.8 AND stone blocks")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
