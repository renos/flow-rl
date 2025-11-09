#!/usr/bin/env python3
"""
Analyze ladder positions across multiple random seeds.
Prints mean and standard deviation of down ladder positions on floor 0.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax.constants import ItemType


def find_down_ladder_position(state, level=0):
    """
    Find the position of the down ladder on a given floor.
    Returns (y, x) position or None if not found.
    """
    # Try using down_ladders attribute first (if available)
    if hasattr(state, "down_ladders"):
        ladder_yx = jnp.array(state.down_ladders[level])
        return ladder_yx

    # Otherwise, search in item_map
    if hasattr(state, "item_map"):
        item_layer = state.item_map[level]
        ys, xs = jnp.where(item_layer == ItemType.LADDER_DOWN.value)
        if ys.size > 0:
            return jnp.array([ys[0], xs[0]])

    return None


def analyze_ladder_positions(num_seeds=100, env_name="Craftax-Symbolic-v1"):
    """
    Analyze ladder positions across multiple random seeds.
    """
    print(f"Analyzing ladder positions for {env_name}")
    print(f"Running {num_seeds} different seeds...")
    print("=" * 60)

    # Create environment once (we'll reset it with different seeds)
    env = make_craftax_env_from_name(env_name, auto_reset=True)
    env_params = env.default_params

    ladder_positions = []

    for seed in range(num_seeds):
        # Reset environment with this seed
        rng = jax.random.PRNGKey(seed)
        obs, state = env.reset(rng, env_params)

        # Find ladder position on floor 0 (overworld)
        ladder_pos = find_down_ladder_position(state, level=0)

        if ladder_pos is not None:
            y, x = int(ladder_pos[0]), int(ladder_pos[1])
            ladder_positions.append([y, x])

        # Print progress every 10 seeds
        if (seed + 1) % 10 == 0:
            print(f"  Processed {seed + 1}/{num_seeds} seeds...")

    # Convert to numpy array for easy statistics
    ladder_positions = np.array(ladder_positions)

    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)

    if len(ladder_positions) > 0:
        # Compute statistics
        mean_y = np.mean(ladder_positions[:, 0])
        mean_x = np.mean(ladder_positions[:, 1])
        std_y = np.std(ladder_positions[:, 0])
        std_x = np.std(ladder_positions[:, 1])

        min_y = np.min(ladder_positions[:, 0])
        max_y = np.max(ladder_positions[:, 0])
        min_x = np.min(ladder_positions[:, 1])
        max_x = np.max(ladder_positions[:, 1])

        print(f"\n📊 Ladder Position Statistics (n={len(ladder_positions)}):")
        print(f"   Y-coordinate (row):")
        print(f"      Mean:  {mean_y:.2f}")
        print(f"      Std:   {std_y:.2f}")
        print(f"      Range: [{min_y}, {max_y}]")
        print(f"\n   X-coordinate (col):")
        print(f"      Mean:  {mean_x:.2f}")
        print(f"      Std:   {std_x:.2f}")
        print(f"      Range: [{min_x}, {max_x}]")

        # Check if ladders are concentrated near edges
        # Assuming map size is around 48x48 based on Craftax
        map_size = 48
        edge_threshold = 5  # Within 5 tiles of edge

        near_top = np.sum(ladder_positions[:, 0] < edge_threshold)
        near_bottom = np.sum(ladder_positions[:, 0] > (map_size - edge_threshold))
        near_left = np.sum(ladder_positions[:, 1] < edge_threshold)
        near_right = np.sum(ladder_positions[:, 1] > (map_size - edge_threshold))

        total = len(ladder_positions)
        print(f"\n📍 Edge Analysis (within {edge_threshold} tiles of edge):")
        print(f"   Near top:    {near_top}/{total} ({100*near_top/total:.1f}%)")
        print(f"   Near bottom: {near_bottom}/{total} ({100*near_bottom/total:.1f}%)")
        print(f"   Near left:   {near_left}/{total} ({100*near_left/total:.1f}%)")
        print(f"   Near right:  {near_right}/{total} ({100*near_right/total:.1f}%)")

        near_any_edge = np.sum(
            (ladder_positions[:, 0] < edge_threshold)
            | (ladder_positions[:, 0] > (map_size - edge_threshold))
            | (ladder_positions[:, 1] < edge_threshold)
            | (ladder_positions[:, 1] > (map_size - edge_threshold))
        )
        print(f"   Near ANY edge: {near_any_edge}/{total} ({100*near_any_edge/total:.1f}%)")

        # Show first 10 positions as examples
        print(f"\n📋 First 10 Ladder Positions (y, x):")
        for i in range(min(10, len(ladder_positions))):
            y, x = ladder_positions[i]
            print(f"   Seed {i}: ({y}, {x})")

        # Generate visualization
        plot_ladder_positions(ladder_positions, env_name, num_seeds)

    else:
        print("❌ No ladders found in any seed!")

    print("\n" + "=" * 60)


def plot_ladder_positions(ladder_positions, env_name, num_seeds):
    """
    Create a scatter plot showing all ladder spawn positions.
    """
    map_size = 48  # Craftax map size

    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot all ladder positions
    ax.scatter(
        ladder_positions[:, 1],  # X coordinates
        ladder_positions[:, 0],  # Y coordinates
        alpha=0.6,
        s=100,
        c='red',
        marker='o',
        edgecolors='darkred',
        linewidths=1.5,
        label=f'Ladder spawns (n={len(ladder_positions)})'
    )

    # Plot mean position
    mean_y = np.mean(ladder_positions[:, 0])
    mean_x = np.mean(ladder_positions[:, 1])
    ax.scatter(
        mean_x, mean_y,
        s=300,
        c='yellow',
        marker='*',
        edgecolors='black',
        linewidths=2,
        label=f'Mean position ({mean_y:.1f}, {mean_x:.1f})',
        zorder=5
    )

    # Draw standard deviation ellipse
    std_y = np.std(ladder_positions[:, 0])
    std_x = np.std(ladder_positions[:, 1])
    from matplotlib.patches import Ellipse
    ellipse = Ellipse(
        (mean_x, mean_y),
        width=2*std_x,
        height=2*std_y,
        facecolor='yellow',
        alpha=0.2,
        edgecolor='orange',
        linewidth=2,
        linestyle='--',
        label=f'±1 std ({std_y:.1f}, {std_x:.1f})'
    )
    ax.add_patch(ellipse)

    # Draw map boundaries
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=2, alpha=0.5)
    ax.axhline(y=map_size-1, color='gray', linestyle='-', linewidth=2, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=2, alpha=0.5)
    ax.axvline(x=map_size-1, color='gray', linestyle='-', linewidth=2, alpha=0.5)

    # Draw edge zones (5 tiles from edge)
    edge_threshold = 5
    ax.axhline(y=edge_threshold, color='blue', linestyle='--', linewidth=1, alpha=0.3)
    ax.axhline(y=map_size-edge_threshold, color='blue', linestyle='--', linewidth=1, alpha=0.3)
    ax.axvline(x=edge_threshold, color='blue', linestyle='--', linewidth=1, alpha=0.3)
    ax.axvline(x=map_size-edge_threshold, color='blue', linestyle='--', linewidth=1, alpha=0.3)

    # Formatting
    ax.set_xlim(-2, map_size+1)
    ax.set_ylim(-2, map_size+1)
    ax.set_xlabel('X Coordinate (Column)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y Coordinate (Row)', fontsize=12, fontweight='bold')
    ax.set_title(
        f'{env_name}\nDown Ladder Spawn Positions (Floor 0)\n{num_seeds} Random Seeds',
        fontsize=14,
        fontweight='bold'
    )
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.set_aspect('equal')

    # Invert Y axis so (0,0) is top-left (standard map convention)
    ax.invert_yaxis()

    plt.tight_layout()

    # Save figure
    output_path = '/Users/renos/Documents/flow-rl/ladder_positions.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Plot saved to: {output_path}")

    # Show plot
    plt.show()


if __name__ == "__main__":
    analyze_ladder_positions(num_seeds=100)
