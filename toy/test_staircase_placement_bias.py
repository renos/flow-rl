"""Test if staircase placement has a bias across episodes."""

import jax
import jax.numpy as jnp
from staircase_env import StaircaseEnv, StaticEnvParams

# Create environment with corridor mode
static_params = StaticEnvParams(
    grid_height=1,
    grid_width=25,
    place_stairs_at_ends=True,
)
env = StaircaseEnv(static_params)
params = env.default_params

# Test across multiple episodes
num_episodes = 100
num_floors_to_check = 10

# Track: for each floor, how many times type 2 spawns on the left
type2_on_left_count = jnp.zeros(num_floors_to_check)

for ep in range(num_episodes):
    rng = jax.random.PRNGKey(ep)
    obs, state = env.reset_env(rng, params)

    # Check first few floors
    for floor_idx in range(num_floors_to_check):
        floor_grid = state.grids[floor_idx]

        # Find where type 2 is (should be at either x=0 or x=24)
        # Grid is (height, width), and we have height=1, so check row 0
        left_tile = floor_grid[0, 0]  # Left end
        right_tile = floor_grid[0, -1]  # Right end

        if left_tile == 2:
            type2_on_left_count = type2_on_left_count.at[floor_idx].add(1)
        elif right_tile == 2:
            pass  # Type 2 is on right
        else:
            print(f"ERROR: Episode {ep}, Floor {floor_idx} - Type 2 not found at ends!")
            print(f"Left: {left_tile}, Right: {right_tile}")

print(f"\n=== Results across {num_episodes} episodes ===")
print(f"Floor | Type2 on Left | Type2 on Right | Left %")
print("-" * 55)
for floor_idx in range(num_floors_to_check):
    left_count = int(type2_on_left_count[floor_idx])
    right_count = num_episodes - left_count
    left_pct = (left_count / num_episodes) * 100

    # Check which type is correct for this floor
    is_type2_correct = static_params.correct_stair_pattern[floor_idx]
    correct_marker = " (Type 2 correct)" if is_type2_correct else " (Type 3 correct)"

    print(f"{floor_idx:5} | {left_count:13} | {right_count:14} | {left_pct:6.1f}%{correct_marker}")

print("\nInterpretation:")
print("- If Left % is close to 50%, placement is unbiased")
print("- If Left % is close to 0% or 100%, there's a deterministic bias")
print("- Agent learning 'always go left' will fail when the correct staircase switches sides")
