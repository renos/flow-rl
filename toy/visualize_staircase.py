"""Visualize the Staircase environment."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from staircase_env import StaircaseEnv, StaticEnvParams

# Create environment with corridor mode
static_params = StaticEnvParams(
    grid_height=1,
    grid_width=25,
    place_stairs_at_ends=True,
)
env = StaircaseEnv(static_params)
params = env.default_params

# Reset environment
rng = jax.random.PRNGKey(42)
obs, state = env.reset_env(rng, params)

print(f"Current floor: {state.current_floor}")
print(f"Agent position: {state.agent_pos}")
print(f"Correct staircase pattern (first 5 floors): {static_params.correct_stair_pattern[:5]}")
print(f"  (True = Type 2 correct, False = Type 3 correct)")

# Get current floor grid
current_grid = state.grids[state.current_floor]

# Create visualization grid (add agent to it)
vis_grid = current_grid.copy()
vis_grid = vis_grid.at[state.agent_pos[1], state.agent_pos[0]].set(1)

# Create figure
fig, ax = plt.subplots(figsize=(15, 2))

# Define colors
# 0 = empty (white)
# 1 = agent (blue)
# 2 = Type 2 staircase (cyan)
# 3 = Type 3 staircase (magenta)
colors = ['white', 'blue', 'cyan', 'magenta']
cmap = plt.matplotlib.colors.ListedColormap(colors)

# Plot
im = ax.imshow(vis_grid, cmap=cmap, vmin=0, vmax=3)

# Add grid lines
ax.set_xticks(jnp.arange(-.5, static_params.grid_width, 1), minor=True)
ax.set_yticks(jnp.arange(-.5, static_params.grid_height, 1), minor=True)
ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
ax.tick_params(which="minor", size=0)

# Set major ticks
ax.set_xticks(range(0, static_params.grid_width, 5))
ax.set_yticks([0])

# Labels
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')

# Determine which type is correct for this floor
is_type2_correct = static_params.correct_stair_pattern[state.current_floor]
correct_type = "Type 2 (cyan)" if is_type2_correct else "Type 3 (magenta)"

ax.set_title(f'Floor {state.current_floor} - Correct staircase: {correct_type}')

# Create legend
legend_elements = [
    mpatches.Patch(color='white', edgecolor='black', label='Empty'),
    mpatches.Patch(color='blue', label='Agent'),
    mpatches.Patch(color='cyan', label='Type 2 Staircase'),
    mpatches.Patch(color='magenta', label='Type 3 Staircase'),
]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.savefig('staircase_frame.png', dpi=150, bbox_inches='tight')
print(f"\nVisualization saved to 'staircase_frame.png'")
plt.show()

# Take a few steps and visualize
print("\n=== Taking a few steps ===")
for step in range(5):
    # Random action
    rng, action_rng = jax.random.split(rng)
    action = jax.random.randint(action_rng, (), 0, 5)
    action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'NOOP']

    rng, step_rng = jax.random.split(rng)
    obs, state, reward, done, info = env.step_env(step_rng, state, action, params)

    print(f"Step {step+1}: Action={action_names[action]}, "
          f"Position={state.agent_pos}, Reward={reward:.1f}, Done={done}")

    if done:
        print(f"  Episode ended! Won={info['won']}, Died={info['died']}")
        break

# Visualize final state
current_grid = state.grids[state.current_floor]
vis_grid = current_grid.copy()
vis_grid = vis_grid.at[state.agent_pos[1], state.agent_pos[0]].set(1)

fig, ax = plt.subplots(figsize=(15, 2))
im = ax.imshow(vis_grid, cmap=cmap, vmin=0, vmax=3)
ax.set_xticks(jnp.arange(-.5, static_params.grid_width, 1), minor=True)
ax.set_yticks(jnp.arange(-.5, static_params.grid_height, 1), minor=True)
ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
ax.tick_params(which="minor", size=0)
ax.set_xticks(range(0, static_params.grid_width, 5))
ax.set_yticks([0])
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')

is_type2_correct = static_params.correct_stair_pattern[state.current_floor]
correct_type = "Type 2 (cyan)" if is_type2_correct else "Type 3 (magenta)"
ax.set_title(f'Floor {state.current_floor} - After {step+1} steps - Correct: {correct_type}')
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.savefig('staircase_frame_after.png', dpi=150, bbox_inches='tight')
print(f"Final visualization saved to 'staircase_frame_after.png'")
plt.show()
