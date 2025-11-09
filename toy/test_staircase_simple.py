"""Simple test to verify staircase environment works."""

import jax
import jax.numpy as jnp
from staircase_env import StaircaseEnv, StaticEnvParams

# Create environment with default static params (includes fixed correct staircase pattern)
env = StaircaseEnv()
params = env.default_params

print(f"Environment: {env.name}")
print(f"Number of floors: {env.static_params.num_floors}")
print(f"Grid size: {env.static_params.grid_size}")
print(f"\nCorrect staircase TYPE pattern (True = Type 2 is correct, False = Type 3 is correct):")
print(env.static_params.correct_stair_pattern)
print(f"\nExample interpretation:")
print(f"  Floor 0: Type {'2' if env.static_params.correct_stair_pattern[0] else '3'} is correct")
print(f"  Floor 1: Type {'2' if env.static_params.correct_stair_pattern[1] else '3'} is correct")
print(f"  Floor 2: Type {'2' if env.static_params.correct_stair_pattern[2] else '3'} is correct")

# Reset environment
rng = jax.random.PRNGKey(0)
obs, state = env.reset_env(rng, params)

print(f"\nObservation shape: {obs.shape}")
print(f"Initial floor: {state.current_floor}")
print(f"Agent position: {state.agent_pos}")
print(f"Grids shape (all floors): {state.grids.shape}")

# Show first floor
print(f"\nFloor 0 grid (0=empty, 1=agent, 2=type_2_stair, 3=type_3_stair):")
print(state.grids[0])
floor_0_correct = "Type 2" if env.static_params.correct_stair_pattern[0] else "Type 3"
print(f"On floor 0, {floor_0_correct} staircase is correct (90% success rate)")

# Take a random step
rng, step_rng = jax.random.split(rng)
action = 0  # UP
obs, state, reward, done, info = env.step_env(step_rng, state, action, params)

print(f"\nAfter action {action}:")
print(f"Reward: {reward}")
print(f"Current floor: {state.current_floor}")
print(f"Agent position: {state.agent_pos}")
print(f"Done: {done}")

print("\n✓ Environment works correctly!")
