"""Test script for the staircase environment."""

import jax
import jax.numpy as jnp
from staircase_env import StaircaseEnv, StaticEnvParams


def test_basic_functionality():
    """Test basic environment functionality."""
    env = StaircaseEnv()
    params = env.default_params

    # Reset environment
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset_env(rng, params)

    print(f"Environment: {env.name}")
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space(params)}")
    print(f"Initial floor: {state.current_floor}")
    print(f"Agent position: {state.agent_pos}")
    print(f"\nInitial grid (1=agent, 2=correct_stair, 3=wrong_stair):")
    print(state.grid)

    # Take some random steps
    print("\n--- Taking 10 random steps ---")
    for i in range(10):
        rng, step_rng, action_rng = jax.random.split(rng, 3)
        action = jax.random.randint(action_rng, (), 0, 5)

        obs, state, reward, done, info = env.step_env(step_rng, state, action, params)

        print(f"Step {i+1}: action={action}, reward={reward:.1f}, floor={info['current_floor']}, "
              f"done={done}, pos={state.agent_pos}")

        if done:
            print(f"Episode ended! Won: {info['won']}, Died: {info['died']}")
            break


def test_staircase_mechanics():
    """Test that staircases work correctly."""
    env = StaircaseEnv()
    params = env.default_params

    print("\n=== Testing Staircase Mechanics ===")

    # Run multiple episodes to test stochasticity
    num_episodes = 5
    for ep in range(num_episodes):
        rng = jax.random.PRNGKey(ep)
        obs, state = env.reset_env(rng, params)

        print(f"\nEpisode {ep + 1}:")
        print(f"Grid layout:")
        print(state.grid)

        # Find staircase positions
        correct_stair_pos = jnp.where(state.grid == 2, size=1)
        wrong_stair_pos = jnp.where(state.grid == 3, size=1)

        print(f"Correct staircase (2): y={correct_stair_pos[0][0]}, x={correct_stair_pos[1][0]}")
        print(f"Wrong staircase (3): y={wrong_stair_pos[0][0]}, x={wrong_stair_pos[1][0]}")
        print(f"Agent position: {state.agent_pos}")


def test_floor_consistency():
    """Test that staircase correctness is consistent across episodes."""
    env = StaircaseEnv()
    params = env.default_params

    print("\n=== Testing Floor Consistency ===")
    print("Testing that the same seed produces the same correct staircase pattern...")

    # Use same seed for two episodes
    rng = jax.random.PRNGKey(42)

    obs1, state1 = env.reset_env(rng, params)
    obs2, state2 = env.reset_env(rng, params)

    # Check if correct_stair_is_first pattern is the same
    same_pattern = jnp.all(state1.correct_stair_is_first == state2.correct_stair_is_first)
    print(f"Same seed produces same staircase pattern: {same_pattern}")

    # Use different seed
    rng_diff = jax.random.PRNGKey(43)
    obs3, state3 = env.reset_env(rng_diff, params)

    different_pattern = jnp.any(state1.correct_stair_is_first != state3.correct_stair_is_first)
    print(f"Different seed produces different pattern: {different_pattern}")

    print(f"\nFirst 5 floors - Seed 42: {state1.correct_stair_is_first[:5]}")
    print(f"First 5 floors - Seed 43: {state3.correct_stair_is_first[:5]}")


if __name__ == "__main__":
    test_basic_functionality()
    test_staircase_mechanics()
    test_floor_consistency()
    print("\n=== All tests completed! ===")
