#!/usr/bin/env python3
"""
Quick test to verify Fabrax observation shapes are correct after our changes.
"""

import sys
sys.path.append('Craftax')

import jax
import jax.numpy as jnp
from craftax.craftax_env import make_craftax_env_from_name

def test_fabrax_shapes():
    """Test that Fabrax environment has correct observation shapes."""
    print("Testing Fabrax observation shapes...")

    # Create Fabrax environment
    env = make_craftax_env_from_name("Fabrax-Symbolic-v1")
    env_params = env.default_params

    # Get observation space info
    obs_space = env.observation_space(env_params)
    print(f"Fabrax observation space shape: {obs_space.shape}")

    # Reset environment and get actual observation
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng, env_params)

    print(f"Actual observation shape: {obs.shape}")
    print(f"Expected: 2968 features")
    print(f"Match: {obs.shape[0] == 2968}")

    if obs.shape[0] == 2968:
        print("✅ Fabrax observation shapes are correct!")
        return True
    else:
        print("❌ Fabrax observation shapes are incorrect!")
        return False

if __name__ == "__main__":
    test_fabrax_shapes()