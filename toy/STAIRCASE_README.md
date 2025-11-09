# Staircase Dungeon Environment

A simple 10x10 grid-based JAX environment where an agent must navigate through 30 floors by choosing the correct staircase.

## Environment Description

### Grid Layout
- **Size**: 10x10 grid
- **Agent**: Represented as `1` on the grid
- **Correct Staircase**: Represented as `2` on the grid (90% success rate, 10% death)
- **Wrong Staircase**: Represented as `3` on the grid (100% death)
- **Empty Tile**: Represented as `0`

### Game Mechanics

1. **Objective**: Reach floor 30 by successfully navigating through staircases

2. **Staircase Logic**:
   - Each floor has exactly 2 staircases spawned randomly
   - One staircase is "correct" (90% chance to advance, 10% chance to die)
   - One staircase is "wrong" (100% chance to die)
   - Which staircase is correct for each floor is **deterministic** (based on RNG seed)
   - Only the **spawn locations** of staircases are randomized per episode

3. **Actions**:
   - `0`: Move UP
   - `1`: Move DOWN
   - `2`: Move LEFT
   - `3`: Move RIGHT
   - `4`: NOOP (no operation)

4. **Rewards**:
   - `+1.0`: Successfully advance to next floor
   - `-1.0`: Die (from wrong staircase or failed roll on correct staircase)
   - `+10.0`: Complete all 30 floors (win condition)
   - `0.0`: Movement without reaching a staircase

5. **Episode Termination**:
   - Agent dies (wrong staircase or failed roll)
   - Agent completes all 30 floors
   - Maximum timesteps reached (default: 1000)

### Observation Space

The observation is a 130-dimensional vector:
- First 100 dimensions: Flattened 10x10 grid
- Last 30 dimensions: One-hot encoding of current floor (0-29)

## Files

- `staircase_env.py`: Main environment implementation
- `test_staircase_env.py`: Test script to verify environment mechanics
- `ppo_staircase.py`: PPO training script with wandb logging

## Usage

### Testing the Environment

```bash
cd /Users/renos/Documents/flow-rl/toy
python test_staircase_env.py
```

This will run several tests:
1. Basic functionality test
2. Staircase mechanics test
3. Floor consistency test (verifying determinism)

### Training with PPO

```bash
cd /Users/renos/Documents/flow-rl/toy
python ppo_staircase.py
```

Configuration options in `ppo_staircase.py`:
- `TOTAL_TIMESTEPS`: Total training steps (default: 5M)
- `NUM_ENVS`: Number of parallel environments (default: 16)
- `NUM_STEPS`: Steps per environment per update (default: 256)
- `LR`: Learning rate (default: 3e-4)
- `USE_WANDB`: Enable/disable wandb logging

### Using the Environment Programmatically

```python
import jax
from staircase_env import StaircaseEnv

# Create environment
env = StaircaseEnv()
params = env.default_params

# Reset
rng = jax.random.PRNGKey(0)
obs, state = env.reset_env(rng, params)

# Take a step
rng, step_rng = jax.random.split(rng)
action = 0  # Move UP
obs, state, reward, done, info = env.step_env(step_rng, state, action, params)

print(f"Current floor: {state.current_floor}")
print(f"Agent position: {state.agent_pos}")
print(f"Reward: {reward}")
print(f"Done: {done}")
```

## Environment Parameters

You can customize the environment by modifying `EnvParams`:

```python
from staircase_env import EnvParams

params = EnvParams(
    max_timesteps=2000,      # Increase time limit
    num_floors=50,            # More floors
    grid_size=15,             # Larger grid
    success_prob=0.8,         # Lower success probability
)
```

## Key Features

1. **Deterministic Staircase Assignment**: The "correct" staircase for each floor is consistent across episodes with the same seed, making the environment suitable for studying exploration vs exploitation

2. **Stochastic Success**: Even the correct staircase has a 90% success rate, adding uncertainty

3. **Procedural Generation**: Staircase positions are randomized each episode while maintaining the underlying pattern

4. **JAX-Based**: Fully JIT-compilable for fast parallel training

5. **Compatible with PureJaxRL**: Follows the same patterns as other gymnax environments

## Training Tips

1. **Exploration**: The agent needs to explore to find staircases and learn which is correct
2. **Memory**: The agent should learn to remember which staircase was correct on each floor
3. **Risk Assessment**: The agent must balance exploration (finding new staircases) vs exploitation (using known good staircases)

## Wandb Metrics

When training with wandb enabled, the following metrics are logged:
- `charts/episodic_return`: Average episode return
- `charts/episodic_length`: Average episode length
- `env/max_floor_reached`: Highest floor reached in episode
- `env/avg_floor_reached`: Average floor reached
- `env/win_rate`: Percentage of episodes that completed all floors
- `env/death_rate`: Percentage of episodes that ended in death
