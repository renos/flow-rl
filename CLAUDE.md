# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Structure

This is a Flow-RL (Flow-based Reinforcement Learning) research project that integrates LLM-generated code with reinforcement learning agents for the Craftax environment. The project has three main components:

1. **flowrl/**: Main Python package containing RL algorithms and LLM integration
2. **Craftax/**: Embedded Craftax environment (JAX-based RL environment)
3. **exp/**: Experimental configurations and generated code modules

## Key Architecture Components

### RL Training (flowrl/ppo.py, flowrl/ppo_flow.py)
- **ppo.py**: Standard PPO implementation for Craftax environment
- **ppo_flow.py**: Extended PPO with mixture-of-experts (MoE) architecture for flow-based learning
- Both use JAX/Flax for high-performance computation
- Support for optimistic resets and vectorized environments

### LLM Integration (flowrl/llm/)
- **flow.py**: Main Flow class that orchestrates LLM code generation and RL training
- **craftax_classic/**: Contains LLM prompts and code generation logic for Craftax Classic
- **compose_prompts.py**: Prompt composition utilities
- **parse_code.py**: Code validation and parsing

### Neural Network Models (flowrl/models/)
- **actor_critic.py**: Standard and convolutional actor-critic networks, plus MoE variants
- **icm.py**: Intrinsic Curiosity Module for exploration
- **rnd.py**: Random Network Distillation for exploration

## Development Commands

### Running PPO Training
```bash
python -m flowrl.ppo --env_name "Craftax-Symbolic-v1" --total_timesteps 1e9 --use_wandb
```

### Running Flow-based Training
```bash
python -m flowrl.ppo_flow --module_path "path/to/module.py" --env_name "Craftax-Symbolic-v1"
```

### Key Parameters
- `--total_timesteps`: Total training timesteps (accepts scientific notation)
- `--num_envs`: Number of parallel environments (default: 1024)
- `--use_wandb`: Enable Weights & Biases logging
- `--module_path`: Path to LLM-generated reward/state module for flow training
- `--success_state_rate`: Success rate threshold for advancing to next skill (default: 0.8)

## Environment Setup

### Dependencies
Install via pip: `pip install -e .`
Key dependencies managed in `pyproject.toml`:
- JAX ecosystem (jax, flax, optax, distrax)
- RL utilities (gymnax, chex, orbax-checkpoint)
- LLM integration (openai, tiktoken)
- Logging and visualization (wandb, matplotlib, imageio)

### JAX Configuration
The codebase uses JAX with compilation caching enabled for performance:
```python
jax.config.update("jax_compilation_cache_dir", "/path/to/cache/")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
```

## Craftax Environment

The embedded Craftax environment is a JAX-based RL environment inspired by Minecraft-like gameplay. Key features:
- Symbolic and pixel-based observation modes
- Complex crafting and resource management
- Multi-task learning support
- Optimistic reset wrappers for efficient training

## Flow-Based Learning Architecture

The project implements a novel flow-based approach where:
1. LLMs generate Python modules defining reward functions and state transitions
2. These modules are dynamically loaded into the RL training loop
3. MoE networks learn task-specific policies
4. Success thresholds trigger progression to more complex skills

### Generated Module Structure
LLM-generated modules typically contain:
- `reward_and_state`: Function mapping observations to rewards and state transitions
- Task-specific logic for crafting, exploration, combat, etc.

## Logging and Monitoring

- Weights & Biases integration for experiment tracking
- Custom batch logging utilities in `flowrl/logz/`
- Trajectory explanations and visualizations in `flowrl/utils/`

## File Organization Patterns

- Experiment configurations in `exp/` with numbered directories
- Generated policies saved alongside config YAML files
- Cached LLM responses in `flowrl/llm/cache/`
- Human trajectory data in `resources/people/`