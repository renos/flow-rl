# SCALAR Method Implementation Summary

This document provides a detailed summary of how the SCALAR (Self-Supervised Composition and Learning of Skills) method works in practice, based on the actual codebase implementation. This serves as a comprehensive guide for understanding the practical instantiation details that need to be documented in the paper's Experiments section.

## 1. System Architecture Overview

### Core Components
The SCALAR implementation consists of three main architectural components:

1. **Flow Controller** (`flowrl/llm/flow.py`): Orchestrates the entire skill discovery and learning pipeline
2. **PPO Flow Trainer** (`flowrl/ppo_flow.py`): Handles RL training with mixture-of-experts networks for multi-skill policies
3. **Skill Dependency Resolver** (`flowrl/skill_dependency_resolver.py`): Manages skill composition and execution ordering

### Environment Integration
- **Environment**: Craftax-Classic-Symbolic-v1 (JAX-based Minecraft-like environment)
- **Symbolic Encoder φ**: Direct access to structured game state (inventory, player status, block positions)
- **Fluent Alphabet 𝒻**: Craftax game elements (wood, stone, tools, achievements, etc.)

## 2. Symbolic Encoder and Task Setup

### Symbolic State Representation
The symbolic encoder φ exposes two primary components of spatial information:

1. **Local Map View (7×9 grid)**: One-hot encoding of blocks within render distance
   - **OBS_DIM**: 7×9 observation window centered on player
   - **Block Types**: 17 different block types (grass, tree, stone, coal, iron, diamond, water, etc.)
   - **Mob Layer**: 4 additional channels for zombies, cows, skeletons, arrows

2. **Closest Block Distances**: Relative positions of nearest blocks of each type
   - **Shape**: `(num_block_types, 2, k)` where k=1 closest block per type
   - **Coordinates**: `(x_relative, y_relative)` distances from player position
   - **Distance Metric**: Manhattan (L1) distance for efficient computation
   - **Out-of-range Encoding**: Blocks not found encoded as (30, 30) - far away placeholder

Additional state components:
```python
# Other symbolic state elements
- Inventory: wood, stone, coal, iron, diamond, sapling, tools (counts 0-9 each)
- Player Status: health, food, drink, energy (each 0-9)  
- Direction: 4-way directional encoding (up, down, left, right)
- Environment: light_level, is_sleeping (binary)
- Player State: Current skill/task index (0-17 discrete states)
```

### Task Node Structure
Each skill in the dependency graph represents a node with the following structure:

```python
skill_data = {
    "skill_name": "Collect Wood",
    "skill_with_consumption": {
        "requirements": {"inventory_space": "lambda n: 0*n + 1"},  # Infrastructure needed
        "consumption": {"energy": "lambda n: 1*n + 0"},           # Resources consumed  
        "gain": {"wood": "lambda n: 4*n + 0"},                    # Resources produced
        "ephemeral": False                                         # Whether skill produces tools vs resources
    },
    "functions": [task_is_done, task_reward, task_network_number], # RL training functions
    "iteration": 5                                                # When skill was learned
}
```

### Lambda Function System
Requirements, consumption, and gain are expressed as lambda functions `lambda n: a*n + b` where:
- `n`: Number of times the skill is executed
- `a`: Scaling coefficient (how much per execution)
- `b`: Fixed cost/gain (one-time requirements)

## 3. Skill Composition and Dependency Resolution

### Graph Construction Algorithm
The `SkillDependencyResolver` implements a sophisticated graph-based planning system:

1. **Dependency Graph Building**: For each skill requirement, recursively find providing skills
2. **Ephemeral Skill Inlining**: Replace tool requirements with their constituent resource requirements
3. **Level-Order Execution**: Use breadth-first traversal to collect resources just-in-time
4. **Graph Pruning**: Remove redundant tool productions using forward simulation
5. **Inventory Constraints**: Apply 9-item capacity limits and split collection tasks

### Example Execution Order Generation
```python
# For "Make Iron Pickaxe" skill:
# 1. Collect Stone (target=3) - for furnace  
# 2. Collect Wood (target=6) - for tools + fuel
# 3. Make Wood Pickaxe (target=1) - for mining iron
# 4. Collect Iron (target=3) - raw material
# 5. Make Iron Pickaxe (target=1) - final goal
```

### Frontier Computation (Approximation for JAX Compatibility)
Due to JAX's static graph requirements during training, the frontier 𝒻*_Σ(P₀) is computed using a backwards dependency resolution approximation:

1. **Proposed Skill Requirements**: Start from the requirements of the currently proposed skill
2. **Backwards Dependency Resolution**: Use `SkillDependencyResolver` to trace backwards through the skill dependency graph
3. **Reachable State Computation**: Determine what states can be reached by executing the resolved dependency chain
4. **Frontier Subset**: This gives a computationally tractable subset of the full frontier 𝒻*_Σ(P₀)
5. **Feasibility/Novelty Checks**: Use this approximated frontier for Equations (1-2) validation

**Computational Justification**: The full frontier computation would require dynamic graph construction during training, which is incompatible with JAX's compilation model. This backwards-focused approximation ensures that at least the states relevant to the current skill proposal are properly validated while maintaining JAX compatibility.

## 4. LLM Integration and Skill Proposal

### Graph-Based Prompt System
The LLM integration uses a sophisticated prompt graph system:

```python
# AgentKit graph nodes for prompt composition
- "next_task": Proposes new skills based on current frontier
- "implementation": Generates reward/verifier functions  
- "skill_with_consumption": Defines requirements/gains in lambda notation
- "code_generation": Creates executable Python functions
```

### Skill Proposal Pipeline
1. **Frontier Analysis**: Determine what new skills can extend current capabilities
2. **LLM Proposal**: Generate skill contracts (requirements → gains) 
3. **Symbolic Verification**: Check feasibility (Eq. 2) and novelty (Eq. 1) constraints
4. **Code Generation**: Create reward function `r_𝒵`, verifier `κ_𝒵`, and network selector
5. **RL Training**: Learn executable policies using mixture-of-experts PPO

### Knowledge Base Integration
The system maintains a knowledge base containing:
- Crafting recipes and resource dependencies
- Previously learned skill patterns
- Failure cases and their causes
- Used to bias LLM proposals toward feasible skills

## 5. Multi-Task RL Training with PPO Flow

### Mixture-of-Experts Architecture
```python
# PPO Flow training uses MoE networks:
network = ActorCriticMoE(
    action_dim=env.action_space.n,     # 7 actions (move, mine, craft, etc.)
    layer_width=512,                   # Hidden layer size
    num_layers=4,                      # Network depth  
    num_tasks=num_heads               # One expert per skill
)
```

### Skill Indexing and Routing
```python
# Map player state to skill index for expert selection
def map_player_state_to_skill(player_state):
    player_state_one_hot = jnp.eye(num_tasks_)[player_state]
    player_skill = (player_state_one_hot @ task_to_skill_index).astype(jnp.int32)
    return player_skill
```

### Training Loop Details
- **Success Criterion**: Transition from skill state `n` to state `n+1` with success rate ≥ 80%
- **Early Stopping**: Training terminates when success threshold reached or max timesteps exceeded
- **Optimistic Resets**: Environment resets to skill-appropriate starting states for efficiency

## 6. Knowledge Base Updates and Trajectory Analysis

### Trajectory Processing Pipeline
After successful skill execution:

1. **Trajectory Extraction**: Extract state sequence from skill start to completion
2. **Inventory Analysis**: Track item gains/losses at each timestep with actions
3. **LLM Analysis**: Use inventory graph to analyze what actually happened vs. planned
4. **Skill Refinement**: Update requirements/gains based on observed trajectories

### Example Trajectory Explanation
```python
# Trajectory analysis output for "Collect Wood" skill:
trajectory_explanation = [
    "Timestep 1: Action: MINE, Gained 1 wood",
    "Timestep 3: Action: MINE, Gained 1 wood", 
    "Timestep 7: Action: MINE, Gained 2 wood"
]
```

### Knowledge Base Update Mechanism
```python
# Update format applied after trajectory analysis:
kb_updates_applied = [
    {
        "path": ["Collect Wood", "requirements"],
        "old_requirements": {"inventory_space": "lambda n: 0*n + 4"},
        "new_requirements": {"inventory_space": "lambda n: 0*n + 1"}, 
        "reason": "Observed successful collection with minimal space"
    }
]
```

## 7. Checkpointing and Iteration Management

### Checkpoint System
The Flow controller implements sophisticated state management:

```python
# Checkpoint contains complete system state
checkpoint = {
    "current_i": 5,                    # Current iteration
    "skills": complete_skills_dict,    # All learned skills
    "db": knowledge_base,             # LLM knowledge base (minus prompts)
    "execution_order": skill_sequence  # Dependency resolution result
}
```

### Iteration Progression
1. **Skill Discovery**: Generate new skill from current frontier
2. **Training**: Learn policy using PPO Flow until success threshold
3. **Analysis**: Analyze trajectories and update knowledge base
4. **Checkpointing**: Save complete state for potential rollback
5. **Integration**: Add skill to library and recompute frontier

## 8. Practical Training Details

### Hyperparameters
```python
# Key training parameters
NUM_ENVS = 1024                    # Parallel environments
TOTAL_TIMESTEPS = 100M             # Max training per skill
SUCCESS_STATE_RATE = 0.8           # Required success rate (80%)
LAYER_SIZE = 512                   # Network width
NUM_STEPS = 64                     # PPO rollout length
UPDATE_EPOCHS = 4                  # PPO update epochs
```

### Reward Ensemble Strategy
To mitigate reward hacking:
1. Generate multiple reward function variants
2. Train separate policies for each variant
3. Prune policies with success rate < threshold
4. Select final policy based on sample efficiency

### Environment Adaptations
- **Optimistic Resets**: Reset to states matching skill preconditions
- **Player State Tracking**: 18 discrete states for skill progression
- **Achievement Integration**: 21 achievements as auxiliary rewards
- **Inventory Capacity**: 9-item limit per resource type enforced

## 9. Failure Handling and Robustness

### Cycle Detection
The dependency resolver includes cycle detection to prevent infinite loops in skill dependencies.

### Code Validation
All LLM-generated code goes through multiple validation stages:
1. **Syntax Checking**: Ensure valid Python syntax
2. **Function Testing**: Verify reward/verifier functions work with test states
3. **Semantic Validation**: Check that functions align with skill specifications

### Fallback Mechanisms
- **Legacy Format Loading**: Graceful handling of old checkpoint formats
- **Graceful Degradation**: Continue operation even with missing knowledge base entries
- **Error Recovery**: Restart graph evaluation on LLM failures

## 10. Integration with Broader Experimental Pipeline

### Wandb Logging
Comprehensive experiment tracking including:
- Success rates per skill
- Training statistics
- Inventory distributions
- Achievement progress
- Skill dependency graphs

### Model Persistence
Trained policies saved with complete configuration:
```python
# Policy checkpoint includes:
- Network parameters (JAX arrays)
- Training configuration 
- Skill metadata
- Success statistics
```

This implementation provides a complete instantiation of the SCALAR algorithm, demonstrating how the theoretical framework translates into practical skill discovery and composition in a complex symbolic environment.