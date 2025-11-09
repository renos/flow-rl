# Phase 3 Architecture Review - Critical Findings

## Current System Architecture

### How Module → MoE Works

**1. Module Structure:**
```python
# Example: exp/premade/1.py
def task_0_is_done(...): ...
def task_0_reward(...): ...
def task_0_network_number():
    return 0  # Global expert index

def task_1_is_done(...): ...
def task_1_reward(...): ...
def task_1_network_number():
    return 2  # Global expert index (sparse!)
```

**2. Code Parser (`Craftax/craftax/craftax/util/code_parser.py`):**
```python
def task_and_reward_funcs(module_dict):
    task_to_skill_index = task_to_skill(module_dict)  # e.g., [0, 2, 3, 0, 4, 9, 6]
    num_skills = max(task_to_skill_index) + 1  # e.g., 10 (even if only 7 unique experts)
    common_heads = jnp.arange(num_skills)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    num_common_heads = num_skills  # 10
    return (..., common_heads, num_common_heads, task_to_skill_index)
```

**3. MoE Initialization (`ppo_flow.py` + `actor_critic.py`):**
```python
# ppo_flow.py line 148
network = ActorCriticMoE(
    action_dim=env.action_space(env_params).n,
    layer_width=config["LAYER_SIZE"],
    num_layers=4,
    num_tasks=num_heads,  # num_heads = num_common_heads = 10
)

# actor_critic.py line 625-632
class ActorCriticMoE(nn.Module):
    num_tasks: int  # 10

    def setup(self):
        self.actor_networks = [
            self.create_network(is_actor=True)
            for _ in range(self.num_tasks)  # Creates 10 complete networks!
        ]
        # Even if experts 1, 5, 8 are never used!
```

**4. Parameter Loading (`ppo_flow.py` line 136-141):**
```python
def param_updater(new_state):
    if "params" not in prev_state:
        return new_state
    for key, value in prev_state["params"]["params"].items():
        new_state["params"][key] = value  # Copies expert_0, expert_2, etc.
    return new_state
```

### Key Insight: Sparse Expert Indices

**Current system already handles sparse indices!**
- If module has experts [0, 2, 3, 9], the MoE creates 10 experts (0-9)
- Experts 1, 4, 5, 6, 7, 8 are initialized randomly but never trained
- This wastes memory but keeps the code simple

---

## Parallel Training: Two Approaches

### Approach A: Keep Sparse Indices ❌ INEFFICIENT

**How it works:**
```python
# Skills: expert_0, expert_2, expert_5 are completed
# New skill (expert_7) depends on [expert_0, expert_2]

# Training run setup:
run_folder = "training_runs/skill_7_Make_Pickaxe/"
module_path = f"{run_folder}/7.py"

# Module generation:
def task_0_network_number():
    return 0  # Global index
def task_1_network_number():
    return 2  # Global index
def task_2_network_number():
    return 7  # Global index (new skill)

# MoE initialization:
num_tasks = 8  # max(0, 2, 7) + 1 = 8
# Creates 8 experts, but only 0, 2, 7 are used
# Experts 1, 3, 4, 5, 6 are wasted memory
```

**Problems:**
- Wastes memory on unused experts
- Scales poorly (if expert_50 depends on expert_0, creates 51 experts)
- GPU memory issues with many skills

### Approach B: Remap to Contiguous ✅ EFFICIENT (RECOMMENDED)

**How it works:**
```python
# Skills: expert_0, expert_2, expert_5 are completed
# New skill (expert_7) depends on [expert_0, expert_2]

# Training run setup:
run_folder = "training_runs/skill_7_Make_Pickaxe/"
module_path = f"{run_folder}/7.py"

# Build remapping:
global_to_local = {0: 0, 2: 1, 7: 2}  # Map global → local indices
local_to_global = {0: 0, 1: 2, 2: 7}  # Map local → global indices

# Module generation with injected remapping:
"""
# Injected at top of module
GLOBAL_TO_LOCAL = {0: 0, 2: 1, 7: 2}
LOCAL_TO_GLOBAL = {0: 0, 1: 2, 2: 7}

def task_0_network_number():
    return 0  # Returns LOCAL index (already remapped)

def task_1_network_number():
    return 1  # Returns LOCAL index (already remapped)

def task_2_network_number():
    return 2  # Returns LOCAL index (new skill, also remapped)
"""

# MoE initialization:
num_tasks = 3  # Only 3 experts needed!
# Creates exactly 3 experts: expert_0, expert_1, expert_2

# Load params:
checkpoint = {
    "params": {
        "actor_networks_0": load("skills/0_Collect_Wood/expert_0_policy/"),
        "actor_networks_1": load("skills/2_Collect_Stone/expert_2_policy/"),
        "actor_networks_2": random_init(),  # New expert
        "critic_networks_0": ...,
        "critic_networks_1": ...,
        "critic_networks_2": ...,
    }
}
```

**Benefits:**
- ✅ Efficient memory usage (only 3 experts instead of 8)
- ✅ Scales well (50 skills with 3 dependencies = only 4 experts)
- ✅ Faster training (less unused parameters)

**Challenge:**
- Need to modify LLM-generated module to use local indices
- Need to inject GLOBAL_TO_LOCAL mapping
- Need to wrap/remap task_X_network_number() functions

---

## Updated Plan: Remapping Strategy

### Module Generation Changes

**Current LLM output:**
```python
def task_0_network_number():
    return 0  # LLM outputs global expert index
```

**After injection:**
```python
# === INJECTED HEADER ===
GLOBAL_TO_LOCAL = {0: 0, 2: 1, 7: 2}
LOCAL_TO_GLOBAL = {0: 0, 1: 2, 2: 7}

# Store original functions
_original_task_0_network_number = lambda: 0
_original_task_1_network_number = lambda: 2
_original_task_2_network_number = lambda: 7

# Create remapped versions
def task_0_network_number():
    return GLOBAL_TO_LOCAL[0]  # Returns 0

def task_1_network_number():
    return GLOBAL_TO_LOCAL[2]  # Returns 1

def task_2_network_number():
    return GLOBAL_TO_LOCAL[7]  # Returns 2
# === END INJECTED HEADER ===

# ... rest of LLM-generated code ...
```

### Implementation Requirements

**1. LLM Code Generation (flowrl/llm/flow.py):**
- Currently generates module with global expert indices
- Store skill's global expert index in skill_data
- When creating training run, don't modify LLM output yet

**2. Training Run Setup (new: flowrl/parallel/training_setup.py):**
```python
def prepare_training_run(skill_name, skill_data, dependencies, global_expert_idx, completed_skills):
    # Build remapping
    global_to_local, local_to_global = build_remapping(dependencies, global_expert_idx, completed_skills)

    # Load LLM-generated code from skill_data
    llm_code = skill_data["code"]

    # Inject remapping header
    remapped_module = inject_remapping_header(llm_code, global_to_local, local_to_global)

    # Save remapped module
    with open(f"{run_folder}/{global_expert_idx}.py", 'w') as f:
        f.write(remapped_module)

    # Build MoE with only required experts
    num_experts = len(global_to_local)  # Contiguous: 3 experts
    moe_checkpoint = build_moe_checkpoint(dependencies, global_to_local, completed_skills)

    return run_folder, module_path
```

**3. Post-Training Processing:**
- Load final checkpoint with local indices [0, 1, 2]
- Use LOCAL_TO_GLOBAL to map back: [0→0, 1→2, 2→7]
- Save expert_0, expert_2, expert_7 to respective skills/ folders

---

## Verification: Does This Work?

**Q: Will param_updater work with remapped indices?**
A: YES! The param_updater copies params by key name:
```python
new_state["params"]["actor_networks_0"] = prev_state["params"]["actor_networks_0"]
new_state["params"]["actor_networks_1"] = prev_state["params"]["actor_networks_1"]
# etc.
```

Since we load the checkpoint with correct key names, it will work!

**Q: Will the environment work with remapped indices?**
A: YES! The environment calls `task_X_network_number()` which returns local indices after remapping.

**Q: Will checkpoint saving/loading work?**
A: YES! We just need to track the remapping metadata to convert back to global indices.

---

## Conclusion

**The remapping approach (Approach B) is the correct design.**

The Phase 3 plan is sound, but needs clarification on:
1. LLM generates code with global indices initially
2. Remapping injection happens during `prepare_training_run()`
3. Training uses local indices (0, 1, 2, ...)
4. Post-processing maps back to global indices

The key insight: **ppo_flow.py doesn't need changes** - we just set up the module and checkpoint correctly!
