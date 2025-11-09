# Phase 3: Parallel Training Architecture - Detailed Plan

## Current Status
**Last Updated**: 2025-10-06

**Phases 1-2**: ✅ COMPLETE
- Infrastructure for tracking training vs completed skills
- Frontier-based dependency detection

**Phase 3**: 🔄 IN PLANNING
- Tmux orchestration
- Network propagation logic
- Checkpoint merging with frame-count heuristic

---

## Core Challenge: Network Propagation in Parallel Training

### Current Sequential Behavior

```python
# Sequential training (current)
Train skill_0:
  - Creates expert_0
  - Saves checkpoint with: {expert_0: params_0}

Train skill_1:
  - Loads checkpoint: {expert_0: params_0}
  - Creates expert_1
  - Trains both experts (continual learning)
  - Saves checkpoint with: {expert_0: params_0', expert_1: params_1}

Train skill_2:
  - Loads checkpoint: {expert_0: params_0', expert_1: params_1}
  - Creates expert_2
  - Trains all three experts (continual learning)
  - Saves checkpoint with: {expert_0: params_0'', expert_1: params_1', expert_2: params_2}
```

**Key insight**: Each skill trains ALL previous experts (continual learning)

### Parallel Training Conflict

```python
# Parallel training (problematic)
Load base checkpoint: {expert_0: params_0, expert_1: params_1}

Launch skill_A in parallel:
  - Creates expert_2
  - Trains experts {0, 1, 2}
  - Saves: {expert_0: A_0, expert_1: A_1, expert_2: A_2}

Launch skill_B in parallel:
  - Creates expert_3
  - Trains experts {0, 1, 3}
  - Saves: {expert_0: B_0, expert_1: B_1, expert_3: B_3}

Merge conflict:
  - expert_0: A_0 vs B_0? (both updated from params_0)
  - expert_1: A_1 vs B_1? (both updated from params_1)
```

**Problem**: Which version of shared experts to keep?

---

## Solution Architecture

### 1. Folder Structure

```
exp/bottom_up/
├── checkpoints/
│   ├── global_base.pkl              # Base checkpoint before parallel batch
│   ├── global_latest.pkl            # Latest merged checkpoint
│   └── skill_metadata.json          # Expert assignments and frame counts
│
├── skills/                          # Completed skills (permanent storage)
│   ├── 0_Collect_Wood/
│   │   ├── expert_0_policy/         # THIS skill's expert network ONLY (not full MoE)
│   │   │   └── params.pkl
│   │   ├── frame_count.json         # Total frames trained for expert_0
│   │   ├── training.log
│   │   └── video.mp4
│   │
│   ├── 1_Collect_Stone/
│   │   ├── expert_1_policy/         # THIS skill's expert network ONLY
│   │   │   └── params.pkl
│   │   └── ...
│   │
│   ├── 2_Make_Pickaxe/
│   │   ├── expert_2_policy/         # THIS skill's expert network ONLY
│   │   │   └── params.pkl
│   │   └── ...
│
├── training_runs/                   # Active training (temporary)
│   ├── skill_0_Collect_Wood/
│   │   ├── 0.py                     # Generated execution graph (remapped network numbers)
│   │   ├── 0_policies/              # Full MoE with REMAPPED indices
│   │   │   └── params.pkl           # Contains: {expert_0: wood_params}
│   │   └── training.log
│   │
│   ├── skill_2_Make_Pickaxe/
│   │   ├── 2.py                     # Generated execution graph (remapped: 0=wood, 1=stone, 2=pickaxe)
│   │   ├── 2_policies/              # Full MoE with REMAPPED indices
│   │   │   └── params.pkl           # Contains: {expert_0: wood, expert_1: stone, expert_2: pickaxe}
│   │   └── training.log
│
└── scheduler_state.json              # Scheduler tracking
```

**Key Differences**:
- **skills/**: Store ONLY individual expert networks (not full MoE, no .py file)
- **training_runs/**: Temporary folder with .py file and full MoE with REMAPPED network indices
- **Network Remapping**: Training always uses contiguous indices (0, 1, 2, ...) regardless of global expert indices

### 2. Expert Assignment Strategy

**Option A: Pre-assign Expert Numbers (Sequential)**
```python
# Assign expert numbers during skill generation, before training
skill_0 → expert_0  # Generated first
skill_1 → expert_1  # Generated second
skill_2 → expert_2  # Generated third (even if skills 0,1 still training)

# Problem: Wastes expert slots if skill fails validation/training
```

**Option B: Assign Expert Numbers on Training Start** ✅ RECOMMENDED
```python
# Assign expert numbers when skill actually starts training
skill_0: starts training → assigned expert_0
skill_1: starts training → assigned expert_1
skill_2: waits (depends on skill_0) → not assigned yet

skill_0: completes → expert_0 finalized
skill_2: starts training → assigned expert_2

# Benefits:
# - No wasted expert slots
# - Expert numbers reflect actual training order
# - Failed skills don't consume expert indices
```

**Implementation**:
```python
class SkillScheduler:
    def __init__(self):
        self.next_expert_idx = 0
        self.expert_assignments = {}  # skill_name → expert_idx

    def assign_expert(self, skill_name: str) -> int:
        """Assign expert index when skill starts training"""
        if skill_name not in self.expert_assignments:
            expert_idx = self.next_expert_idx
            self.expert_assignments[skill_name] = expert_idx
            self.next_expert_idx += 1
            return expert_idx
        return self.expert_assignments[skill_name]
```

### 3. Network Remapping Strategy ⭐ KEY INSIGHT

**Problem**: Global expert indices (0, 1, 2, ..., 10, ..., 50) become sparse and inefficient
**Solution**: Remap to contiguous indices (0, 1, 2, ...) for each training run

**Example**:
```python
# Global skill assignments:
skill_0 (Collect Wood) → global expert_0
skill_1 (Collect Stone) → global expert_1
skill_5 (Make Pickaxe) → global expert_5  # Depends on skills 0, 1

# Training skill_5 (Make Pickaxe):
# Step 1: Identify required global experts
required_global_experts = [0, 1]  # Dependencies
new_global_expert = 5             # This skill

# Step 2: Create remapping (global → local)
remap = {
    0: 0,  # global expert_0 (wood) → local expert_0
    1: 1,  # global expert_1 (stone) → local expert_1
    5: 2,  # global expert_5 (pickaxe) → local expert_2
}

# Step 3: Load individual expert networks and build MoE
moe_params = {
    "expert_0": load_expert_params("skills/0_Collect_Wood/expert_0_policy/"),
    "expert_1": load_expert_params("skills/1_Collect_Stone/expert_1_policy/"),
    "expert_2": initialize_new_expert(),  # New expert for pickaxe
}

# Step 4: Update .py file to use remapped indices
# In 5.py (generated execution graph):
def get_expert_idx(state):
    if should_use_wood_skill(state):
        return 0  # Remapped from global expert_0
    elif should_use_stone_skill(state):
        return 1  # Remapped from global expert_1
    elif should_use_pickaxe_skill(state):
        return 2  # Remapped from global expert_5
```

**Benefits**:
- ✅ MoE always has contiguous indices (0, 1, 2, ...)
- ✅ Efficient memory usage (no sparse arrays)
- ✅ Training code doesn't care about global expert numbers
- ✅ Each training run is self-contained

**Implementation**:
```python
def build_training_moe(skill_name: str, global_expert_idx: int, dependencies: List[str], completed_skills: Dict):
    """
    Build MoE network for training with remapped indices.

    Args:
        skill_name: Name of skill to train
        global_expert_idx: This skill's global expert index
        dependencies: List of dependency skill names
        completed_skills: Dict of completed skills

    Returns:
        (moe_params, global_to_local_map, module_path)
    """
    # Step 1: Collect global expert indices
    global_experts = []
    for dep_skill in dependencies:
        dep_expert_idx = completed_skills[dep_skill]["expert_idx"]
        global_experts.append((dep_expert_idx, dep_skill))

    # Add this skill's new expert
    global_experts.append((global_expert_idx, skill_name))

    # Sort by global index for consistent ordering
    global_experts.sort(key=lambda x: x[0])

    # Step 2: Create remapping
    global_to_local = {}
    local_to_global = {}
    for local_idx, (global_idx, skill) in enumerate(global_experts):
        global_to_local[global_idx] = local_idx
        local_to_global[local_idx] = global_idx

    # Step 3: Load expert networks with remapped indices
    moe_params = {}
    for local_idx, (global_idx, skill) in enumerate(global_experts):
        if skill == skill_name:
            # New expert - initialize randomly
            moe_params[f"expert_{local_idx}"] = initialize_expert()
        else:
            # Load from completed skill
            expert_path = f"skills/{skill}/expert_{global_idx}_policy/params.pkl"
            moe_params[f"expert_{local_idx}"] = load_params(expert_path)

    # Step 4: Generate .py file with remapped indices
    module_path = generate_remapped_module(skill_name, global_to_local, local_to_global)

    return moe_params, global_to_local, local_to_global, module_path
```

**Generated Module Example** (5.py for Make Pickaxe):
```python
# Auto-generated with remapped indices
# Global mapping: {0→0 (wood), 1→1 (stone), 5→2 (pickaxe)}

GLOBAL_TO_LOCAL = {0: 0, 1: 1, 5: 2}
LOCAL_TO_GLOBAL = {0: 0, 1: 1, 2: 5}

def get_expert_idx(state):
    # Logic uses LOCAL indices (0, 1, 2)
    if wood_skill_condition(state):
        return 0  # Local expert_0 (global expert_0)
    elif stone_skill_condition(state):
        return 1  # Local expert_1 (global expert_1)
    else:
        return 2  # Local expert_2 (global expert_5)
```

### 4. Checkpoint Format with Frame Tracking

**Global Checkpoint** (checkpoints/global_latest.pkl):
```python
# Global checkpoint (metadata only, no network params)
global_checkpoint = {
    # Skill metadata (network params stored separately)
    "skills": {
        "Collect_Wood": {
            "expert_idx": 0,
            "skill_with_consumption": {...},
            "expert_path": "skills/0_Collect_Wood/expert_0_policy/",
            "total_frames": 50_000_000,
        },
        "Collect_Stone": {
            "expert_idx": 1,
            "skill_with_consumption": {...},
            "expert_path": "skills/1_Collect_Stone/expert_1_policy/",
            "total_frames": 30_000_000,
        },
        "Make_Pickaxe": {
            "expert_idx": 2,
            "skill_with_consumption": {...},
            "expert_path": "skills/2_Make_Pickaxe/expert_2_policy/",
            "total_frames": 80_000_000,
        },
    },

    # Database state (frontier, etc.)
    "db": {...},
}
```

**Individual Expert Storage** (skills/{skill}/expert_{idx}_policy/):
```python
# Each skill's expert is stored separately
# skills/0_Collect_Wood/expert_0_policy/params.pkl
{
    "params": expert_0_params,  # Just this expert's network
    "metadata": {
        "skill_name": "Collect_Wood",
        "global_expert_idx": 0,
        "total_frames": 50_000_000,
    }
}
```

**Training Run Checkpoint** (training_runs/{skill}/{skill}_policies/):
```python
# Full MoE for this training run (REMAPPED indices)
# training_runs/skill_2_Make_Pickaxe/2_policies/checkpoint.pkl
{
    "expert_params": {
        0: wood_params,    # Loaded from skills/0_Collect_Wood/expert_0_policy/
        1: stone_params,   # Loaded from skills/1_Collect_Stone/expert_1_policy/
        2: pickaxe_params, # New expert for this skill
    },

    # Remapping for this training run
    "global_to_local": {0: 0, 1: 1, 5: 2},  # Global→Local expert indices
    "local_to_global": {0: 0, 1: 1, 2: 5},  # Local→Global expert indices

    # Frame counts during THIS training run
    "run_frame_counts": {
        0: 100_000_000,  # expert_0 trained for 100M frames during this run
        1: 100_000_000,  # expert_1 trained for 100M frames during this run
        2: 100_000_000,  # expert_2 trained for 100M frames during this run
    },
}
```

**Key Design Decisions**:
- ✅ Global checkpoint: metadata only, no network params
- ✅ Individual experts: stored in skills/ folders
- ✅ Training run: full MoE with remapped indices
- ✅ Frame counts: tracked per expert during training

### 5. Training Run Setup (No ppo_flow.py Changes Needed!)

**Key Insight**: `ppo_flow.py` doesn't need to know about remapping. We just set up the training folder correctly.

**Setup Process**:
```python
def prepare_training_run(skill_name: str, global_expert_idx: int, dependencies: List[str], completed_skills: Dict):
    """
    Prepare training_runs/ folder with .py file and policies.
    ppo_flow.py will load this naturally without modifications.
    """
    # 1. Build remapping
    global_experts = []
    for dep_skill in dependencies:
        dep_expert_idx = completed_skills[dep_skill]["expert_idx"]
        global_experts.append((dep_expert_idx, dep_skill))
    global_experts.append((global_expert_idx, skill_name))
    global_experts.sort(key=lambda x: x[0])

    global_to_local = {g: i for i, (g, _) in enumerate(global_experts)}
    local_to_global = {i: g for i, (g, _) in enumerate(global_experts)}

    # 2. Create training_runs folder
    run_folder = f"training_runs/skill_{global_expert_idx}_{skill_name.replace(' ', '_')}"
    os.makedirs(run_folder, exist_ok=True)

    # 3. Load dependency experts and build MoE with REMAPPED indices
    moe_params = {}
    initial_frame_counts = {}

    for local_idx, (global_idx, skill) in enumerate(global_experts):
        if skill == skill_name:
            # New expert - will be initialized by ppo_flow.py
            moe_params[f"expert_{local_idx}"] = None  # Placeholder
            initial_frame_counts[global_idx] = 0
        else:
            # Load from skills/ folder
            expert_path = f"skills/{global_idx}_{skill}/expert_{global_idx}_policy/params.pkl"
            expert_ckpt = load_checkpoint(expert_path)
            moe_params[f"expert_{local_idx}"] = expert_ckpt["params"]
            initial_frame_counts[global_idx] = expert_ckpt["metadata"]["total_frames"]

    # 4. Save MoE policies with remapped indices
    policies_folder = f"{run_folder}/{global_expert_idx}_policies"
    os.makedirs(policies_folder, exist_ok=True)

    # Save initial checkpoint (ppo_flow.py will load and continue from this)
    checkpoint = {
        "expert_params": moe_params,  # Keys: expert_0, expert_1, ... (remapped)
        # Metadata for post-training processing
        "remapping_metadata": {
            "global_to_local": global_to_local,
            "local_to_global": local_to_global,
            "initial_frame_counts": initial_frame_counts,  # Frame counts before this run
        }
    }
    save_checkpoint(checkpoint, f"{policies_folder}/checkpoint_0.pkl")

    # 5. Generate .py file with remapped indices and save mapping
    module_content = generate_module_with_remapping(
        skill_name,
        skill_data,
        global_to_local,
        local_to_global
    )
    module_path = f"{run_folder}/{global_expert_idx}.py"
    with open(module_path, 'w') as f:
        f.write(module_content)

    return run_folder, module_path, policies_folder
```

**ppo_flow.py works unchanged because**:
- Loads .py file: already has remapped expert selection logic
- Loads policies: MoE already uses indices 0, 1, 2, ...
- Saves checkpoints: continues using indices 0, 1, 2, ...
- Tracks frames: metadata in checkpoint tracks everything needed

### 6. Post-Training: Extract and Merge

After training completes, extract individual experts and update global state.

```python
def process_completed_training(run_folder: str, skill_name: str, global_expert_idx: int):
    """
    Process completed training run:
    1. Extract individual expert networks to skills/ folder
    2. Update global checkpoint with frame counts
    3. Use frame-count heuristic for conflicts
    """
    # Load final training checkpoint
    policies_folder = f"{run_folder}/{global_expert_idx}_policies"
    final_ckpt = load_checkpoint(f"{policies_folder}/checkpoint_final.pkl")

    # Get remapping metadata
    metadata = final_ckpt["remapping_metadata"]
    global_to_local = metadata["global_to_local"]
    local_to_global = metadata["local_to_global"]
    initial_frame_counts = metadata["initial_frame_counts"]

    # Calculate frames trained during this run
    total_timesteps = final_ckpt.get("total_timesteps", 0)  # From ppo_flow training

    # Extract expert params (using local indices)
    local_expert_params = final_ckpt["expert_params"]

    print(f"\n=== Processing completed skill '{skill_name}' (global expert {global_expert_idx}) ===")
    print(f"Total timesteps trained: {total_timesteps:,}")

    # Load global checkpoint
    global_ckpt = load_checkpoint("checkpoints/global_latest.pkl")

    # Process each expert
    for local_idx, global_idx in local_to_global.items():
        expert_params = local_expert_params[f"expert_{local_idx}"]

        # Calculate new total frames for this expert
        initial_frames = initial_frame_counts.get(global_idx, 0)
        new_total_frames = initial_frames + total_timesteps

        # Find which skill owns this expert
        owning_skill = None
        for skill, skill_data in global_ckpt["skills"].items():
            if skill_data["expert_idx"] == global_idx:
                owning_skill = skill
                break

        if owning_skill:
            # Expert exists - compare frame counts
            existing_frames = global_ckpt["skills"][owning_skill]["total_frames"]

            if new_total_frames > existing_frames:
                # This version has more training → update
                print(f"  ✓ Expert {global_idx} ({owning_skill}): updating ({new_total_frames:,} > {existing_frames:,} frames)")
                save_expert_to_skills(global_idx, owning_skill, expert_params, new_total_frames)
                global_ckpt["skills"][owning_skill]["total_frames"] = new_total_frames
            else:
                # Keep existing version
                print(f"  → Expert {global_idx} ({owning_skill}): keeping existing ({existing_frames:,} ≥ {new_total_frames:,} frames)")
        else:
            # New expert for this skill
            print(f"  ✓ Expert {global_idx} (new): saving ({new_total_frames:,} frames)")
            skill_folder = f"skills/{global_expert_idx}_{skill_name.replace(' ', '_')}"
            os.makedirs(skill_folder, exist_ok=True)

            save_expert_to_skills(global_idx, skill_name, expert_params, new_total_frames)

            # Add to global checkpoint
            global_ckpt["skills"][skill_name] = {
                "expert_idx": global_expert_idx,
                "skill_with_consumption": final_ckpt.get("skill_with_consumption"),
                "expert_path": f"{skill_folder}/expert_{global_expert_idx}_policy/",
                "total_frames": new_total_frames,
            }

    # Save updated global checkpoint
    save_checkpoint(global_ckpt, "checkpoints/global_latest.pkl")

    # Move training_run to skills/ for archival
    skill_folder = f"skills/{global_expert_idx}_{skill_name.replace(' ', '_')}"
    shutil.copy(f"{run_folder}/{global_expert_idx}.py", f"{skill_folder}/{global_expert_idx}.py")
    shutil.copy(f"{run_folder}/training.log", f"{skill_folder}/training.log")

    # Clean up training_runs folder
    shutil.rmtree(run_folder)

    print(f"=== Processing complete ===\n")


def save_expert_to_skills(global_idx: int, skill_name: str, params, total_frames: int):
    """Save individual expert to skills/ folder"""
    skill_folder = f"skills/{global_idx}_{skill_name.replace(' ', '_')}"
    expert_folder = f"{skill_folder}/expert_{global_idx}_policy"
    os.makedirs(expert_folder, exist_ok=True)

    expert_ckpt = {
        "params": params,
        "metadata": {
            "skill_name": skill_name,
            "global_expert_idx": global_idx,
            "total_frames": total_frames,
        }
    }
    save_checkpoint(expert_ckpt, f"{expert_folder}/params.pkl")


```

---

## Training Flow Example

### Scenario: 3 Skills in Parallel

```
Skills:
  0. Collect_Wood (no deps) → expert_0
  1. Collect_Stone (no deps) → expert_1
  2. Make_Pickaxe (deps: [0, 1]) → expert_2

Timeline:

T=0: Generate skill_0, skill_1, skill_2
  - skill_0, skill_1: independent → assign expert_0, expert_1
  - skill_2: depends on [0, 1] → wait

T=1: Setup and launch skill_0
  - prepare_training_run():
    - Create training_runs/skill_0_Collect_Wood/
    - Generate 0.py with remapping: {0→0} (only this skill)
    - Create 0_policies/ with MoE: {expert_0: random_init}
  - Launch: python -m flowrl.ppo_flow --module_path training_runs/skill_0_Collect_Wood/0.py

T=1: Setup and launch skill_1 (parallel)
  - prepare_training_run():
    - Create training_runs/skill_1_Collect_Stone/
    - Generate 1.py with remapping: {1→0} (only this skill)
    - Create 1_policies/ with MoE: {expert_0: random_init}
  - Launch: python -m flowrl.ppo_flow --module_path training_runs/skill_1_Collect_Stone/1.py

T=15min: skill_0 completes (50M frames)
  - process_completed_training():
    - Extract expert_0 → skills/0_Collect_Wood/expert_0_policy/params.pkl
    - Update global: Collect_Wood: {expert_idx: 0, total_frames: 50M}
    - Clean up training_runs/skill_0_Collect_Wood/
  - flow.on_skill_complete("Collect_Wood")
  - skill_2 dependencies check: needs [0✓, 1✗] → still waiting

T=20min: skill_1 completes (40M frames)
  - process_completed_training():
    - Extract expert_1 → skills/1_Collect_Stone/expert_1_policy/params.pkl
    - Update global: Collect_Stone: {expert_idx: 1, total_frames: 40M}
  - flow.on_skill_complete("Collect_Stone")
  - skill_2 dependencies satisfied [0✓, 1✓] → assign expert_2, setup and launch

T=20min: Setup and launch skill_2
  - prepare_training_run():
    - Create training_runs/skill_2_Make_Pickaxe/
    - Remapping: {0→0 (wood), 1→1 (stone), 2→2 (pickaxe)}
    - Load experts:
      - expert_0: load from skills/0_Collect_Wood/expert_0_policy/ (50M frames)
      - expert_1: load from skills/1_Collect_Stone/expert_1_policy/ (40M frames)
      - expert_2: random_init (0 frames)
    - Generate 2.py with logic for 3 experts
    - Create 2_policies/ with MoE: {expert_0: wood, expert_1: stone, expert_2: pickaxe_init}
    - Save metadata: initial_frame_counts: {0: 50M, 1: 40M, 2: 0}
  - Launch: python -m flowrl.ppo_flow --module_path training_runs/skill_2_Make_Pickaxe/2.py

T=45min: skill_2 completes (100M frames trained)
  - process_completed_training():
    - Load final checkpoint with all 3 experts
    - Calculate new totals:
      - expert_0: 50M + 100M = 150M
      - expert_1: 40M + 100M = 140M
      - expert_2: 0 + 100M = 100M
    - Compare and update:
      - expert_0 (Collect_Wood): 150M > 50M → UPDATE ✓
      - expert_1 (Collect_Stone): 140M > 40M → UPDATE ✓
      - expert_2 (Make_Pickaxe): new → SAVE ✓
    - Extract:
      - skills/0_Collect_Wood/expert_0_policy/params.pkl (updated, 150M frames)
      - skills/1_Collect_Stone/expert_1_policy/params.pkl (updated, 140M frames)
      - skills/2_Make_Pickaxe/expert_2_policy/params.pkl (new, 100M frames)

Final State:
  skills/:
    0_Collect_Wood/expert_0_policy/ (150M frames)
    1_Collect_Stone/expert_1_policy/ (140M frames)
    2_Make_Pickaxe/expert_2_policy/ (100M frames)

  global_latest.pkl:
    skills: {
      "Collect_Wood": {expert_idx: 0, total_frames: 150M, ...},
      "Collect_Stone": {expert_idx: 1, total_frames: 140M, ...},
      "Make_Pickaxe": {expert_idx: 2, total_frames: 100M, ...}
    }
```

### Conflict Resolution Example

```
Two skills training in parallel that share dependencies:

Initial state:
  skills/0_Collect_Wood/expert_0_policy/ (100M frames)
  skills/1_Collect_Stone/expert_1_policy/ (50M frames)
  skills/2_Collect_Iron/expert_2_policy/ (70M frames)

skill_A (Make_Pickaxe): Depends on [Collect_Wood, Collect_Stone]
  - Assigned expert_3
  - Setup training_run:
    - Load expert_0 (100M frames), expert_1 (50M frames)
    - Initialize expert_3 (0 frames)
    - Remapping: {0→0, 1→1, 3→2}
  - Trains for 80M frames
  - Final: {expert_0: 180M, expert_1: 130M, expert_3: 80M}

skill_B (Make_Sword): Depends on [Collect_Wood, Collect_Iron]
  - Assigned expert_4
  - Setup training_run:
    - Load expert_0 (100M frames), expert_2 (70M frames)
    - Initialize expert_4 (0 frames)
    - Remapping: {0→0, 2→1, 4→2}
  - Trains for 60M frames
  - Final: {expert_0: 160M, expert_2: 130M, expert_4: 60M}

skill_A completes first (T=20min):
  - process_completed_training():
    - expert_0 (wood): 180M > 100M → UPDATE ✓
    - expert_1 (stone): 130M > 50M → UPDATE ✓
    - expert_3 (pickaxe): new → SAVE ✓
  - State after skill_A:
    - skills/0_Collect_Wood/expert_0_policy/ (180M frames)
    - skills/1_Collect_Stone/expert_1_policy/ (130M frames)
    - skills/3_Make_Pickaxe/expert_3_policy/ (80M frames)

skill_B completes second (T=22min):
  - process_completed_training():
    - expert_0 (wood): 160M < 180M → KEEP EXISTING ✓
    - expert_2 (iron): 130M > 70M → UPDATE ✓
    - expert_4 (sword): new → SAVE ✓
  - State after skill_B:
    - skills/0_Collect_Wood/expert_0_policy/ (180M frames) ← from skill_A
    - skills/1_Collect_Stone/expert_1_policy/ (130M frames)
    - skills/2_Collect_Iron/expert_2_policy/ (130M frames) ← updated
    - skills/3_Make_Pickaxe/expert_3_policy/ (80M frames)
    - skills/4_Make_Sword/expert_4_policy/ (60M frames)

Final global checkpoint:
  skills: {
    "Collect_Wood": {expert_idx: 0, total_frames: 180M},
    "Collect_Stone": {expert_idx: 1, total_frames: 130M},
    "Collect_Iron": {expert_idx: 2, total_frames: 130M},
    "Make_Pickaxe": {expert_idx: 3, total_frames: 80M},
    "Make_Sword": {expert_idx: 4, total_frames: 60M}
  }

Result: expert_0 (wood) uses the version from skill_A because it was trained
more (180M > 160M). All other experts are saved/updated appropriately.
```

**Why this works**: Frame count reflects total training, so expert with more training is likely better.

---

## Implementation Tasks

### Task 3.1: Training Run Setup (`flowrl/parallel/training_setup.py`)
- [ ] `prepare_training_run()` function
  - [ ] Build global→local remapping
  - [ ] Create training_runs/ folder structure
  - [ ] Load dependency expert networks from skills/
  - [ ] Build MoE checkpoint with remapped indices
  - [ ] Generate .py file with remapping metadata
- [ ] `generate_module_with_remapping()` function
  - [ ] Inject GLOBAL_TO_LOCAL and LOCAL_TO_GLOBAL dicts
  - [ ] Update expert selection logic to use local indices
  - [ ] Include skill execution graph

### Task 3.2: Post-Training Processing (`flowrl/parallel/training_processor.py`)
- [ ] `process_completed_training()` function
  - [ ] Load final checkpoint from training_runs/
  - [ ] Extract remapping metadata
  - [ ] Calculate updated frame counts per expert
  - [ ] Compare with global checkpoint (frame-count heuristic)
  - [ ] Save individual experts to skills/ folders
  - [ ] Update global checkpoint metadata
  - [ ] Archive training run to skills/
  - [ ] Clean up training_runs/
- [ ] File locking for concurrent merges

### Task 3.3: Skill Scheduler (`flowrl/parallel/scheduler.py`)
- [ ] `SkillScheduler` class
  - [ ] Expert assignment (assign on training start)
  - [ ] Skill state tracking (waiting, running, completed, failed)
  - [ ] Dependency resolution
  - [ ] Tmux session management
  - [ ] Monitor training progress (poll tmux)
  - [ ] Trigger processing on completion
  - [ ] Handle failures (retry logic)

### Task 3.4: Modify bottomup.py for Parallel Training
- [ ] Initialize SkillScheduler
- [ ] Event-driven generation loop:
  - [ ] Generate skills until frontier blocked
  - [ ] Queue skills with scheduler
  - [ ] Poll scheduler for completions
  - [ ] Call flow.on_skill_complete() when done
  - [ ] Continue generation after unblocking
- [ ] Graceful shutdown (wait for all skills)

### Task 3.5: Module Generation with Remapping
- [ ] Modify LLM-generated module creation
- [ ] Inject remapping dictionaries at top of .py file
- [ ] Update expert selection to use local indices
- [ ] Ensure reward/done functions use correct expert logic

### Task 3.6: Testing
- [ ] Unit tests for prepare_training_run()
- [ ] Unit tests for process_completed_training()
- [ ] Test frame-count conflict resolution
- [ ] Test remapping logic (global ↔ local indices)
- [ ] Integration test: 2 parallel independent skills
- [ ] Integration test: 1 skill depends on 2 parallel skills
- [ ] Stress test: 5+ skills with complex dependencies
- [ ] Test failure recovery and retry logic

---

## Open Questions

### Q1: What if skill training fails partway?
**Options**:
- A: Discard training_run, retry from scratch
- B: Keep partial checkpoint, resume training
- C: User decision via flag

**Recommendation**: A (discard and retry) for simplicity
- Failed runs leave training_runs/ folder untouched
- Scheduler can retry up to N times before giving up
- On final failure, mark skill as failed and continue

### Q2: Should we update ALL dependency experts during training?
**Current**: Yes (continual learning - all loaded experts get updated)
**Alternative**: Freeze dependency experts, only train new expert

**Recommendation**: Keep continual learning (current behavior)
- Pros: Better performance, experts improve over time
- Cons: More conflicts to resolve (mitigated by frame-count heuristic)

### Q3: How to handle GPU memory with many experts?
**Current**: Load all dependency experts into MoE
**Problem**: N experts × model_size could exceed GPU memory

**Solutions**:
- A: Parameter sharing (shared base + expert heads) - need architecture changes
- B: Expert pruning (remove rarely-used experts) - need usage tracking
- C: Dynamic loading (swap experts in/out) - complex
- D: Limit max experts per skill (fail if too many dependencies)

**Recommendation**: D for now (fail gracefully if >10 dependency experts)
- Most skills should have <5 dependencies
- Can implement A/B/C later if needed

### Q4: What if two skills finish at nearly the same time?
**Race condition**: Both try to update global checkpoint and skills/ folders

**Solution**: File locking in `process_completed_training()`
```python
import fcntl
import time

def process_completed_training_with_lock(run_folder, skill_name, global_expert_idx):
    lock_path = "checkpoints/global.lock"
    max_retries = 10

    for attempt in range(max_retries):
        try:
            with open(lock_path, 'w') as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                # Got lock - perform processing
                process_completed_training(run_folder, skill_name, global_expert_idx)
                return
        except IOError:
            # Lock held by another process - wait and retry
            time.sleep(1)

    raise Exception(f"Failed to acquire lock after {max_retries} attempts")
```

### Q5: How to handle .py file generation with remapping?
**Current**: LLM generates module with skill logic
**Challenge**: Need to inject remapping dicts and update expert indices

**Solution**: Template-based injection
```python
def generate_module_with_remapping(skill_name, skill_data, global_to_local, local_to_global):
    # Get LLM-generated code from skill_data
    llm_code = skill_data.get("code", "")

    # Template with remapping
    template = f'''
# Auto-generated remapping for {skill_name}
GLOBAL_TO_LOCAL = {global_to_local}
LOCAL_TO_GLOBAL = {local_to_global}

{llm_code}

# Original get_expert_idx likely uses global indices
# Wrap it to use local indices
_original_get_expert_idx = get_expert_idx

def get_expert_idx(state):
    global_idx = _original_get_expert_idx(state)
    return GLOBAL_TO_LOCAL.get(global_idx, global_idx)
'''

    return template
```

---

## Summary of Architecture

### Key Design Principles

1. **No ppo_flow.py Changes**: Training code stays unchanged, we just set up folders correctly
2. **Network Remapping**: Each training run uses contiguous indices (0, 1, 2, ...) regardless of global expert numbers
3. **Individual Expert Storage**: Each skill stores only its own expert in skills/ folder
4. **Frame-Count Heuristic**: Conflicts resolved by keeping expert version with most training
5. **Temporary Training Runs**: training_runs/ is ephemeral, cleaned up after processing
6. **Global Metadata Only**: Global checkpoint has no network params, just metadata and pointers

### Data Flow

```
1. Generate Skill → LLM proposes skill with requirements/gains
2. Check Frontier → Uses completed + training skills for feasibility
3. Assign Expert → Assign global expert index when dependencies satisfied
4. Setup Training Run:
   - Create training_runs/{skill}/
   - Build remapping (global → local indices)
   - Load dependency experts from skills/ folders
   - Create MoE with local indices
   - Generate .py with injected remapping
5. Launch Training → ppo_flow.py trains MoE normally
6. Process Completion:
   - Load final checkpoint with remapped experts
   - Calculate updated frame counts per global expert
   - Compare with existing (frame-count heuristic)
   - Extract individual experts to skills/ folders
   - Update global checkpoint metadata
   - Clean up training_runs/
7. Skill Complete → Move from training_skills to skills in Flow
8. Repeat → Generate next skills until frontier blocked
```

### Folder Structure Final State

```
exp/bottom_up/
├── checkpoints/
│   ├── global_latest.pkl           # Metadata only (no params)
│   └── global.lock                 # For concurrent processing
│
├── skills/                         # Permanent storage
│   ├── 0_Collect_Wood/
│   │   ├── expert_0_policy/
│   │   │   └── params.pkl          # Individual expert network
│   │   ├── 0.py                    # Archived from training_run
│   │   ├── training.log
│   │   └── video.mp4
│   │
│   ├── 1_Collect_Stone/
│   │   └── expert_1_policy/...
│   │
│   └── 2_Make_Pickaxe/
│       └── expert_2_policy/...
│
├── training_runs/                  # Temporary (active training only)
│   ├── skill_3_Mine_Iron/
│   │   ├── 3.py                    # Generated with remapping
│   │   ├── 3_policies/
│   │   │   └── checkpoint_*.pkl    # Full MoE (local indices)
│   │   └── training.log
│   └── (cleaned up after completion)
│
└── scheduler_state.json            # Scheduler tracking
```

## Next Steps

1. ✅ Review this plan (DONE)
2. Implement Task 3.1: Training run setup (`prepare_training_run()`)
3. Implement Task 3.2: Post-training processing (`process_completed_training()`)
4. Implement Task 3.3: Skill scheduler (SkillScheduler class)
5. Implement Task 3.4: Modify bottomup.py for parallel training
6. Implement Task 3.5: Module generation with remapping
7. Test with simple 2-skill scenario
8. Scale to full parallel training

**Estimated Timeline**:
- Task 3.1-3.2: 1 day
- Task 3.3: 1 day
- Task 3.4-3.5: 1 day
- Testing: 1 day
- **Total: ~4 days**
