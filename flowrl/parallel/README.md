# Parallel Skill Training

This module provides infrastructure for training multiple skills in parallel using tmux orchestration.

## Components

### 1. `training_setup.py`
Prepares training runs with network remapping for efficient memory usage.

**Key Functions:**
- `build_remapping()`: Creates global→local expert index mappings
- `prepare_training_run()`: Sets up training folder with remapped experts

**Example:**
```python
from flowrl.parallel.training_setup import prepare_training_run

run_folder, module_path, policies_folder = prepare_training_run(
    skill_name="Make_Pickaxe",
    skill_data=skill_data_from_llm,
    global_expert_idx=2,
    dependency_skill_names=["Collect_Wood", "Collect_Stone"],
    completed_skills={
        "Collect_Wood": {"expert_idx": 0},
        "Collect_Stone": {"expert_idx": 1}
    },
    base_dir=Path("exp/bottom_up/")
)
```

### 2. `training_processor.py`
Processes completed training runs with frame-count heuristic for conflict resolution.

**Key Functions:**
- `process_completed_training()`: Extracts and merges expert networks
- `save_expert_to_skills()`: Saves individual experts to skills/ folders

**Example:**
```python
from flowrl.parallel.training_processor import process_completed_training

results = process_completed_training(
    run_folder=Path("training_runs/skill_2_Make_Pickaxe/"),
    skill_name="Make_Pickaxe",
    global_expert_idx=2,
    base_dir=Path("exp/bottom_up/"),
    total_timesteps=100_000_000
)

# Results show which experts were updated
# results["expert_updates"] = {
#     0: {"action": "updated", "frames": 150_000_000},
#     1: {"action": "updated", "frames": 140_000_000},
#     2: {"action": "new", "frames": 100_000_000}
# }
```

### 3. `scheduler.py`
Orchestrates parallel skill training with dependency resolution.

**Key Class: `SkillScheduler`**
- Manages skill queue (waiting, running, completed, failed)
- Assigns expert indices dynamically
- Launches tmux sessions for training
- Monitors completion and triggers processing

**Example:**
```python
from flowrl.parallel.scheduler import SkillScheduler
from flowrl.parallel.training_setup import prepare_training_run
from flowrl.parallel.training_processor import process_completed_training

# Initialize scheduler
scheduler = SkillScheduler(
    base_dir=Path("exp/bottom_up/"),
    max_parallel=3,
    poll_interval=30  # seconds
)

# Set callbacks
scheduler.set_callbacks(
    on_skill_complete=lambda skill_name: flow.on_skill_complete(skill_name),
    prepare_training_run=prepare_training_run,
    process_completed_training=process_completed_training
)

# Add skills
scheduler.add_skill("Collect_Wood", skill_data_1, dependencies=[])
scheduler.add_skill("Collect_Stone", skill_data_2, dependencies=[])
scheduler.add_skill("Make_Pickaxe", skill_data_3, dependencies=["Collect_Wood", "Collect_Stone"])

# Run scheduler (blocks until all complete)
scheduler.run(completed_skills={})
```

## Architecture Overview

### Folder Structure

```
exp/bottom_up/
├── skills/                          # Permanent storage
│   ├── 0_Collect_Wood/
│   │   ├── expert_0_policy/         # Individual expert network
│   │   │   └── params.pkl
│   │   ├── 0.py                     # Archived module
│   │   └── training.log
│   ├── 1_Collect_Stone/
│   │   └── expert_1_policy/...
│   └── 2_Make_Pickaxe/
│       └── expert_2_policy/...
│
├── training_runs/                   # Temporary (active training only)
│   └── skill_2_Make_Pickaxe/
│       ├── 2.py                     # Generated with remapping
│       ├── 2_policies/              # Full MoE (local indices)
│       │   └── checkpoint_*.pkl
│       └── training.log
│
├── checkpoints/
│   ├── global_latest.pkl            # Metadata only
│   └── global.lock                  # For concurrent processing
│
└── scheduler_state.json              # Scheduler state
```

### Network Remapping

The system uses **contiguous local indices** during training for memory efficiency.

**Example:**
- Global expert indices: 0, 5, 10 (sparse)
- Local indices during training: 0, 1, 2 (contiguous)

This avoids creating unused expert networks (e.g., experts 1-4, 6-9).

**Remapping in generated module:**
```python
# Auto-generated in training_runs/skill_10_Test/10.py
GLOBAL_TO_LOCAL = {0: 0, 5: 1, 10: 2}
LOCAL_TO_GLOBAL = {0: 0, 1: 5, 2: 10}

# Module uses local indices (0, 1, 2) during training
```

### Frame-Count Heuristic

When multiple skills train in parallel and update shared dependency experts, conflicts are resolved by keeping the expert version with **most training frames**.

**Example:**
```
Initial: expert_0 has 50M frames

Parallel training:
- Skill_A trains expert_0 for 80M steps → 130M total
- Skill_B trains expert_0 for 100M steps → 150M total

Skill_A completes first: expert_0 updated to 130M
Skill_B completes second: expert_0 updated to 150M (overwrites A)

Result: Keep Skill_B's version (150M > 130M)
```

## Integration with bottomup.py

**Event-driven generation loop:**

```python
from flowrl.parallel.scheduler import SkillScheduler
from flowrl.parallel.training_setup import prepare_training_run
from flowrl.parallel.training_processor import process_completed_training

# Initialize
scheduler = SkillScheduler(args, base_dir=args.graph_path)
scheduler.set_callbacks(
    on_skill_complete=flow_graph.on_skill_complete,
    prepare_training_run=prepare_training_run,
    process_completed_training=process_completed_training
)

# Start scheduler in background thread
import threading
scheduler_thread = threading.Thread(target=lambda: scheduler.run(flow_graph.skills), daemon=True)
scheduler_thread.start()

# Event-driven generation
total_skills_generated = 0
frontier_blocked = False

while total_skills_generated < args.max_nodes or not scheduler._is_complete():
    # Check available GPU slots
    available_slots = scheduler.max_parallel - len(scheduler.state["currently_running"])

    # Generate if not blocked
    if available_slots > 0 and not frontier_blocked:
        # Generate next skill
        skill_name, skill_data = flow_graph.next_skill()

        # Check if frontier is blocked
        frontier_blocked = flow_graph.check_frontier_blocked(skill_name, skill_data)

        # Add to scheduler
        dependencies = get_dependencies(skill_name, flow_graph.skills)
        scheduler.add_skill(skill_name, skill_data, dependencies)
        total_skills_generated += 1

        if frontier_blocked:
            print("Frontier blocked, waiting for skills to complete...")

    # Monitor completions
    if scheduler.get_completed_skill_names():
        frontier_blocked = False  # Unblock frontier

    time.sleep(args.scheduler_poll_interval)

# Wait for all to finish
scheduler_thread.join()
```

## Testing

Run tests:
```bash
python -m pytest tests/parallel/ -v
```

Test coverage:
- `test_training_setup.py`: Remapping logic
- `test_training_processor.py`: Frame-count heuristic
- `test_scheduler.py`: Scheduling and state management
- `test_integration.py`: End-to-end workflows

## Configuration

**Command-line flags (to be added to bottomup.py):**
```bash
--max_parallel_skills 3           # Number of parallel training sessions
--scheduler_poll_interval 30      # Seconds between checks
--max_retries 2                   # Retries for failed skills
```

## Monitoring

**View scheduler state:**
```bash
cat exp/bottom_up/scheduler_state.json | jq
```

**List active tmux sessions:**
```bash
tmux list-sessions | grep flowrl_
```

**Attach to a training session:**
```bash
tmux attach -t flowrl_e2_Make_Pickaxe
```

**Watch progress:**
```bash
watch -n 5 'cat exp/bottom_up/scheduler_state.json | jq ".skills | group_by(.status) | map({status: .[0].status, count: length})"'
```
