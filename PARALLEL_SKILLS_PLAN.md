# Parallel Skill Learning: Implementation Plan

## Current Status

**Last Updated**: 2025-10-06

**Phase 1**: ✅ COMPLETE (infrastructure for tracking training vs completed skills)
**Phase 2**: ✅ COMPLETE (frontier-based dependency detection)
**Phase 3**: ⏸️ PENDING (tmux orchestration & checkpoint merging)
**Phase 4**: ⏸️ PENDING (integration & testing)
**Phase 5**: ⏸️ PENDING (monitoring & debugging)

**Recent Changes** (Phase 2):
- Modified `verify_skill_frontier_compliance()` to use all skills (completed + training) during LLM generation
- Added `check_frontier_blocked()` method to Flow class for post-generation frontier blocking detection
- Now properly distinguishes between skills that are independent vs dependent on training skills

**Phase 1 Changes**:
- Added `self.training_skills = {}` to Flow class for tracking in-progress skills
- Modified prompts to show currently training skills
- Added `on_skill_complete()` method to move skills from training to completed
- Updated prompts with instruction to prefer independent skills

---

## Overview

Extend the bottom-up skill learning system to propose multiple skills per iteration and train independent skills in parallel using tmux sessions.

**Current System**: Sequential skill generation and training (one skill → train → next skill)

**Target System**: Continuous scheduling with dynamic parallelism
- Generate skills in batches
- Analyze dependencies between skills
- **Continuously** launch skills as soon as:
  - Dependencies are satisfied
  - Resources (GPU slots) are available
- **No waiting** for entire "waves" to complete
- Incrementally merge checkpoints as skills finish

### Key Architectural Shift

**❌ Wave-Based (Initial Design)**
```
Generate 5 skills → Partition into waves → Train wave 1 → Wait for ALL to finish → Train wave 2
```
Problem: Blocked by slowest skill in each wave

**✅ Continuous Scheduling (Final Design)**
```
Generate skills → Add to queue → Launch immediately when deps satisfied → Merge on completion
```
Benefit: Maximum GPU utilization, skills don't block each other

### Why Continuous Scheduling?

**Problem with Wave-Based**:
```
Wave 0: [Collect Wood (10M steps, 15 min)] [Collect Stone (10M steps, 15 min)] [Make Diamond Pickaxe (100M steps, 3 hours)]
         └─ completes at 15min           └─ completes at 15min                └─ completes at 3 hours

Wave 0 doesn't finish until 3 hours (blocked by slowest skill)
Wave 1 can't start until 3 hours (2h45m of wasted GPU time!)
```

**Solution: Continuous Scheduling**:
```
T=0:    Launch: Collect Wood, Collect Stone, Make Diamond Pickaxe
T=15m:  Wood completes → Launch next independent skill (Eat Food)
T=15m:  Stone completes → Launch next independent skill (Sleep)
T=30m:  Eat Food completes → Launch dependent skill (Drink Water)
T=3h:   Make Diamond Pickaxe completes
        → All skills processed with ~95% GPU utilization
```

**Key Advantages**:
1. **Higher Throughput**: GPUs stay saturated (always N skills running)
2. **Lower Latency**: Fast skills don't wait for slow ones
3. **Better Resource Allocation**: Can generate new skills reactively based on what completes
4. **Fault Isolation**: One failed skill doesn't block unrelated skills

---

## Architecture Changes

### 1. Continuous Scheduling Experiment Structure

**Key Insight**: Don't wait for entire waves to complete - launch skills as soon as dependencies are satisfied and resources are available.

```
exp/bottom_up/
├── skills/
│   ├── 0_collect_wood/
│   │   ├── 0.py                    # Generated module
│   │   ├── 0_policies/             # Trained policies
│   │   ├── checkpoint.pkl          # Skill checkpoint
│   │   ├── video.mp4              # Success trajectory
│   │   └── training.log           # PPO training logs
│   ├── 1_collect_stone/
│   │   └── ...
│   ├── 2_make_pickaxe/            # Launched as soon as 0 & 1 complete
│   │   └── ...
├── scheduler_state.json            # Current state of all skills
└── checkpoints/
    ├── global_latest.pkl           # Incrementally updated as skills complete
    └── skill_0_collect_wood.pkl    # Per-skill checkpoints
```

#### scheduler_state.json Structure
```json
{
  "skills": {
    "0_collect_wood": {
      "skill_idx": 0,
      "skill_name": "Collect Wood",
      "status": "completed",
      "dependencies": [],
      "started_at": "2025-10-04T10:30:00",
      "completed_at": "2025-10-04T11:15:00",
      "success_rate": 0.85,
      "tmux_session": "flowrl_s0_collect_wood"
    },
    "1_collect_stone": {
      "skill_idx": 1,
      "skill_name": "Collect Stone",
      "status": "completed",
      "dependencies": [],
      "started_at": "2025-10-04T10:30:00",
      "completed_at": "2025-10-04T11:45:00",
      "success_rate": 0.82,
      "tmux_session": "flowrl_s1_collect_stone"
    },
    "2_make_pickaxe": {
      "skill_idx": 2,
      "skill_name": "Make Pickaxe",
      "status": "running",
      "dependencies": ["0_collect_wood", "1_collect_stone"],
      "started_at": "2025-10-04T11:45:00",
      "completed_at": null,
      "success_rate": null,
      "tmux_session": "flowrl_s2_make_pickaxe"
    },
    "3_eat_food": {
      "skill_idx": 3,
      "skill_name": "Eat Food",
      "status": "waiting",
      "dependencies": [],
      "started_at": null,
      "completed_at": null,
      "success_rate": null,
      "tmux_session": null
    }
  },
  "max_parallel": 3,
  "currently_running": ["2_make_pickaxe"]
}
```

---

## Core Components

### 2. Modified Single-Skill Generation (`flowrl/llm/flow.py`)

**Current**: `next_skill() -> (skill_name, skill_data)` ✅ Already exists!
**Changes**: Add `training_skills` tracking (6 lines total)

**What's Already There**:
- ✅ `self.skills = {}` (line 21)
- ✅ `self.db['skills_without_code']` populated (line 165)
- ✅ Validation loop with error retry (lines 166-178)
- ✅ Skill extraction and return (lines 189-200)

**What to Add**:

```python
class Flow:
    def __init__(self, args):
        # ... existing init ...
        self.skills = {}  # Already exists
        self.training_skills = {}  # ADD THIS LINE

    def next_skill(self):
        """Generate next skill (same signature as before)"""
        # ... existing code ...
        self.db['skills_without_code'] = {...}  # Already exists

        # ADD THIS: Show training skills too
        self.db['training_skills_without_code'] = {
            key: value["skill_with_consumption"]
            for key, value in self.training_skills.items()
        }

        # ... rest of existing generation code ...

        # ADD THIS: Store in training_skills instead of self.skills
        self.training_skills[skill_name] = skill_data

        return skill_name, skill_data

    def on_skill_complete(self, skill_name):  # NEW METHOD
        """Move skill from training to completed"""
        if skill_name in self.training_skills:
            skill_data = self.training_skills.pop(skill_name)
            self.skills[skill_name] = skill_data
            self.db["skills"][skill_name] = skill_data
```

**Prompt Modification** (in `flowrl/llm/craftax/prompts/main.py`):

Add after line 194:
```python
Currently Training Skills (do NOT propose duplicates):
```
$db.training_skills_without_code$
```
```

Add after line 231:
```python
- PREFER skills that don't depend on currently training skills (only depend on them if no other valuable skills exist)
```

---

### 3. Simplified Dependency Analysis (`flowrl/parallel_dependency_analyzer.py`)

**Goal**: Given a new skill, determine which completed skills it depends on.

```python
class DependencyAnalyzer:
    """
    Analyzes skill dependencies to determine which skills must complete
    before a new skill can be trained.
    """

    def __init__(self, skills):
        self.skills = skills

    def get_dependencies(self, skill_name: str, all_skills: dict) -> List[str]:
        """
        Determine which skills the given skill depends on.

        Args:
            skill_name: Name of the skill to analyze
            all_skills: Dict of all known skills {skill_name: skill_data}

        Returns:
            List of skill_ids (e.g., ["0_Collect_Wood", "1_Collect_Stone"])
            that must complete before this skill can be trained
        """
        if skill_name not in all_skills:
            return []

        skill_data = all_skills[skill_name]
        requirements = skill_data["skill_with_consumption"].get("requirements", {})

        dependencies = []

        # For each item this skill requires
        for req_item in requirements.keys():
            # Find which skill produces this item
            providing_skill = self._find_skill_providing_item(req_item, all_skills)

            if providing_skill and providing_skill != skill_name:
                dependencies.append(providing_skill)

        # Remove duplicates
        return list(set(dependencies))

    def _find_skill_providing_item(self, item_name: str, skills_dict: dict) -> str:
        """Find the skill that gains/produces the specified item."""
        for skill_name, skill_data in skills_dict.items():
            gain = skill_data["skill_with_consumption"].get("gain", {})
            if item_name in gain:
                return skill_name
        return None

    def can_train_now(self, skill_name: str, all_skills: dict,
                     completed_skills: List[str]) -> bool:
        """
        Check if a skill can be trained given currently completed skills.

        Args:
            skill_name: Name of skill to check
            all_skills: All known skills
            completed_skills: List of skill names that have completed

        Returns:
            True if all dependencies are satisfied
        """
        dependencies = self.get_dependencies(skill_name, all_skills)

        for dep in dependencies:
            if dep not in completed_skills:
                return False

        return True
```

**Simplified Approach**:
- No batching/partitioning needed
- Just check: "Does this new skill depend on anything?"
- Dependencies = skills that produce required items
- Let scheduler handle parallelism automatically

---

### 4. Continuous Scheduler (`flowrl/skill_scheduler.py`)

**Goal**: Continuously schedule skills as dependencies complete and resources free up.

```python
import subprocess
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple, Set
import datetime

class SkillScheduler:
    """
    Continuous scheduler for parallel skill training.

    Maintains:
    - Queue of pending skills (waiting for dependencies)
    - Set of running skills (currently training)
    - Set of completed skills

    Behavior:
    - As soon as a skill completes, merges its checkpoint
    - Checks which pending skills can now run
    - Launches new skills up to max_parallel limit
    """

    def __init__(self, args, base_dir):
        self.args = args
        self.base_dir = Path(base_dir)
        self.skills_dir = self.base_dir / "skills"
        self.skills_dir.mkdir(parents=True, exist_ok=True)

        self.max_parallel = args.max_parallel_skills
        self.poll_interval = args.scheduler_poll_interval  # seconds

        # Load or initialize scheduler state
        self.state_file = self.base_dir / "scheduler_state.json"
        if self.state_file.exists():
            self.state = self._load_state()
        else:
            self.state = {
                "skills": {},
                "max_parallel": self.max_parallel,
                "currently_running": [],
                "next_skill_idx": 0
            }

    def add_skill(self, skill_name: str, skill_data: dict, dependencies: List[str] = None):
        """
        Add a skill to the scheduler queue.

        Args:
            skill_name: Name of the skill
            skill_data: Skill data from Flow.next_skill()
            dependencies: List of skill_ids this skill depends on
        """
        skill_id = f"{self.state['next_skill_idx']}_{skill_name.replace(' ', '_')}"
        self.state["next_skill_idx"] += 1

        self.state["skills"][skill_id] = {
            "skill_idx": self.state["next_skill_idx"] - 1,
            "skill_name": skill_name,
            "skill_data": skill_data,
            "status": "waiting",
            "dependencies": dependencies or [],
            "started_at": None,
            "completed_at": None,
            "success_rate": None,
            "tmux_session": None
        }

        self._save_state()
        print(f"Added skill: {skill_id} (deps: {dependencies})")

    def run(self):
        """
        Main scheduler loop.

        Continuously:
        1. Check for completed skills
        2. Merge their checkpoints
        3. Launch newly-runnable skills
        4. Stop when all skills complete or fail
        """
        print(f"Starting scheduler (max_parallel={self.max_parallel})")

        while True:
            # Update status of running skills
            self._update_running_skills()

            # Try to launch new skills
            launched = self._launch_runnable_skills()

            # Check if we're done
            if self._is_complete():
                print("All skills completed!")
                break

            # Print status
            self._print_status()

            # Wait before next poll
            time.sleep(self.poll_interval)

    def _update_running_skills(self):
        """Check status of currently running skills."""
        for skill_id in list(self.state["currently_running"]):
            skill = self.state["skills"][skill_id]
            session = skill["tmux_session"]

            # Check if session is still alive
            if not self._is_session_alive(session):
                # Session finished
                success = self._check_training_success(skill)
                skill["status"] = "completed" if success else "failed"
                skill["completed_at"] = datetime.datetime.now().isoformat()
                skill["success_rate"] = self._extract_success_rate(skill)

                # Remove from running
                self.state["currently_running"].remove(skill_id)

                print(f"\n{'='*60}")
                print(f"Skill {skill_id} {skill['status']}!")
                print(f"  Success rate: {skill['success_rate']}")
                print(f"  Duration: {skill['started_at']} -> {skill['completed_at']}")
                print(f"{'='*60}\n")

                # Merge checkpoint into global state
                if success:
                    self._merge_skill_checkpoint(skill_id)

                self._save_state()

    def _launch_runnable_skills(self) -> int:
        """
        Launch skills that:
        1. Have all dependencies satisfied
        2. Fit within max_parallel limit

        Returns:
            Number of skills launched
        """
        launched = 0
        available_slots = self.max_parallel - len(self.state["currently_running"])

        if available_slots <= 0:
            return 0

        # Find runnable skills
        for skill_id, skill in self.state["skills"].items():
            if skill["status"] != "waiting":
                continue

            # Check dependencies
            if not self._are_dependencies_satisfied(skill_id):
                continue

            # Launch it!
            self._launch_skill(skill_id)
            launched += 1

            # Check if we hit the limit
            if len(self.state["currently_running"]) >= self.max_parallel:
                break

        return launched

    def _are_dependencies_satisfied(self, skill_id: str) -> bool:
        """Check if all dependencies of a skill are completed."""
        skill = self.state["skills"][skill_id]

        for dep_id in skill["dependencies"]:
            dep_skill = self.state["skills"].get(dep_id)

            if not dep_skill:
                print(f"Warning: Dependency {dep_id} not found for {skill_id}")
                return False

            if dep_skill["status"] != "completed":
                return False

        return True

    def _launch_skill(self, skill_id: str):
        """Launch training for a single skill."""
        skill = self.state["skills"][skill_id]

        # Create skill directory
        skill_dir = self.skills_dir / skill_id
        skill_dir.mkdir(parents=True, exist_ok=True)

        # Generate session name
        session_name = f"flowrl_s{skill['skill_idx']}_{skill['skill_name'].replace(' ', '_')}"

        # Build training command
        cmd = self._build_training_command(skill, skill_dir)

        # Launch tmux session
        self._launch_tmux_session(session_name, cmd, skill_dir)

        # Update state
        skill["status"] = "running"
        skill["started_at"] = datetime.datetime.now().isoformat()
        skill["tmux_session"] = session_name
        self.state["currently_running"].append(skill_id)

        print(f"Launched: {skill_id} (session: {session_name})")
        print(f"  Attach with: tmux attach -t {session_name}")

        self._save_state()

    def _is_complete(self) -> bool:
        """Check if all skills are done (completed or failed)."""
        for skill in self.state["skills"].values():
            if skill["status"] in ["waiting", "running"]:
                return False
        return True

    def _print_status(self):
        """Print current scheduler status."""
        waiting = sum(1 for s in self.state["skills"].values() if s["status"] == "waiting")
        running = len(self.state["currently_running"])
        completed = sum(1 for s in self.state["skills"].values() if s["status"] == "completed")
        failed = sum(1 for s in self.state["skills"].values() if s["status"] == "failed")

        print(f"[Status] Waiting: {waiting} | Running: {running}/{self.max_parallel} | " +
              f"Completed: {completed} | Failed: {failed}")

    def _merge_skill_checkpoint(self, skill_id: str):
        """
        Merge a completed skill's checkpoint into the global checkpoint.

        Handles conflicts by keeping expert versions trained on the most frames.
        """
        skill = self.state["skills"][skill_id]
        skill_dir = Path(skill["skill_dir"])
        ckpt_path = skill_dir / "checkpoint.pkl"

        if not ckpt_path.exists():
            print(f"Warning: Checkpoint not found for {skill_id}")
            return

        import pickle
        with open(ckpt_path, 'rb') as f:
            skill_ckpt = pickle.load(f)

        # Load global checkpoint
        global_ckpt_path = self.base_dir / "checkpoints" / "global_latest.pkl"
        if global_ckpt_path.exists():
            with open(global_ckpt_path, 'rb') as f:
                global_ckpt = pickle.load(f)
        else:
            global_ckpt = {
                "skills": {},
                "expert_params": {},
                "expert_frame_counts": {},  # Track frames per expert
                "db": {}
            }

        # Merge skill experts with frame-count heuristic
        skill_experts = skill_ckpt.get("expert_params", {})
        skill_frame_counts = skill_ckpt.get("expert_frame_counts", {})

        for expert_name, expert_params in skill_experts.items():
            skill_frames = skill_frame_counts.get(expert_name, 0)
            global_frames = global_ckpt["expert_frame_counts"].get(expert_name, 0)

            # Keep version trained on more frames
            if skill_frames > global_frames:
                global_ckpt["expert_params"][expert_name] = expert_params
                global_ckpt["expert_frame_counts"][expert_name] = skill_frames
                print(f"  Updated {expert_name}: {global_frames} -> {skill_frames} frames")
            else:
                print(f"  Kept existing {expert_name}: {global_frames} frames (skill had {skill_frames})")

        # Add new skill metadata
        global_ckpt["skills"].update(skill_ckpt.get("skills", {}))

        # Merge db
        for key, value in skill_ckpt.get("db", {}).items():
            if key not in ["prompts", "temp_prompts"]:
                global_ckpt["db"][key] = value

        # Save updated global checkpoint
        global_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(global_ckpt_path, 'wb') as f:
            pickle.dump(global_ckpt, f)

        print(f"  Merged {skill_id} into global checkpoint")

    def _get_session_name(self, skill_idx, skill_name):
        """Generate tmux session name."""
        clean_name = skill_name.replace(' ', '_').lower()
        return f"flowrl_w{self.wave_id}_s{skill_idx}_{clean_name}"

    def _build_training_command(self, skill_idx, skill_name, skill_data, skill_dir):
        """Build PPO training command."""
        # Write skill to module file
        module_path = skill_dir / f"{skill_idx}.py"
        # (Generate module using flow_graph.write_code() adapted for single skill)

        # Build command args
        cmd_parts = [
            "python -m flowrl.ppo_flow",
            f"--module_path {module_path}",
            f"--env_name {self.args.env_name}",
            f"--total_timesteps {self.args.total_timesteps}",
            f"--num_envs {self.args.num_envs}",
            f"--success_state_rate {self.args.success_state_rate}",
            f"--seed {self.args.seed + skill_idx}",  # Different seed per skill
            "--use_wandb" if self.args.use_wandb else "--no-use_wandb",
            f"--wandb_project {self.args.wandb_project}_wave{self.wave_id}" if self.args.use_wandb else "",
            f"> {skill_dir}/training.log 2>&1"  # Redirect output
        ]

        return " ".join(cmd_parts)

    def _launch_tmux_session(self, session_name, cmd, working_dir):
        """Launch a tmux session with the given command."""
        # Kill existing session if present
        subprocess.run(["tmux", "kill-session", "-t", session_name],
                      stderr=subprocess.DEVNULL)

        # Create new session
        subprocess.run([
            "tmux", "new-session", "-d", "-s", session_name,
            "-c", str(working_dir),  # Set working directory
            cmd
        ])

        print(f"Launched tmux session: {session_name}")
        print(f"  Attach with: tmux attach -t {session_name}")

    def _is_session_alive(self, session_name):
        """Check if tmux session is still running."""
        result = subprocess.run(
            ["tmux", "has-session", "-t", session_name],
            capture_output=True
        )
        return result.returncode == 0

    def _check_training_success(self, skill):
        """Check if training succeeded by examining output/checkpoints."""
        skill_dir = Path(skill["skill_dir"])

        # Check 1: Policy file exists
        policy_dir = skill_dir / f"{skill['skill_idx']}_policies"
        if not policy_dir.exists():
            return False

        # Check 2: Video file exists (trajectory found)
        video_path = skill_dir / "video.mp4"
        if not video_path.exists():
            return False

        # Check 3: Success rate in logs meets threshold
        # (Parse training.log for final success rate)

        return True

    def _extract_success_rate(self, skill):
        """Extract final success rate from training logs."""
        log_path = Path(skill["skill_dir"]) / "training.log"

        if not log_path.exists():
            return 0.0

        # Parse log file for success rate
        # (Implementation depends on log format)
        # For now, return dummy value
        return 0.8

    def _save_metadata(self):
        """Save wave metadata to JSON."""
        metadata_path = self.wave_dir / "wave_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, indent=2, fp=f)
```

---

### 5. Event-Driven Bottom-Up Loop (`flowrl/bottomup.py`)

**Key Changes**: Reactive generation with programmatic frontier detection

```python
# OLD (current - sequential)
while flow_graph.current_i < args.max_nodes:
    skill_name, skill_data = flow_graph.next_skill()
    flow_graph.add_skill(skill_name, skill_data)
    # ... train single skill ...
    flow_graph.current_i += 1

# NEW (event-driven parallel)
from flowrl.skill_scheduler import SkillScheduler
from flowrl.dependency_analyzer import DependencyAnalyzer

# Initialize
scheduler = SkillScheduler(args, base_dir=args.graph_path)
analyzer = DependencyAnalyzer(flow_graph.skills)

# Start scheduler in background thread
import threading
scheduler_thread = threading.Thread(target=scheduler.run, daemon=True)
scheduler_thread.start()

# Event-driven generation loop
total_skills_generated = 0
last_completed_count = 0
frontier_blocked = False

print(f"Starting event-driven skill generation (max: {args.max_nodes})")

while total_skills_generated < args.max_nodes or not scheduler._is_complete():
    # Check if we have available GPU slots
    available_slots = scheduler.max_parallel - len(scheduler.state["currently_running"])

    # Try to generate if:
    # 1. We have available slots
    # 2. Frontier is not blocked (last skill didn't depend on training)
    if (available_slots > 0 or len(scheduler.state["currently_running"]) == 0) and not frontier_blocked:
        # Get currently training skill names
        currently_training_names = [
            scheduler.state["skills"][sid]["skill_name"]
            for sid in scheduler.state["currently_running"]
        ]

        # Generate next skill
        print(f"\nAttempting to generate skill #{total_skills_generated + 1}...")
        print(f"  Available slots: {available_slots}/{scheduler.max_parallel}")
        print(f"  Currently training: {currently_training_names}")

        skill_name, skill_data = flow_graph.next_skill()
        # Note: skill is now in flow_graph.training_skills

        # Determine dependencies (check against ALL skills: completed + training)
        all_skills = {**flow_graph.skills, **flow_graph.training_skills}
        dependencies = analyzer.get_dependencies(skill_name, all_skills)

        # FRONTIER CHECK: Does this skill depend on currently training skills?
        training_skill_names = list(flow_graph.training_skills.keys())
        depends_on_training = any(dep in training_skill_names for dep in dependencies)

        if depends_on_training:
            # Frontier is blocked!
            blocking_deps = [d for d in dependencies if d in training_skill_names]
            print(f"  → Generated: {skill_name}")
            print(f"  → Dependencies: {dependencies}")
            print(f"  → FRONTIER BLOCKED: Depends on training skills {blocking_deps}")
            print(f"  → Pausing generation until skills complete...")

            # Add to scheduler (will launch when deps satisfied)
            scheduler.add_skill(skill_name, skill_data, dependencies)
            total_skills_generated += 1

            # Set flag to stop generating
            frontier_blocked = True
        else:
            # Independent skill or depends only on completed - frontier not blocked
            print(f"  → Generated: {skill_name} (deps: {dependencies})")

            # Add to scheduler queue
            scheduler.add_skill(skill_name, skill_data, dependencies)
            total_skills_generated += 1

            # Try to generate another immediately (loop continues)
            continue

    # Wait and monitor
    time.sleep(args.scheduler_poll_interval)

    # Check if any skills completed
    completed_count = sum(1 for s in scheduler.state["skills"].values()
                         if s["status"] == "completed")
    if completed_count > last_completed_count:
        last_completed_count = completed_count
        print(f"\n[Progress] {completed_count}/{total_skills_generated} skills completed")

        # Move completed skills from training to completed
        for skill_id, skill_info in scheduler.state["skills"].items():
            if skill_info["status"] == "completed":
                skill_name = skill_info["skill_name"]
                flow_graph.on_skill_complete(skill_name)

        # Frontier may be unblocked now - resume generation
        frontier_blocked = False
        print(f"  → Frontier unblocked - resuming generation")

# Wait for all skills to finish
print("\nAll skills generated. Waiting for training to complete...")
scheduler_thread.join()

print("All skills completed!")
print(f"Final checkpoint: {args.graph_path}/checkpoints/global_latest.pkl")
```

**How It Works**:

1. **Initial state**: Generate skills greedily
2. **After each generation**:
   - Check dependencies programmatically
   - If depends on training skills → **frontier blocked** → stop generating
   - If independent → continue generating
3. **When a skill completes**:
   - Move from `training_skills` to `skills`
   - **Unblock frontier** → resume generation
   - Scheduler also launches any waiting skills with satisfied dependencies
4. **No LLM "WAIT" decision**: We detect blocking programmatically (simpler, cheaper, more reliable)

---

## Implementation Phases

### Phase 1: Separate Training Skills ✅ COMPLETE

**Already Implemented** ✅:
- `next_skill()` method with validation loop
- `self.skills = {}` dictionary
- `self.db['skills_without_code']` populated
- Prompts show "Existing Skills: $db.skills_without_code$"
- Frontier checking and duplicate prevention
- Instructions: "Do NOT propose skills that already exist"

**Changes Completed** ✅:
- [x] Add `self.training_skills = {}` to `Flow.__init__()` in `flowrl/llm/flow.py:78`
- [x] Populate `self.db['training_skills_without_code']` in `next_skill()` at `flowrl/llm/flow.py:167`
- [x] Update `flowrl/llm/craftax/prompts/main.py:196-199`:
  ```python
  Currently Training Skills (do NOT propose duplicates):
  ```
  $db.training_skills_without_code$
  ```
  ```
- [x] Add instruction line at `flowrl/llm/craftax/prompts/main.py:239`:
  ```
  - PREFER skills that don't depend on currently training skills (only depend on them if no other valuable skills exist)
  ```
- [x] Add `on_skill_complete(skill_name)` method to `Flow` class at `flowrl/llm/flow.py:204-210`:
  ```python
  def on_skill_complete(self, skill_name):
      """Move skill from training to completed when training finishes"""
      if skill_name in self.training_skills:
          skill_data = self.training_skills.pop(skill_name)
          self.skills[skill_name] = skill_data
          self.db["skills"][skill_name] = skill_data
          print(f"Skill '{skill_name}' completed and moved to skills database")
  ```
- [ ] Test: Generate skill with training skills visible, verify no duplicates proposed

**Deliverable**: ✅ `Flow` tracks training vs completed skills separately (6 lines of changes)

---

### Phase 2: Frontier-Based Dependency Detection ✅ COMPLETE

**Already Implemented** ✅:
- **Frontier calculation**: `compute_frontier_summary()` in `flowrl/llm/craftax/symbolic_state.py`
- **Sufficiency checking**: `verify_skill()` in `flowrl/llm/craftax/symbolic_state.py:709`
  - Converts skills to symbolic operators
  - Checks if skill is novel (not already achievable)
  - Checks if skill is feasible (preconditions can be satisfied)
- **Used in after_queries**: `verify_skill_frontier_compliance()` in `flowrl/llm/craftax/after_queries.py:209`
  - Currently uses `self.node.db.get("skills", {})` for verification
  - Returns (is_novel, is_feasible, message)

**How It Works (User Clarification)**:

1. **During LLM generation** (in after_queries):
   - Use ALL skills (completed + training) for frontier calculation
   - Ensures LLM sees full reachable frontier
   - Prevents proposing skills already achievable with training skills

2. **After skill generation** (frontier blocking check):
   - Re-run sufficiency analysis using ONLY completed skills
   - If feasible with completed → independent skill → continue generating
   - If NOT feasible with completed → depends on training → frontier blocked

**Implementation Changes**:

**Change 1: Modify after_queries to use all skills during generation**
```python
# In flowrl/llm/craftax/after_queries.py:209
def verify_skill_frontier_compliance(self, parsed_answer):
    # ...existing code...

    # Get BOTH completed and training skills for frontier calculation
    completed_skills = self.node.db.get("skills", {})
    training_skills = self.node.db.get("training_skills", {})  # ADD THIS

    # Merge for full frontier view
    all_skills = {**completed_skills, **training_skills}  # ADD THIS

    # Use merged skills for verification
    is_novel, is_feasible = verify_skill(skill_name, parsed_answer, all_skills, max_capacity)  # CHANGE: use all_skills

    # ...rest of existing code...
```

**Change 2: Add frontier blocking check in flow.py after generation**
```python
# In flowrl/llm/flow.py after next_skill() returns

def check_frontier_blocked(self, skill_name: str, skill_data: dict) -> bool:
    """
    Check if skill depends on currently training skills.
    Returns True if frontier is blocked (skill depends on training).
    """
    from flowrl.llm.craftax.symbolic_state import verify_skill

    # Check feasibility using ONLY completed skills
    max_capacity = 99  # or get from env config
    _, is_feasible_with_completed = verify_skill(
        skill_name,
        skill_data["skill_with_consumption"],
        self.skills,  # Only completed skills
        max_capacity
    )

    # If NOT feasible with completed skills, must depend on training
    return not is_feasible_with_completed
```

**Usage in bottomup.py**:
```python
# After skill generation
skill_name, skill_data = flow_graph.next_skill()

# Check if frontier is blocked
frontier_blocked = flow_graph.check_frontier_blocked(skill_name, skill_data)

if frontier_blocked:
    print(f"Frontier blocked: {skill_name} depends on training skills")
    # Add to queue, stop generating
else:
    print(f"Frontier open: {skill_name} is independent")
    # Add to queue, continue generating
```

**Why This Works**:
- Uses full execution planning machinery (SkillDependencyResolver under the hood)
- Handles all edge cases: tiered items, achievements, ephemeral requirements
- Simple boolean check: feasible with completed? → independent : depends on training
- No need to track individual dependencies, just binary frontier state

**Changes Completed** ✅:
- [x] Modified `verify_skill_frontier_compliance()` in `flowrl/llm/craftax/after_queries.py:223-228` to merge completed + training skills
- [x] Added `check_frontier_blocked()` method in `flowrl/llm/flow.py:212-261` to detect frontier blocking

**Deliverable**: ✅ Two small modifications (~10 lines each) to existing frontier verification

---

### Phase 3: Tmux Orchestration & Checkpoint Merging (Week 2-3)
- [ ] Implement `SkillScheduler` class
- [ ] Test tmux session creation/monitoring
- [ ] Handle edge cases (crashes, OOM, etc.)
- [ ] **Modify `ppo_flow.py` to track frames per expert**:
  - Add frame counter for each expert head
  - Save `expert_frame_counts` dict in checkpoint
- [ ] **Implement frame-count checkpoint merging**:
  - Load global checkpoint + new skill checkpoint
  - Compare frame counts for each expert
  - Keep version with highest frame count
  - Test with 2 parallel skills that conflict

**Deliverable**: Working parallel training for 2-3 skills with correct checkpoint merging

---

### Phase 4: Integration & Testing (Week 3-4)
- [ ] Integrate into `bottomup.py`
- [ ] Add command-line flags (`--parallel`, `--wave_size`, etc.)
- [ ] Test full end-to-end pipeline
- [ ] Validate merged checkpoints work correctly

**Deliverable**: Full parallel training system

---

### Phase 5: Monitoring & Debugging (Week 4+)
- [ ] Add real-time progress dashboard (tmux status bar?)
- [ ] Implement recovery from failed skills
- [ ] Add logging/telemetry
- [ ] Performance profiling (speedup vs sequential)

**Deliverable**: Production-ready system with monitoring

---

## Checkpoint Conflict Resolution

### Problem: Parallel Training Creates Conflicting Expert Updates

**Current Sequential Behavior**:
When training a new skill, ALL previous MoE experts get updated (continual learning). This works fine sequentially:
```
Train skill_0 → experts: {0}
Train skill_1 → experts: {0', 1}  (skill_0 expert updated during skill_1 training)
Train skill_2 → experts: {0'', 1', 2}  (all previous updated)
```

**Parallel Conflict**:
If skills A, B, C train in parallel from base checkpoint `{0, 1, 2}`:
```
Skill_A training: {0, 1, 2} → {0', 1', 2', A}
Skill_B training: {0, 1, 2} → {0'', 1'', 2'', B}
Skill_C training: {0, 1, 2} → {0''', 1''', 2''', C}

Conflict: Which version of experts {0, 1, 2} to keep?
```

### Solution: Frame-Count Heuristic

**During checkpoint merging**:
1. Track total frames trained for each expert in each checkpoint
2. When merging, compare frame counts for conflicting experts
3. Keep the version trained on the most frames
4. This preserves continual learning while enabling clean parallelism

**Example**:
```python
# Skill A trained for 50M frames, updated expert_0 (now at 150M total)
# Skill B trained for 80M frames, updated expert_0 (now at 180M total)
# Skill C trained for 30M frames, updated expert_0 (now at 130M total)

# Merge: Keep expert_0 from Skill B (180M > 150M > 130M)
```

**Benefits**:
- ✅ Simple, deterministic merging rule
- ✅ Preserves continual learning (experts still improve)
- ✅ Bias towards more training = better performance
- ✅ Minimal implementation complexity

**Implementation**: Modified checkpoint format includes `expert_frame_counts` dict tracking frames per expert.

---

## Open Questions & Decisions

### 1. LLM Adherence to "Prefer Independent" Instruction
**Q**: How often will the LLM propose dependent skills despite instruction?

**Expected Behavior**:
- Most of the time: LLM proposes independent skills when possible
- Sometimes: LLM proposes dependent skill (e.g., "Make Pickaxe" when wood/stone are training)
  - This is actually fine! Programmatic check catches it → frontier blocked
  - Skill still gets queued, will launch when deps complete

**No Problem**: Even if LLM ignores instruction, programmatic check ensures correctness
- LLM instruction is an optimization (reduces wasted generations)
- Programmatic check is the safety net

**Recommendation**: Monitor how often frontier blocks occur, adjust prompt if excessive

---

### 2. Failure Handling
**Q**: What to do when a skill in a wave fails?

**Options**:
- **A**: Continue with successful skills, drop failed ones
  - Pros: Makes progress, doesn't block wave
  - Cons: May waste dependent skills in future waves

- **B**: Retry failed skill (up to N times)
  - Pros: Higher success rate
  - Cons: Delays entire wave

- **C**: Mark as failed, re-generate in next wave
  - Pros: Doesn't block, can try different approach
  - Cons: Complex bookkeeping

**Recommendation**: A for prototype, add B later with configurable retries

---

### 3. Resource Limits
**Q**: How many parallel skills can we train?

**Constraints**:
- **GPU Memory**: Each PPO run needs ~4-6GB (depends on num_envs)
- **CPU**: JAX compilation can saturate cores
- **Wall Time**: More parallel = faster total, but harder to debug

**Options**:
- **A**: Hard limit (max 4 parallel on single GPU)
- **B**: Auto-detect GPU memory, partition waves accordingly
- **C**: User-specified `--max_parallel_skills`

**Recommendation**: C (user control) with sensible defaults

---

### 4. Dependency Conflicts
**Q**: What if LLM generates skills that *look* independent but interfere?

**Example**:
- Skill A: "Collect 5 wood"
- Skill B: "Explore until finding water"
- Both modify movement policy → conflict in action distribution

**Detection**:
- Hard to detect statically (requires understanding RL dynamics)
- May only discover at training time (one skill succeeds, other fails)

**Mitigation**:
- Accept some failures as exploration cost
- Add "interference detection" in Phase 5 (compare policies, detect conflicts)
- LLM prompt: "Generate skills that use different action subsets"

**Recommendation**: Accept for now, revisit if failure rate is high

---

### 5. Checkpoint Merging Strategy
**Q**: How to merge skills that modify shared state (e.g., "achievements_completed")?

**Options**:
- **A**: Union (merge all achievements from all skills)
  - Pros: Preserves all progress
  - Cons: May overcount if skills overlap

- **B**: Intersection (only shared achievements)
  - Pros: Conservative, guarantees correctness
  - Cons: May lose valid progress

- **C**: Custom merge logic per field
  - Pros: Most accurate
  - Cons: Complex, error-prone

**Recommendation**: A (union) with validation checks

---

## Success Metrics

### Training Efficiency
- **Speedup**: Wall-clock time for N skills (parallel vs sequential)
  - Target: Near-linear speedup up to `max_parallel` (e.g., 2.7x for 3 parallel)
  - Measure: Total time to train 20 skills (continuous vs sequential)

- **GPU Utilization**: % of time with `max_parallel` skills running
  - Target: >85% (some downtime expected during skill transitions)
  - Measure: `(time_with_N_running / total_time) * 100`

- **Latency Improvement**: Time from skill generation to completion
  - Target: Fast skills complete 2-10x faster than wave-based
  - Measure: Compare completion times for simple skills (wood, stone)

### Skill Quality
- **Success Rate**: % of generated skills that train successfully
  - Target: >70% (same as current sequential)
  - Ensure parallelism doesn't degrade quality

### Dependency Analysis Accuracy
- **False Negatives**: Skills marked dependent but could run in parallel
  - Target: <10% (measure via manual inspection)
  - Impact: Lower throughput (unnecessary blocking)

- **False Positives**: Skills marked independent but actually conflict
  - Target: <5% (training failures due to interference)
  - Impact: Wasted GPU time, need to retry

### System Robustness
- **Crash Recovery**: Scheduler can resume after restart
  - Load `scheduler_state.json`, continue from last checkpoint

- **Fault Isolation**: Failed skill doesn't block unrelated skills
  - Other skills continue training, only dependent skills are blocked

- **Checkpoint Consistency**: Merged checkpoints are valid and complete
  - Can load `global_latest.pkl` and train next skill successfully

---

## Example Run

```bash
# Launch parallel bottom-up training with event-driven scheduling
python -m flowrl.bottomup \
  --env_name "Craftax-Symbolic-v1" \
  --graph_path "exp/parallel_test" \
  --max_nodes 20 \
  --max_parallel_skills 3 \
  --scheduler_poll_interval 30 \
  --total_timesteps 1e8 \
  --use_wandb

# Monitor progress (in another terminal)
watch -n 5 'cat exp/parallel_test/scheduler_state.json | jq ".skills | group_by(.status) | map({status: .[0].status, count: length})"'

# List active tmux sessions
tmux list-sessions | grep flowrl_

# Attach to specific skill
tmux attach -t flowrl_s1_collect_stone

# View scheduler state
cat exp/parallel_test/scheduler_state.json | jq
```

**Expected Output** (event-driven, programmatic frontier detection):
```
Starting event-driven skill generation (max: 20)
Starting scheduler (max_parallel=3)

Attempting to generate skill #1...
  Available slots: 3/3
  Currently training: []
  → Generated: Collect Wood (deps: [])
Added skill: 0_Collect_Wood (deps: [])
Launched: 0_Collect_Wood (session: flowrl_s0_Collect_Wood)

Attempting to generate skill #2...
  Available slots: 2/3
  Currently training: ['Collect Wood']
  → Generated: Collect Stone (deps: [])
Added skill: 1_Collect_Stone (deps: [])
Launched: 1_Collect_Stone (session: flowrl_s1_Collect_Stone)

Attempting to generate skill #3...
  Available slots: 1/3
  Currently training: ['Collect Wood', 'Collect Stone']
  → Generated: Eat Food (deps: [])
Added skill: 2_Eat_Food (deps: [])
Launched: 2_Eat_Food (session: flowrl_s2_Eat_Food)

Attempting to generate skill #4...
  Available slots: 0/3
  Currently training: ['Collect Wood', 'Collect Stone', 'Eat Food']
  → Generated: Make Pickaxe
  → Dependencies: ['Collect Wood', 'Collect Stone']
  → FRONTIER BLOCKED: Depends on training skills ['Collect Wood', 'Collect Stone']
  → Pausing generation until skills complete...
Added skill: 3_Make_Pickaxe (deps: ['Collect Wood', 'Collect Stone'])
[Status] Waiting: 1 | Running: 3/3 | Completed: 0 | Failed: 0

[30s later...]
[Status] Waiting: 1 | Running: 3/3 | Completed: 0 | Failed: 0

[45s later - first skill completes!]
============================================================
Skill 0_Collect_Wood completed!
  Success rate: 0.85
  Duration: 2025-10-04T10:30:00 -> 2025-10-04T11:15:00
============================================================

Merging checkpoint: 0_Collect_Wood -> global_latest.pkl
[Progress] 1/4 skills completed
  → Frontier unblocked - resuming generation

Attempting to generate skill #5...
  Available slots: 1/3
  Currently training: ['Collect Stone', 'Eat Food']
  → Generated: Drink Water (deps: [])
Added skill: 4_Drink_Water (deps: [])
Launched: 4_Drink_Water (session: flowrl_s4_Drink_Water)

Attempting to generate skill #6...
  Available slots: 0/3
  Currently training: ['Collect Stone', 'Eat Food', 'Drink Water']
  → Generated: Sleep (deps: [])
Added skill: 5_Sleep (deps: [])
[Status] Waiting: 2 | Running: 3/3 | Completed: 1 | Failed: 0

[60s later - second skill completes]
============================================================
Skill 1_Collect_Stone completed!
  Success rate: 0.82
  Duration: 2025-10-04T10:30:00 -> 2025-10-04T11:30:00
============================================================

Merging checkpoint: 1_Collect_Stone -> global_latest.pkl
Launched: 3_Make_Pickaxe (session: flowrl_s3_Make_Pickaxe)
  (dependencies satisfied: Collect Wood ✓, Collect Stone ✓)
[Progress] 2/6 skills completed
  → Frontier unblocked - resuming generation

Attempting to generate skill #7...
  Available slots: 0/3
  Currently training: ['Eat Food', 'Drink Water', 'Make Pickaxe']
  → Generated: Collect Coal (deps: [])
Added skill: 6_Collect_Coal (deps: [])
[Status] Waiting: 3 | Running: 3/3 | Completed: 2 | Failed: 0

... continues until all 20 skills generated and trained ...

All skills generated. Waiting for training to complete...
All skills completed!
Final checkpoint: exp/parallel_test/checkpoints/global_latest.pkl
```

**Key Differences from Wave-Based**:
- ✅ Skills launch **immediately** when dependencies are satisfied
- ✅ No waiting for slowest skill in a "wave"
- ✅ GPU utilization stays at max (always 3 skills running)
- ✅ New skills can be generated while training continues
- ✅ Early skills feed into later skills without blocking

---

## Next Steps

1. **Prototype Phase 1** (Multi-skill generation)
   - Modify `flowrl/llm/flow.py` to add `next_skills()` method
   - Test with simple prompt modifications

2. **Design Review**
   - Review this plan
   - Identify gaps/concerns
   - Adjust based on feedback

3. **Incremental Development**
   - Build one phase at a time
   - Validate with small experiments before scaling up

---

## Summary: Key Design Decisions

### Architecture
✅ **Event-Driven with Programmatic Frontier Detection** (not batch/wave-based)
- Single skill generated per LLM call (same as current)
- Separate `skills` (completed) from `training_skills` (in-progress)
- LLM sees both lists in prompts (prevents duplicates, allows dependencies)
- LLM **instructed** to prefer independent skills (soft constraint)
- **Programmatic check** after generation detects frontier blocking (hard constraint)
- Skills launch immediately when dependencies satisfied
- Maximum GPU utilization through continuous scheduling

### Components
1. **`Flow.next_skill()`** - Minimal changes to existing method
   - Adds skill to `training_skills` instead of `skills`
   - Shows both `skills_without_code` and `training_skills_without_code` in prompts
   - `on_skill_complete()` moves skill from training → completed
2. **`DependencyAnalyzer`** - Analyze single skill's dependencies
   - Determines which skills this skill requires (by checking gain/requirements)
3. **`SkillScheduler`** - Continuous job scheduler
   - Maintains: waiting queue, running set, completed set
   - Polls tmux sessions every N seconds
   - Launches skills up to `max_parallel` limit
4. **Frame-Count Checkpoint Merging** - Resolve expert conflicts
   - Track frames trained per expert
   - Keep most-trained version when merging
   - Preserves continual learning with parallelism

### Workflow (Event-Driven with Frontier Detection)
```
1. Check: Do we have available GPU slots AND frontier not blocked?
2. If yes:
   a. Call Flow.next_skill() (generates one skill)
   b. Check dependencies programmatically
   c. If depends on training_skills:
      - FRONTIER BLOCKED → add to queue → STOP generating
   d. If independent or depends only on completed:
      - Add to queue → continue generating
3. Scheduler continuously:
   - Monitors tmux sessions
   - Merges completed checkpoints
   - Launches newly-runnable skills
4. When skill completes:
   - Move from training_skills to skills
   - Unblock frontier
   - Main loop resumes generation attempts
5. Repeat until max_nodes reached
```

### Frontier Blocking Logic
**When to stop generating:**
- Generate skill
- Check: does it require anything from `training_skills`?
- YES → frontier blocked (all reachable skills need in-progress work)
- NO → frontier open (continue generating)

**When to resume:**
- Any skill completes → move to `skills` → frontier unblocked

### Key Parameters
- `--max_parallel_skills`: GPU slots (e.g., 3 for single GPU)
- `--scheduler_poll_interval`: Seconds between checks (e.g., 30)
- `--max_nodes`: Total skills to generate/train (e.g., 20)

### Benefits vs Sequential
- **2-3x speedup** for typical skill distributions
- **Higher GPU utilization** (~85-95% vs ~33% in sequential)
- **Lower latency** for fast skills (don't wait for slow ones)
- **Fault isolation** (failures don't block unrelated skills)
- **Simple & reliable** (programmatic check, no LLM decision needed)
