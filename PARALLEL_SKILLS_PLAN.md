# Parallel Skill Learning: Implementation Plan

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

**Current**: `next_skill() -> (skill_name, skill_data)`
**New**: `next_skill_or_wait(currently_training) -> (skill_name, skill_data) | "WAIT"`

**Key Idea**: Augment the existing prompt with context about what's training, let LLM decide:
- Propose a new skill (independent or depends on completed skills only)
- Return "WAIT" if all feasible next skills depend on currently-training skills

```python
class Flow:
    def next_skill_or_wait(self, currently_training: List[str] = None):
        """
        Generate next skill, or return 'WAIT' if blocked by in-progress skills.

        Args:
            currently_training: List of skill names currently being trained

        Returns:
            (skill_name, skill_data) if a new skill is possible
            "WAIT" if we should wait for training skills to complete
        """
        currently_training = currently_training or []

        # Add context to database for prompt
        self.db["currently_training_skills"] = currently_training
        self.db["completed_skills"] = list(self.skills.keys())

        # Generate skill (LLM sees both lists in prompt)
        # Prompt will be modified to include instruction:
        # "You can propose a skill that depends ONLY on completed skills.
        #  If all valuable next skills require currently-training skills,
        #  respond with: WAIT: <skill_name> (reason: need X to finish)"

        error = "Not generated yet"
        while error != "":
            try:
                evaluated = self.graph.evaluate()
                generated_code = evaluated[list(evaluated.keys())[-1]]

                # Check for WAIT response
                if generated_code.strip().startswith("WAIT:"):
                    reason = generated_code.strip()
                    print(f"LLM decided to wait: {reason}")
                    return "WAIT", {"reason": reason}

                # Validate as normal
                functions, error = self.validate_code(generated_code)
                if error != "":
                    print(f"Error generating: {error}")
            except RestartGraphException as e:
                print(f"Graph restart requested: {e}")
                continue

        # Extract skill info
        skill_name = self.db["current"]["skill_name"]
        skill_with_consumption = self.db["current"]["skill_with_consumption"]

        skill_data = {
            "skill_name": skill_name,
            "skill_with_consumption": skill_with_consumption,
            "functions": functions,
            "iteration": len(self.skills)  # Use skill count instead of current_i
        }

        return skill_name, skill_data
```

**Prompt Modification** (in `flowrl/llm/craftax_classic/prompts/main.py`):

```python
# Add to next_task prompt
"""
## Context

**Completed Skills:**
{db[completed_skills]}

**Currently Training Skills (DO NOT depend on these):**
{db[currently_training_skills]}

## Instructions

You may propose a skill that:
1. Is independent (no requirements), OR
2. Depends ONLY on completed skills (not currently-training skills)

If the most valuable next skills ALL require currently-training skills to complete,
respond with:

WAIT: <skill_name_we_need_to_wait_for>
Reason: <why we need to wait>

Otherwise, generate a skill as normal.
"""
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

    def merge_checkpoints(self) -> dict:
        """
        Merge checkpoints from all successful skills.

        Returns:
            Merged checkpoint dict
        """
        merged = {
            "skills": {},
            "db": {},
            "wave_id": self.wave_id
        }

        for skill in self.metadata["skills"]:
            if skill["status"] == "completed":
                ckpt_path = Path(skill["skill_dir"]) / "checkpoint.pkl"

                if ckpt_path.exists():
                    import pickle
                    with open(ckpt_path, 'rb') as f:
                        skill_ckpt = pickle.load(f)

                    # Merge skills
                    merged["skills"].update(skill_ckpt.get("skills", {}))

                    # Merge db (careful with conflicts)
                    for key, value in skill_ckpt.get("db", {}).items():
                        if key not in ["prompts", "temp_prompts"]:
                            merged["db"][key] = value

        # Save merged checkpoint
        merged_path = self.wave_dir.parent / "checkpoints" / f"wave_{self.wave_id}_merged.pkl"
        merged_path.parent.mkdir(parents=True, exist_ok=True)

        import pickle
        with open(merged_path, 'wb') as f:
            pickle.dump(merged, f)

        return merged

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

**Key Changes**: Reactive generation triggered by skill completions

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

print(f"Starting event-driven skill generation (max: {args.max_nodes})")

while total_skills_generated < args.max_nodes or not scheduler._is_complete():
    # Check if we have available GPU slots
    available_slots = scheduler.max_parallel - len(scheduler.state["currently_running"])

    # Only try to generate if we have slots OR if nothing is running (initial state)
    if available_slots > 0 or len(scheduler.state["currently_running"]) == 0:
        # Get currently training skill names
        currently_training = [
            scheduler.state["skills"][sid]["skill_name"]
            for sid in scheduler.state["currently_running"]
        ]

        # Try to generate next skill
        print(f"\nAttempting to generate skill #{total_skills_generated + 1}...")
        print(f"  Available slots: {available_slots}/{scheduler.max_parallel}")
        print(f"  Currently training: {currently_training}")

        result = flow_graph.next_skill_or_wait(currently_training)

        if result[0] == "WAIT":
            # LLM says we need to wait for in-progress skills
            reason = result[1]["reason"]
            print(f"  → LLM says WAIT: {reason}")
            print(f"  → Pausing generation until a skill completes...")
        else:
            # Got a new skill!
            skill_name, skill_data = result

            # Add to flow graph
            flow_graph.add_skill(skill_name, skill_data)

            # Determine dependencies
            dependencies = analyzer.get_dependencies(skill_name, flow_graph.skills)

            # Add to scheduler queue
            scheduler.add_skill(skill_name, skill_data, dependencies)

            total_skills_generated += 1
            print(f"  → Generated: {skill_name} (deps: {dependencies})")

            # Try to generate another immediately if we still have slots
            continue

    # Wait a bit before checking again
    time.sleep(args.scheduler_poll_interval)

    # Print status periodically
    completed_count = sum(1 for s in scheduler.state["skills"].values()
                         if s["status"] == "completed")
    if completed_count > last_completed_count:
        last_completed_count = completed_count
        print(f"\n[Progress] {completed_count}/{total_skills_generated} skills completed")

# Wait for all skills to finish
print("\nAll skills generated. Waiting for training to complete...")
scheduler_thread.join()

print("All skills completed!")
print(f"Final checkpoint: {args.graph_path}/checkpoints/global_latest.pkl")
```

**How It Works**:

1. **Initial state**: Generate skills until all GPU slots are full
2. **When a skill completes**:
   - Scheduler merges checkpoint
   - Launches any waiting skills with satisfied dependencies
   - Main loop detects available slot → tries to generate another
3. **LLM decision**:
   - If new independent skill is possible → generate it
   - If all good skills need in-progress skills → return "WAIT"
4. **Wait state**: Main loop idles, only scheduler is active
5. **Resume**: When next skill completes, available_slots > 0, try generation again

---

## Implementation Phases

### Phase 1: Reactive Skill Generation (Week 1)
- [ ] Implement `Flow.next_skill_or_wait(currently_training)` method
- [ ] Modify prompts to include:
  - List of completed skills
  - List of currently training skills
  - WAIT instruction and format
- [ ] Test LLM decision-making:
  - Empty state → proposes skill
  - All slots full, no dependencies possible → returns WAIT
  - Some slots full, independent skill exists → proposes skill
- [ ] Validate WAIT response parsing

**Deliverable**: Modified `Flow` class that can generate skills reactively or signal to wait

---

### Phase 2: Dependency Analysis (Week 1-2)
- [ ] Implement `DependencyAnalyzer` class
- [ ] Write unit tests for `get_dependencies()`
- [ ] Test with real skill examples from Craftax:
  - Basic skills (no deps): Collect Wood, Collect Stone
  - Dependent skills: Make Pickaxe (needs wood + stone)
  - Transitive deps: Mine Iron (needs pickaxe → wood + stone)
- [ ] Visualize dependency DAG for debugging

**Deliverable**: Analyzer that correctly identifies skill dependencies

---

### Phase 3: Tmux Orchestration (Week 2-3)
- [ ] Implement `ParallelCoordinator` class
- [ ] Test tmux session creation/monitoring
- [ ] Handle edge cases (crashes, OOM, etc.)
- [ ] Implement checkpoint merging logic

**Deliverable**: Working parallel training for 2-3 skills

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

## Open Questions & Decisions

### 1. LLM WAIT Decision Accuracy
**Q**: How often will the LLM correctly identify "WAIT" situations?

**Potential Issues**:
- **False WAIT**: LLM says WAIT when an independent skill is actually possible
  - Impact: Underutilized GPU, slower throughput
  - Mitigation: Include examples in prompt showing independent skills

- **Missed WAIT**: LLM proposes skill that depends on in-progress skill
  - Impact: Skill sits in queue waiting, no harm (scheduler handles it)
  - Better than false WAIT

**Recommendation**: Bias towards proposing skills (conservative WAIT), monitor false WAIT rate

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

**Expected Output** (event-driven, reactive):
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
  → LLM says WAIT: Need 'Collect Wood' or 'Collect Stone' to finish for Make Pickaxe
  → Pausing generation until a skill completes...

[Status] Waiting: 0 | Running: 3/3 | Completed: 0 | Failed: 0

[30s later...]
[Status] Waiting: 0 | Running: 3/3 | Completed: 0 | Failed: 0

[45s later - first skill completes!]
============================================================
Skill 0_Collect_Wood completed!
  Success rate: 0.85
  Duration: 2025-10-04T10:30:00 -> 2025-10-04T11:15:00
============================================================

Merging checkpoint: 0_Collect_Wood -> global_latest.pkl
[Status] Waiting: 0 | Running: 2/3 | Completed: 1 | Failed: 0

Attempting to generate skill #4...
  Available slots: 1/3
  Currently training: ['Collect Stone', 'Eat Food']
  → Generated: Drink Water (deps: [])
Added skill: 3_Drink_Water (deps: [])
Launched: 3_Drink_Water (session: flowrl_s3_Drink_Water)

Attempting to generate skill #5...
  Available slots: 0/3
  Currently training: ['Collect Stone', 'Eat Food', 'Drink Water']
  → LLM says WAIT: Need 'Collect Stone' to finish for Make Pickaxe
  → Pausing generation until a skill completes...

[60s later - second skill completes]
============================================================
Skill 1_Collect_Stone completed!
  Success rate: 0.82
  Duration: 2025-10-04T10:30:00 -> 2025-10-04T11:30:00
============================================================

Merging checkpoint: 1_Collect_Stone -> global_latest.pkl
[Status] Waiting: 0 | Running: 2/3 | Completed: 2 | Failed: 0

Attempting to generate skill #5...
  Available slots: 1/3
  Currently training: ['Eat Food', 'Drink Water']
  → Generated: Make Pickaxe (deps: ['0_Collect_Wood', '1_Collect_Stone'])
Added skill: 4_Make_Pickaxe (deps: ['0_Collect_Wood', '1_Collect_Stone'])
Launched: 4_Make_Pickaxe (session: flowrl_s4_Make_Pickaxe)

[Progress] 2/5 skills completed

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
✅ **Event-Driven Reactive Generation** (not batch/wave-based)
- Single skill generated per LLM call
- LLM sees completed + in-progress skills, decides:
  - Propose new skill (independent or depends on completed only)
  - Return "WAIT" if all good skills need in-progress skills
- Skills launch immediately when dependencies satisfied
- Maximum GPU utilization through continuous scheduling

### Components
1. **`Flow.next_skill_or_wait(currently_training)`** - Reactive single-skill generation
   - Returns new skill OR "WAIT" signal
   - LLM makes intelligent decision based on frontier state
2. **`DependencyAnalyzer`** - Analyze single skill's dependencies
   - Determines which completed skills this skill requires
3. **`SkillScheduler`** - Continuous job scheduler
   - Maintains: waiting queue, running set, completed set
   - Polls tmux sessions every N seconds
   - Launches skills up to `max_parallel` limit
4. **Incremental Checkpointing** - Merge as each skill completes

### Workflow (Event-Driven)
```
1. Check: Do we have available GPU slots?
2. If yes:
   a. Call Flow.next_skill_or_wait(currently_training)
   b. LLM decides:
      - New skill possible → generate it, add to queue
      - All skills blocked → return "WAIT"
   c. If new skill: immediately try to generate another
   d. If WAIT: pause generation, let scheduler work
3. Scheduler continuously:
   - Monitors tmux sessions
   - Merges completed checkpoints
   - Launches newly-runnable skills
4. When skill completes:
   - Free up GPU slot
   - Main loop resumes generation attempts
5. Repeat until max_nodes reached
```

### Key Parameters
- `--max_parallel_skills`: GPU slots (e.g., 3 for single GPU)
- `--scheduler_poll_interval`: Seconds between checks (e.g., 30)
- `--max_nodes`: Total skills to generate/train (e.g., 20)

### Benefits vs Sequential
- **2-3x speedup** for typical skill distributions
- **Higher GPU utilization** (~85-95% vs ~33% in sequential)
- **Lower latency** for fast skills (don't wait for slow ones)
- **Fault isolation** (failures don't block unrelated skills)
- **Intelligent blocking** (LLM decides when to wait vs propose)
