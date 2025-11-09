"""
Skill scheduler for parallel training with tmux orchestration.

This module handles:
1. Expert assignment (assign on training start)
2. Skill state tracking (waiting, running, completed, failed)
3. Dependency resolution
4. Tmux session management
5. Training progress monitoring
6. Triggering post-processing on completion
7. Failure handling and retry logic
"""

import subprocess
import sys
import time
import json
import datetime
from pathlib import Path
from typing import List, Dict, Optional, Callable
import os


class SkillScheduler:
    """
    Continuous scheduler for parallel skill training.

    Maintains:
    - Queue of pending skills (waiting for dependencies)
    - Set of running skills (currently training)
    - Set of completed/failed skills

    Behavior:
    - Launches skills as soon as dependencies are satisfied
    - Monitors tmux sessions for completion
    - Triggers post-processing on completion
    - Manages expert index assignment
    """

    def __init__(
        self,
        base_dir: Path,
        max_parallel: int = 3,
        poll_interval: int = 30,
        max_retries: int = 2,
        gpu_ids: Optional[List[str]] = None,
        conda_env: Optional[str] = None,
        skill_budget_timesteps: int = 10_000_000,
        frame_gen_envs: int = 32,
        frame_gen_max_frames: int = 1000,
        env_name: str = "Craftax-Symbolic-v1",
    ):
        """
        Initialize scheduler.

        Args:
            base_dir: Base experiment directory (e.g., exp/bottom_up/)
            max_parallel: Maximum number of parallel training sessions
            poll_interval: Seconds between status checks
            max_retries: Number of retries for failed skills
        """
        self.base_dir = Path(base_dir)
        # GPU assignment (one skill per GPU)
        if gpu_ids is None:
            env_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
            if env_cuda and env_cuda.strip():
                parsed = [g.strip() for g in env_cuda.split(",") if g.strip() != ""]
                self.gpu_ids = parsed
            else:
                # Fallback: assume a single GPU 0 if nothing specified
                self.gpu_ids = ["0"]
        else:
            # Normalize to list of strings
            self.gpu_ids = [str(g).strip() for g in gpu_ids if str(g).strip() != ""]

        # Cap parallelism to number of GPUs
        self.max_parallel = min(max_parallel, len(self.gpu_ids))
        self.poll_interval = poll_interval
        self.max_retries = max_retries
        self.conda_env = conda_env
        self._stop = False  # Run loop stop flag
        # Phase/budget configuration
        self.skill_budget_timesteps = int(skill_budget_timesteps)
        self.frame_gen_envs = int(frame_gen_envs)
        self.frame_gen_max_frames = int(frame_gen_max_frames)
        self.env_name = env_name

        # Expert index tracking
        self.next_expert_idx = 0
        self.expert_assignments = {}  # skill_id → expert_idx

        # Skill state tracking
        self.state_file = self.base_dir / "scheduler_state.json"
        self.state = self._load_or_create_state()

        # Callbacks
        self.on_skill_complete_callback = None
        self.prepare_training_run_callback = None
        self.process_completed_training_callback = None

    def _load_or_create_state(self) -> Dict:
        """Load scheduler state from file or create new."""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            # Restore next_expert_idx
            self.next_expert_idx = state.get("next_expert_idx", 0)
            self.expert_assignments = state.get("expert_assignments", {})
            return state
        else:
            return {
                "skills": {},
                "max_parallel": self.max_parallel,
                "currently_running": [],
                "next_expert_idx": 0,
                "expert_assignments": {},
                "gpu_ids": self.gpu_ids,
                "conda_env": self.conda_env
            }

    def _save_state(self):
        """Save scheduler state to file."""
        self.state["next_expert_idx"] = self.next_expert_idx
        self.state["expert_assignments"] = self.expert_assignments
        self.state["max_parallel"] = self.max_parallel
        self.state["gpu_ids"] = self.gpu_ids
        self.state["conda_env"] = self.conda_env

        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def assign_expert(self, skill_name: str) -> int:
        """
        Assign expert index to skill.

        Expert indices are assigned when skills start training (not at generation time).
        This ensures no gaps in expert numbering from failed skills.

        Args:
            skill_name: Name of the skill

        Returns:
            Assigned expert index
        """
        if skill_name in self.expert_assignments:
            return self.expert_assignments[skill_name]

        expert_idx = self.next_expert_idx
        self.expert_assignments[skill_name] = expert_idx
        self.next_expert_idx += 1
        self._save_state()

        return expert_idx

    def add_skill(
        self,
        skill_name: str,
        skill_data: Dict,
        dependency_skill_names: List[str] = None
    ) -> str:
        """
        Add a skill to the scheduler queue.

        Args:
            skill_name: Name of the skill
            skill_data: Skill data from Flow.next_skill()
            dependency_skill_names: List of dependency skill names

        Returns:
            skill_id for tracking
        """
        skill_id = skill_name.replace(' ', '_').replace('/', '_')

        self.state["skills"][skill_id] = {
            "skill_name": skill_name,
            "skill_data": skill_data,
            "status": "waiting",  # waiting, running, completed, failed
            "phase": "initial",   # initial -> analysis -> final
            "dependencies": dependency_skill_names or [],
            "expert_idx": None,  # Assigned when training starts
            "started_at": None,
            "completed_at": None,
            "tmux_session": None,
            "retry_count": 0,
            "run_folder": None,
            # Phase/budget tracking
            "budget_total": self.skill_budget_timesteps,
            "budget_remaining": self.skill_budget_timesteps,
            "prev_module_path": None,
        }

        self._save_state()
        print(f"  [Scheduler] Added skill '{skill_name}' (deps: {dependency_skill_names or 'none'})")

        return skill_id

    def set_callbacks(
        self,
        on_skill_complete: Optional[Callable] = None,
        prepare_training_run: Optional[Callable] = None,
        process_completed_training: Optional[Callable] = None,
        analyze_trajectories: Optional[Callable] = None,
    ):
        """
        Set callbacks for training workflow.

        Args:
            on_skill_complete: Called when skill completes: (skill_name) → None
            prepare_training_run: Called to setup training: (skill_name, skill_data, expert_idx, deps, completed_skills, base_dir) → (run_folder, module_path, policies_folder)
            process_completed_training: Called to process results: (run_folder, skill_name, expert_idx, base_dir) → results
        """
        self.on_skill_complete_callback = on_skill_complete
        self.prepare_training_run_callback = prepare_training_run
        self.process_completed_training_callback = process_completed_training
        self.analyze_trajectories_callback = analyze_trajectories

    def run(self, completed_skills: Dict[str, Dict]):
        """
        Main scheduler loop.

        Continuously:
        1. Check for completed skills
        2. Process their checkpoints
        3. Launch newly-runnable skills
        4. Stop when all skills are done

        Args:
            completed_skills: Dict of completed skills with expert_idx and paths
        """
        print(f"\n{'='*70}")
        print(f"Starting Skill Scheduler")
        print(f"  Max parallel: {self.max_parallel}")
        print(f"  Poll interval: {self.poll_interval}s")
        print(f"  Base directory: {self.base_dir}")
        print(f"{'='*70}\n")

        while not self._stop:
            # Update status of running skills
            self._update_running_skills(completed_skills)

            # Try to launch new skills
            launched = self._launch_runnable_skills(completed_skills)

            # If currently idle (no waiting or running), just keep polling until stop()
            if self._is_complete() and not launched:
                time.sleep(self.poll_interval)
                continue

            # Print status
            self._print_status()

            # Wait before next poll
            time.sleep(self.poll_interval)

        # Stopped explicitly: print final summary if no active work
        if self._is_complete():
            print("\n" + "="*70)
            print("All skills completed!")
            self._print_final_summary()
            print("="*70 + "\n")

    def stop(self):
        """Signal the scheduler loop to stop (after current poll)."""
        self._stop = True

    def _update_running_skills(self, completed_skills: Dict):
        """Check status of currently running skills."""
        for skill_id in list(self.state["currently_running"]):
            skill = self.state["skills"][skill_id]
            session = skill["tmux_session"]

            # Check if session is still alive
            if not self._is_session_alive(session):
                # Session finished - check if successful
                success = self._check_training_success(skill)

                if success:
                    phase = skill.get("phase", "initial")
                    if phase == "initial":
                        # Deduct budget using TRAINED_TIMESTEPS from config.yaml
                        policies_folder = Path(skill["run_folder"]) / f"{skill['expert_idx']}_policies"
                        import yaml
                        with open(policies_folder / "config.yaml", 'r') as f:
                            cfg = yaml.load(f, Loader=yaml.FullLoader)
                        def get_val(v):
                            return v.get('value') if isinstance(v, dict) and 'value' in v else v
                        trained = 0
                        for key in ['TRAINED_TIMESTEPS','trained_timesteps','TOTAL_TIMESTEPS','total_timesteps']:
                            if key in cfg and get_val(cfg.get(key)) is not None:
                                trained = int(get_val(cfg.get(key)))
                                break

                        prev_remaining = int(skill.get("budget_remaining", self.skill_budget_timesteps))
                        skill["budget_remaining"] = max(0, prev_remaining - trained)
                        skill["prev_module_path"] = str(Path(skill["run_folder"]) / f"{skill['expert_idx']}.py")

                        print(f"\n{'='*70}")
                        print(f"✓ Skill '{skill['skill_name']}' initial training complete")
                        print(f"  Trained timesteps: {trained:,}; remaining budget: {skill['budget_remaining']:,}")
                        print(f"  Transitioning to analysis phase...")
                        print(f"{'='*70}")

                        # Transition to analysis phase; will be launched in next cycle
                        skill["phase"] = "analysis"
                        skill["status"] = "waiting"
                        skill["tmux_session"] = None
                        skill["started_at"] = None
                        self._save_state()

                    elif phase == "analysis":
                        # Analysis completed
                        print(f"\n{'='*70}")
                        print(f"✓ Skill '{skill['skill_name']}' analysis complete")
                        print(f"  Transitioning to final training phase...")
                        print(f"{'='*70}")

                        # Load updated skill data from analysis output
                        analysis_output = Path(skill["run_folder"]) / "analysis_output.pkl"
                        if analysis_output.exists():
                            import pickle
                            with open(analysis_output, 'rb') as f:
                                updated_skill_data = pickle.load(f)
                            skill["skill_data"] = updated_skill_data
                            print(f"  ✓ Loaded updated skill spec from analysis")

                        # Recompute dependencies from updated skill_data
                        def _norm_req_key(k: str) -> str:
                            if k.startswith("achievement:"):
                                return k
                            if k.startswith("inventory:"):
                                return k.split(":", 1)[-1]
                            if k.startswith("inventory."):
                                return k.split(".", 1)[-1]
                            return k

                        providers = {}
                        for s in self.state["skills"].values():
                            gain = s.get("skill_data", {}).get("skill_with_consumption", {}).get("gain", {})
                            for gk in gain.keys():
                                key = gk if gk.startswith("achievement:") else _norm_req_key(gk)
                                providers.setdefault(key, s["skill_name"])

                        new_deps = []
                        reqs = skill["skill_data"].get("skill_with_consumption", {}).get("requirements", {})
                        for rk in reqs.keys():
                            key = rk if rk.startswith("achievement:") else _norm_req_key(rk)
                            prov = providers.get(key)
                            if prov and prov != skill["skill_name"]:
                                if prov not in new_deps:
                                    new_deps.append(prov)

                        merged = list(dict.fromkeys((skill.get("dependencies") or []) + new_deps))
                        skill["dependencies"] = merged

                        # Transition to final phase; will be launched in next cycle
                        skill["phase"] = "final"
                        skill["status"] = "waiting"
                        skill["tmux_session"] = None
                        skill["run_folder"] = None
                        skill["started_at"] = None
                        self._save_state()

                    else:
                        # Final phase completed
                        skill["status"] = "completed"
                        skill["completed_at"] = datetime.datetime.now().isoformat()

                        print(f"\n{'='*70}")
                        print(f"✓ Skill '{skill['skill_name']}' COMPLETED")
                        print(f"  Expert index: {skill['expert_idx']}")
                        print(f"  Duration: {skill['started_at']} → {skill['completed_at']}")
                        print(f"{'='*70}")

                        # Process completed training
                        if self.process_completed_training_callback:
                            results = self.process_completed_training_callback(
                                run_folder=Path(skill["run_folder"]),
                                skill_name=skill["skill_name"],
                                global_expert_idx=skill["expert_idx"],
                                base_dir=self.base_dir
                            )
                            skill["processing_results"] = results

                        # Mark this skill as completed for dependency checks
                        safe_name = skill["skill_name"].replace(' ', '_').replace('/', '_')
                        completed_skills[skill["skill_name"]] = {
                            "expert_idx": skill["expert_idx"],
                            "path": f"skills/{skill['expert_idx']}_{safe_name}/",
                        }

                        # Notify completion
                        if skill["status"] == "completed" and self.on_skill_complete_callback:
                            self.on_skill_complete_callback(skill["skill_name"])

                else:
                    # Training failed
                    skill["retry_count"] += 1

                    if skill["retry_count"] < self.max_retries:
                        print(f"\n✗ Skill '{skill['skill_name']}' FAILED (attempt {skill['retry_count']})")
                        print(f"  Will retry...")
                        skill["status"] = "waiting"  # Back to waiting for retry
                        skill["started_at"] = None
                        skill["completed_at"] = None
                        skill["tmux_session"] = None
                    else:
                        print(f"\n✗ Skill '{skill['skill_name']}' FAILED (max retries reached)")
                        skill["status"] = "failed"
                        skill["completed_at"] = datetime.datetime.now().isoformat()

                # Remove from running
                self.state["currently_running"].remove(skill_id)
                self._save_state()

    def _launch_runnable_skills(self, completed_skills: Dict) -> int:
        """
        Launch skills that have dependencies satisfied and fit in parallel limit.

        Args:
            completed_skills: Dict of completed skills

        Returns:
            Number of skills launched
        """
        launched = 0
        # Available slots limited by both max_parallel and free GPUs
        running_gpu_ids = set(
            self.state["skills"][sid].get("gpu_id")
            for sid in self.state["currently_running"]
            if self.state["skills"][sid].get("gpu_id") is not None
        )
        free_gpus = [g for g in self.gpu_ids if g not in running_gpu_ids]
        available_slots = min(self.max_parallel - len(self.state["currently_running"]), len(free_gpus))

        if available_slots <= 0:
            return 0

        # Find runnable skills
        for skill_id, skill in self.state["skills"].items():
            if skill["status"] != "waiting":
                continue

            # Check dependencies
            if not self._are_dependencies_satisfied(skill_id, completed_skills):
                continue

            # Launch it!
            # Choose the free GPU with the most available memory
            running_gpu_ids = set(
                self.state["skills"][sid].get("gpu_id")
                for sid in self.state["currently_running"]
                if self.state["skills"][sid].get("gpu_id") is not None
            )
            free_gpus = [g for g in self.gpu_ids if g not in running_gpu_ids]
            if not free_gpus:
                continue  # No free GPU at this moment

            gpu_id = self._pick_gpu_with_most_free_mem(free_gpus)
            self._launch_skill(skill_id, completed_skills, gpu_id)
            launched += 1

            # Check if we hit the limit
            # Stop if out of slots or GPUs
            running_gpu_ids = set(
                self.state["skills"][sid].get("gpu_id")
                for sid in self.state["currently_running"]
                if self.state["skills"][sid].get("gpu_id") is not None
            )
            free_gpus = [g for g in self.gpu_ids if g not in running_gpu_ids]
            if len(self.state["currently_running"]) >= self.max_parallel or not free_gpus:
                break

        return launched

    def _pick_gpu_with_most_free_mem(self, candidate_gpu_ids):
        """Return GPU id (as string) with the most free memory among candidates.

        Falls back to first candidate on error or if nvidia-smi is unavailable.
        """
        try:
            # Query all GPUs via nvidia-smi once
            import subprocess
            result = subprocess.run(
                [
                    "nvidia-smi", "--query-gpu=index,memory.free",
                    "--format=csv,noheader,nounits"
                ], capture_output=True, text=True, check=True
            )
            lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
            free_map = {}
            for line in lines:
                # Expect format: "<index>, <free>"
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 2:
                    idx, free = parts[0], parts[1]
                    # Only consider candidate GPUs
                    if idx in candidate_gpu_ids:
                        try:
                            free_map[idx] = int(free)
                        except ValueError:
                            continue
            if free_map:
                # Pick the candidate with max free MB
                return max(free_map.items(), key=lambda kv: kv[1])[0]
        except Exception:
            pass
        # Fallback
        return candidate_gpu_ids[0]

    def _are_dependencies_satisfied(self, skill_id: str, completed_skills: Dict) -> bool:
        """Check if all dependencies of a skill are completed."""
        skill = self.state["skills"][skill_id]

        for dep_name in skill["dependencies"]:
            # Check if dependency is in completed_skills
            if dep_name not in completed_skills:
                return False

        return True

    def _launch_skill(self, skill_id: str, completed_skills: Dict, gpu_id: str):
        """Launch training for a single skill."""
        skill = self.state["skills"][skill_id]

        print(f"\n{'─'*70}")
        print(f"Launching skill: {skill['skill_name']}")

        # Assign expert index
        expert_idx = self.assign_expert(skill["skill_name"])
        skill["expert_idx"] = expert_idx
        print(f"  Assigned expert index: {expert_idx}")

        # Determine phase early to avoid overwriting config before analysis
        phase = skill.get("phase", "initial")

        # Prepare training run unless we're in analysis phase
        if phase == "analysis":
            # Reuse existing artifacts from initial training
            run_folder = Path(skill.get("run_folder", ""))
            if not run_folder or not run_folder.exists():
                raise ValueError("Missing run_folder for analysis phase; initial training artifacts not found.")
            # Best-effort module_path reference for metadata (not used by analysis script)
            module_path = Path(run_folder) / f"{expert_idx}.py"
        else:
            if self.prepare_training_run_callback is None:
                raise ValueError("prepare_training_run_callback not set. Call set_callbacks() first.")

            run_folder, module_path, _ = self.prepare_training_run_callback(
                skill_name=skill["skill_name"],
                skill_data=skill["skill_data"],
                global_expert_idx=expert_idx,
                dependency_skill_names=skill["dependencies"],
                completed_skills=completed_skills,
                base_dir=self.base_dir
            )
            skill["run_folder"] = str(run_folder)

        # Ensure per-skill folder exists for logs and final artifacts
        safe_name = skill["skill_name"].replace(' ', '_').replace('/', '_')
        skill_folder = self.base_dir / "skills" / f"{expert_idx}_{safe_name}"
        skill_folder.mkdir(parents=True, exist_ok=True)
        skill["skill_folder"] = str(skill_folder)

        # Write launch metadata to skill folder for traceability
        metadata = {
            "skill_name": skill["skill_name"],
            "expert_idx": expert_idx,
            "gpu_id": gpu_id,
            "run_folder": str(run_folder),
            "module_path": str(module_path),
            "launched_at": datetime.datetime.now().isoformat(),
        }
        with open(skill_folder / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # Generate tmux session name
        session_name = f"flowrl_e{expert_idx}_{safe_name}"

        # Build training command
        log_path = skill_folder / "training.log"
        # Ensure log file exists ahead of time for visibility
        log_path.touch(exist_ok=True)
        # Build command based on phase
        if phase == "analysis":
            # Launch trajectory analysis
            cmd = self._build_analysis_command(
                skill_data=skill["skill_data"],
                run_folder=run_folder,
                expert_idx=expert_idx,
                gpu_id=gpu_id,
            )
        else:
            # Launch training (initial or final phase)
            exec_plan = skill["skill_data"].get("execution_plan", [])
            success_state = len(exec_plan)

            if phase == "initial":
                total_timesteps = int(skill.get("budget_remaining", self.skill_budget_timesteps))
                success_state_rate = 0.01
                prev_module_path = None
            else:  # final phase
                # Use explicit continuation allocation if present, else remaining budget
                total_timesteps = int(skill.pop("_continuation_timesteps", max(0, skill.get("budget_remaining", 0))))
                success_state_rate = 1.0
                prev_module_path = skill.get("prev_module_path")

            # Calculate initial_frames for continuation training
            initial_frames = 0
            if phase == "final":
                # Load global checkpoint to get current frame count for this expert
                # This is needed for continuation to avoid checkpoint naming conflicts
                try:
                    import sys
                    sys.path.insert(0, str(Path(__file__).parent))
                    from training_processor import load_global_checkpoint
                    global_ckpt = load_global_checkpoint(self.base_dir)
                    # Find frame count for this skill's expert
                    skill_name = skill["skill_name"]
                    if skill_name in global_ckpt.get("skills", {}):
                        initial_frames = int(global_ckpt["skills"][skill_name].get("total_frames", 0))
                        print(f"  Loading initial frames for continuation: {initial_frames:,}")
                except Exception as e:
                    print(f"  Warning: Could not load initial frames from global checkpoint: {e}")
                    initial_frames = 0

            cmd = self._build_training_command(
                module_path, run_folder, gpu_id, log_path, success_state,
                total_timesteps=total_timesteps,
                success_state_rate=success_state_rate,
                prev_module_path=prev_module_path,
                initial_frames=initial_frames,
            )

        # Launch tmux session from repo root for clean imports
        repo_root = Path(self.base_dir).parent.parent
        self._launch_tmux_session(session_name, cmd, repo_root, log_path)

        # Save exact command to training run folder for manual debugging
        cmd_txt = Path(run_folder) / "tmux_command.txt"
        cmd_sh = Path(run_folder) / "tmux_command.sh"
        timestamp = datetime.datetime.now().isoformat()
        with open(cmd_txt, 'w') as f:
            f.write(cmd + "\n")
        with open(cmd_sh, 'w') as f:
            f.write("#!/usr/bin/env bash\n")
            f.write(f"# Skill: {skill['skill_name']}\n")
            f.write(f"# Expert: {expert_idx}\n")
            f.write(f"# GPU: {gpu_id}\n")
            f.write(f"# Created: {timestamp}\n")
            f.write("set -e\n")
            f.write(cmd + "\n")
        os.chmod(cmd_sh, 0o755)

        # Update state
        skill["status"] = "running"
        skill["started_at"] = datetime.datetime.now().isoformat()
        skill["tmux_session"] = session_name
        self.state["currently_running"].append(skill_id)
        skill["gpu_id"] = gpu_id

        print(f"  Tmux session: {session_name}")
        print(f"  Command: {cmd}")
        print(f"  GPU: {gpu_id}")
        print(f"  Log: {log_path}")
        print(f"  Attach with: tmux attach -t {session_name}")
        print(f"{'─'*70}\n")

        self._save_state()

    def _build_analysis_command(self, skill_data: Dict, run_folder: Path, expert_idx: int, gpu_id: str) -> str:
        """Build trajectory analysis command.

        Args:
            skill_data: Skill data dictionary
            run_folder: Path to the run folder
            expert_idx: Expert index
            gpu_id: GPU ID to use

        Returns:
            Command string
        """
        import pickle
        import tempfile

        # Save skill data to temp file for analysis script
        skill_data_path = Path(run_folder) / "analysis_input.pkl"
        with open(skill_data_path, 'wb') as f:
            pickle.dump(skill_data, f)

        output_path = Path(run_folder) / "analysis_output.pkl"

        python_exe = sys.executable
        analysis_script = Path(__file__).parent / "run_trajectory_analysis.py"

        parts = [
            "env",
            f"CUDA_VISIBLE_DEVICES={gpu_id}",
            python_exe,
            str(analysis_script),
            "--skill_data_path", str(skill_data_path),
            "--output_path", str(output_path),
            "--run_folder", str(run_folder),
            "--expert_idx", str(expert_idx),
            "--env_name", self.env_name,
            "--base_dir", str(self.base_dir),
            "--frame_gen_envs", str(self.frame_gen_envs),
            "--frame_gen_max_frames", str(self.frame_gen_max_frames),
        ]

        return " ".join(parts)

    def _build_training_command(self, module_path: Path, run_folder: Path, gpu_id: str, log_path: Path, success_state: int, total_timesteps: Optional[int] = None, success_state_rate: float = 1.0, prev_module_path: Optional[str] = None, initial_frames: int = 0) -> str:
        """Build PPO training command without shell features.

        Uses the current Python interpreter and 'env' to set CUDA_VISIBLE_DEVICES.
        Logging is handled via tmux pipe-pane, not shell redirection.
        """
        python_exe = sys.executable
        parts = [
            "env",
            f"CUDA_VISIBLE_DEVICES={gpu_id}",
            python_exe,
            "-m", "flowrl.ppo_flow",
            "--module_path", str(module_path),
            "--env_name", self.env_name,
            "--success_state", str(success_state),
            "--save_policy",
        ]
        parts += ["--success_state_rate", str(success_state_rate)]
        if total_timesteps is not None and int(total_timesteps) > 0:
            parts += ["--total_timesteps", str(int(total_timesteps))]
        if prev_module_path:
            parts += ["--prev_module_path", str(prev_module_path)]
        else:
            policies_dir = Path(module_path).parent / f"{Path(module_path).stem}_policies" / "policies"
            if (policies_dir / "0").exists():
                parts += ["--prev_module_path", str(module_path)]
        # Add initial_frames for continuation training
        if initial_frames > 0:
            parts += ["--initial_frames", str(int(initial_frames))]
        return " ".join(parts)

    def _launch_tmux_session(self, session_name: str, cmd: str, working_dir: Path, log_path: Path):
        """Launch a tmux session with a clean command and attach logging via pipe-pane."""
        # Kill existing session if present
        subprocess.run(
            ["tmux", "kill-session", "-t", session_name],
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL
        )

        # Create new session (detached)
        subprocess.run([
            "tmux", "new-session", "-d", "-s", session_name,
            "-c", str(working_dir),
            cmd
        ], check=True)

        # Pipe pane output to the log file (append once)
        subprocess.run([
            "tmux", "pipe-pane", "-o", "-t", session_name,
            f"cat >> {log_path} 2>&1"
        ], check=True)

    def _is_session_alive(self, session_name: str) -> bool:
        """Check if tmux session is still running."""
        if session_name is None:
            return False

        result = subprocess.run(
            ["tmux", "has-session", "-t", session_name],
            capture_output=True
        )
        return result.returncode == 0

    def _check_training_success(self, skill: Dict) -> bool:
        """
        Check if training/analysis succeeded by examining outputs.

        Success criteria:
        - For training phases (initial/final):
          - Orbax checkpoints: <run>/<E>_policies/policies/<STEP> where STEP is a positive int (>0)
          - Legacy pickle checkpoints: checkpoint_final.pkl / checkpoint.pkl / checkpoint_*.pkl
        - For analysis phase:
          - Output file exists: <run>/analysis_output.pkl
        """
        run_folder = Path(skill["run_folder"])
        phase = skill.get("phase", "initial")

        # Check analysis phase
        if phase == "analysis":
            analysis_output = run_folder / "analysis_output.pkl"
            return analysis_output.exists()

        # Check training phase (initial or final)
        expert_idx = skill["expert_idx"]
        policies_folder = run_folder / f"{expert_idx}_policies"

        # 1) Orbax format: policies/<STEP>
        orbax_dir = policies_folder / "policies"
        if orbax_dir.exists():
            try:
                entries = [p.name for p in orbax_dir.iterdir() if p.is_dir() or p.is_file()]
                # Consider success only if there's a positive step (>0). Step 0 may be a seed.
                steps = []
                for name in entries:
                    try:
                        steps.append(int(name))
                    except ValueError:
                        continue
                if any(step > 0 for step in steps):
                    return True
            except Exception:
                # Fall through to legacy checks
                pass

        # 2) Legacy pickle checkpoints
        for checkpoint_name in ["checkpoint_final.pkl", "checkpoint.pkl"]:
            if (policies_folder / checkpoint_name).exists():
                return True

        checkpoints = list(policies_folder.glob("checkpoint_*.pkl"))
        return len(checkpoints) > 0

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

        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] " +
              f"Waiting: {waiting} | Running: {running}/{self.max_parallel} | " +
              f"Completed: {completed} | Failed: {failed}")

    def _print_final_summary(self):
        """Print final summary of all skills."""
        completed = [s for s in self.state["skills"].values() if s["status"] == "completed"]
        failed = [s for s in self.state["skills"].values() if s["status"] == "failed"]

        print(f"\nCompleted skills: {len(completed)}")
        for skill in completed:
            duration = "?"
            if skill.get("started_at") and skill.get("completed_at"):
                try:
                    start = datetime.datetime.fromisoformat(skill["started_at"])
                    end = datetime.datetime.fromisoformat(skill["completed_at"])
                    duration = str(end - start).split('.')[0]  # Remove microseconds
                except:
                    pass
            print(f"  ✓ {skill['skill_name']} (expert {skill['expert_idx']}) - {duration}")

        if failed:
            print(f"\nFailed skills: {len(failed)}")
            for skill in failed:
                print(f"  ✗ {skill['skill_name']} (retries: {skill['retry_count']})")

    def get_completed_skill_names(self) -> List[str]:
        """Get list of completed skill names."""
        return [
            skill["skill_name"]
            for skill in self.state["skills"].values()
            if skill["status"] == "completed"
        ]

    def get_running_skill_names(self) -> List[str]:
        """Get list of currently running skill names."""
        return [
            self.state["skills"][skill_id]["skill_name"]
            for skill_id in self.state["currently_running"]
        ]

    def get_skill_info(self, skill_name: str) -> Optional[Dict]:
        """Get information about a specific skill."""
        skill_id = skill_name.replace(' ', '_').replace('/', '_')
        return self.state["skills"].get(skill_id)

    def enqueue_continuation(self, skill_name: str, extra_timesteps: int):
        """Queue a continuation run for an existing skill's final phase.

        Treats continuation like a fresh training run: rebuilds the full MoE checkpoint
        from current experts (avoiding stale checkpoints) and launches as final phase.

        Args:
            skill_name: Human-readable skill name
            extra_timesteps: Additional timesteps to train
        """
        skill_id = skill_name.replace(' ', '_').replace('/', '_')
        if skill_id not in self.state["skills"]:
            raise ValueError(f"Unknown skill '{skill_name}' for continuation")
        skill = self.state["skills"][skill_id]
        if skill_id in self.state["currently_running"] or self._is_session_alive(skill.get("tmux_session")):
            raise RuntimeError(f"Skill '{skill_name}' is currently running; cannot enqueue continuation")

        # Verify skill folder and expert exist
        skill_folder = Path(skill.get("skill_folder", ""))
        if not skill_folder.exists():
            raise ValueError(f"Skill folder not found for '{skill_name}': {skill_folder}")
        expert_idx = skill.get("expert_idx")
        if expert_idx is None:
            raise ValueError(f"Skill '{skill_name}' has no assigned expert index")

        # Respect remaining budget if tracked
        remaining = int(skill.get("budget_remaining", extra_timesteps))
        alloc = max(0, min(int(extra_timesteps), remaining))
        if alloc <= 0:
            raise ValueError(f"No budget remaining for '{skill_name}' continuation")

        # Reset skill state to relaunch as fresh final phase
        # prepare_training_run_callback will rebuild checkpoint from current experts
        skill["phase"] = "final"
        skill["status"] = "waiting"
        skill["prev_module_path"] = None  # Let training_setup rebuild from current experts
        skill["run_folder"] = None  # Will be created fresh
        skill["tmux_session"] = None
        skill["started_at"] = None
        skill["completed_at"] = None
        skill["retry_count"] = 0
        # Stash the desired allocation so _build_training_command uses it
        skill["_continuation_timesteps"] = alloc

        print(f"  [Scheduler] Enqueued continuation for '{skill_name}':")
        print(f"    Expert index: {expert_idx}")
        print(f"    Continuation timesteps: {alloc:,}")
        print(f"    Will rebuild checkpoint from current experts in skills/")

        self._save_state()
