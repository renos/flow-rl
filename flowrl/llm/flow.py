import pickle
from flowrl.llm.craftax_classic.llm import generate_graph
from flowrl.llm.craftax_classic.generate_code import (
    validate_code,
    generate_validated_py,
)

# from craftax.craftax_classic.constants import BlockType
import shutil
from pathlib import Path
import json
import copy

# from flowrl.utils.test import explain_trajectory, render_video
# from flowrl.utils.test import gen_frames_hierarchical
import os
import yaml

from flowrl.utils.trajectory_explanation import (
    explain_trajectory,
)
from flowrl.skill_dependency_resolver import SkillDependencyResolver


class Flow:
    def __init__(self, args):
        self.args = args
        self.graph_path = Path(args.graph_path)
        if args.env_name == "Craftax-Classic-Symbolic-v1":
            self.db, self.graph, self.inventory_graph, self.reuse_graph = (
                generate_graph(return_inventory_graph=True)
            )
        else:
            assert 0, f"Not Implemented"

        self.current_i = args.current_i  # Current iteration (not node count)
        self.previous_i = args.previous_i

        self.skills = {}  # Dictionary of skill_name -> skill_data
        self.old_skills = None
        
        # Checkpoint system
        self._checkpoints = {}  # Dictionary of checkpoint_name -> state_snapshot

        # Load previous checkpoint if specified
        if self.previous_i is not None and self.previous_i >= 0:
            # Look for checkpoint files in the new format
            checkpoint_dir = self.graph_path / "checkpoints"
            if checkpoint_dir.exists():
                # Find the successful checkpoint for the previous iteration
                checkpoint_pattern = f"successful_skill_{self.previous_i}_*.pkl"
                checkpoint_files = list(checkpoint_dir.glob(checkpoint_pattern))
                
                if checkpoint_files:
                    # Load the most recent checkpoint if multiple exist
                    checkpoint_file = sorted(checkpoint_files)[-1]
                    print(f"Loading checkpoint from: {checkpoint_file}")
                    
                    try:
                        with open(checkpoint_file, 'rb') as f:
                            checkpoint = pickle.load(f)
                        
                        # Restore state from checkpoint
                        # Use command line args for current_i and previous_i, checkpoint for data
                        self.skills = checkpoint["skills"]
                        # Keep the command line specified current_i and previous_i
                        # Don't override with checkpoint values
                        
                        # Merge checkpoint db with current db (preserve prompts)
                        for key, value in checkpoint["db"].items():
                            if (not key == "prompts") and (not key == "temp_prompts"):
                                self.db[key] = value
                        
                        # Sync skills to database for prompt access
                        self.db["skills"] = self.skills
                        
                        print(f"Successfully loaded checkpoint from iteration {self.previous_i}")
                        
                    except Exception as e:
                        print(f"Error loading checkpoint {checkpoint_file}: {e}")
                        print("Falling back to legacy format...")
                        self._load_legacy_format()
                else:
                    print(f"No checkpoint found for iteration {self.previous_i}, trying legacy format...")
                    self._load_legacy_format()
            else:
                print("No checkpoints directory found, trying legacy format...")
                self._load_legacy_format()

    def _load_legacy_format(self):
        """Fallback method to load from old db.pkl and skills.json format"""
        if self.previous_i is None:
            return
            
        try:
            with open(self.graph_path / str(self.previous_i) / "db.pkl", "rb") as f:
                db_old = pickle.load(f)
            for key, value in db_old.items():
                self.db[key] = value
        except FileNotFoundError:
            print(f"Warning: Could not find db.pkl for previous_i={self.previous_i}")
        
        try:
            with open(
                self.graph_path / str(self.previous_i) / "skills.json",
                "r",
                encoding="utf-8",
            ) as f:
                skills_data = json.load(f)
            self.skills = skills_data["data"]
            # Sync loaded skills to database for prompt access
            self.db["skills"] = self.skills
        except FileNotFoundError:
            print(f"Warning: Could not find skills.json for previous_i={self.previous_i}")

    def add_skill(self, skill_name, skill_data):
        """Add a skill to the skills dictionary"""
        self.skills[skill_name] = skill_data

    def next_skill(self):
        """Generate and validate a complete skill from the graph"""
        from flowrl.llm.craftax_classic.after_queries import RestartGraphException
        self.db['skills_without_code'] = {key: value["skill_with_consumption"] for key, value in self.db['skills'].items()}
        error = "Not generated yet"
        while error != "":
            try:
                evaluated = self.graph.evaluate()
                generated_code = evaluated[list(evaluated.keys())[-1]]
                functions, error = validate_code(generated_code)
                if error != "":
                    print(f"Error generating: {error}")
            except RestartGraphException as e:
                print(f"Graph restart requested: {e}")
                print("Restarting graph evaluation from next_task...")
                # Continue the loop to re-evaluate the graph
                continue
        
        # Save the graph evaluation
        txt_path = os.path.join(self.graph_path / str(self.current_i), "graph_eval.txt")
        (self.graph_path / str(self.current_i)).mkdir(parents=True, exist_ok=True)
        with open(txt_path, "w") as f:
            for key, value in evaluated.items():
                f.write(f"{key}\n")
                f.write(f"{value}\n")
                f.write("-" * 40 + "\n")  # 40 dashes as separator
        
        # Extract skill information from the database
        skill_name = self.db["current"]["skill_name"]
        skill_with_consumption = self.db["current"]["skill_with_consumption"]
        
        skill_data = {
            "skill_name": skill_name,
            "skill_with_consumption": skill_with_consumption,
            "functions": functions,
            "iteration": self.current_i
        }
        
        return skill_name, skill_data

    def build_skill_dependency_graph(self, target_skill_name):
        """
        Build a dependency graph for the target skill by transforming requirements
        at each level until all dependencies are satisfied.
        
        Args:
            target_skill_name: Name of the skill to build dependencies for
        """
        resolver = SkillDependencyResolver(self.skills)
        self.execution_order = resolver.resolve_dependencies(target_skill_name)
        print(f"Execution order for skill '{target_skill_name}': {self.execution_order}")

    def write_code(self):
        """
        Write all skills to a single module file for the current iteration.
        Creates one node for each step in the execution order.
        """
        save_dir = self.graph_path / str(self.current_i)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Execution order must be available
        assert hasattr(self, 'execution_order') and self.execution_order, "No execution order available. Call build_skill_dependency_graph() first."
        
        print(f"Writing {len(self.execution_order)} execution steps to module")
        
        # Generate the module file with one node per execution step
        module_path = os.path.join(save_dir, f"{self.current_i}.py")
        
        for task_index, (skill_name, count) in enumerate(self.execution_order):
            if skill_name in self.skills:
                skill_data = self.skills[skill_name]
                functions = skill_data.get("functions", [])
                
                if len(functions) >= 3:
                    # generate_validated_py expects exactly 3 functions: [task_is_done, task_reward, task_network_number]
                    generate_validated_py(functions[:3], module_path, task_index, n=count)
                    print(f"Added task_{task_index}: {skill_name} (n={count})")
                else:
                    print(f"Warning: Skill '{skill_name}' has {len(functions)} functions, expected 3")
            else:
                print(f"Warning: Skill '{skill_name}' not found in skills database")
        
        print(f"Successfully wrote {len(self.execution_order)} tasks to {module_path}")

    def update_db(self, out, skill_name, skill_data):
        """Update database with training results and skill information"""
        # Track achievements for prompts
        achievements = {
            key: value[-1]
            for key, value in out["info"].items()
            if "Achievements" in key
        }
        self.db["current"]["achievements_completed"] = {
            key: f"{value:.1f}%" for key, value in achievements.items()
        }
        
        # Update skills count only if this is a new skill (not reused)
        skill_with_consumption = self.db["current"].get("skill_with_consumption", {})
        is_reused = skill_with_consumption.get("reuse_skill", False)
        if not is_reused:
            self.db["current"]["num_skills"] += 1
        
        # Add skill to skills database with complete skill data
        self.db["skills"][skill_name] = skill_data

    def explain_trajectory(self, env_states, actions, goal_state):
        """Explain the trajectory and update skill based on actual execution"""
        try:
            self.db["example_trajectory"] = explain_trajectory(
                env_states, actions, start_state=goal_state
            )
        except:
            breakpoint()

        # Get current skill name
        skill_name = self.db["current"]["skill_with_consumption"]["skill_name"]
        print(f"Updating skill: {skill_name}")
        
        # Update database with current skills before analysis
        self.db["skills"] = self.skills
        
        # Run inventory analysis to update skill and propose KB updates
        results = self.inventory_graph.evaluate()
        
        # Update the skill in self.skills based on the analysis
        if "skill_update_results" in self.db["current"]:
            skill_updates = self.db["current"]["skill_update_results"]
            if skill_name in self.skills:
                # Update the skill_with_consumption in the stored skill
                self.skills[skill_name]["skill_with_consumption"]["requirements"] = skill_updates.get("updated_requirements", {})
                self.skills[skill_name]["skill_with_consumption"]["consumption"] = skill_updates.get("updated_consumption", {})
                self.skills[skill_name]["skill_with_consumption"]["gain"] = skill_updates.get("updated_gain", {})
                print(f"Updated skill {skill_name} based on trajectory")
                print(f"New requirements: {skill_updates.get('updated_requirements', {})}")
                print(f"New consumption: {skill_updates.get('updated_consumption', {})}")
                print(f"New gain: {skill_updates.get('updated_gain', {})}")
                
        # Print knowledge base updates applied
        if "kb_updates_applied" in self.db["current"]:
            kb_updates = self.db["current"]["kb_updates_applied"]
            print("\nKnowledge base updates applied:")
            for i, update in enumerate(kb_updates):
                print(f"  Update {i+1}: {' -> '.join(update.get('path', []))}")
                print(f"    Old: {update.get('old_requirements', [])}")
                print(f"    New: {update.get('new_requirements', [])}")
                print(f"    Reason: {update.get('reason', '')}")
        
        #Update database skills
        self.db["skills"] = self.skills
        
        return results

    def create_checkpoint(self, checkpoint_name):
        """Create a checkpoint of the current state (excluding prompts)"""
        # Create a copy of db without prompts
        db_without_prompts = copy.deepcopy(self.db)
        # Remove prompt-related keys that should not be checkpointed
        prompt_keys_to_exclude = ["prompts", "temp_prompts"]
        for key in prompt_keys_to_exclude:
            if key in db_without_prompts:
                del db_without_prompts[key]
        
        checkpoint = {
            "current_i": self.current_i,
            "previous_i": self.previous_i,
            "skills": copy.deepcopy(self.skills),
            "db": db_without_prompts,
            "execution_order": copy.deepcopy(getattr(self, 'execution_order', None))
        }
        self._checkpoints[checkpoint_name] = checkpoint
        print(f"Created checkpoint: {checkpoint_name}")

    def restore_checkpoint(self, checkpoint_name):
        """Restore state from a checkpoint (preserving current prompts)"""
        if checkpoint_name not in self._checkpoints:
            raise ValueError(f"Checkpoint '{checkpoint_name}' does not exist")
        
        checkpoint = self._checkpoints[checkpoint_name]
        
        # Preserve current prompts before restoration
        current_prompts = {}
        prompt_keys_to_preserve = ["prompts", "temp_prompts"]
        for key in prompt_keys_to_preserve:
            if key in self.db:
                current_prompts[key] = self.db[key]
        
        # Restore checkpoint state
        self.current_i = checkpoint["current_i"]
        self.previous_i = checkpoint["previous_i"]
        self.skills = copy.deepcopy(checkpoint["skills"])
        self.db = copy.deepcopy(checkpoint["db"])
        if checkpoint["execution_order"] is not None:
            self.execution_order = copy.deepcopy(checkpoint["execution_order"])
        
        # Restore preserved prompts
        for key, value in current_prompts.items():
            self.db[key] = value
        
        print(f"Restored from checkpoint: {checkpoint_name} (prompts preserved)")

    def save_checkpoint_to_disk(self, checkpoint_name):
        """Save a checkpoint to disk"""
        if checkpoint_name not in self._checkpoints:
            raise ValueError(f"Checkpoint '{checkpoint_name}' does not exist")
        
        checkpoint_dir = self.graph_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_file = checkpoint_dir / f"{checkpoint_name}.pkl"
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(self._checkpoints[checkpoint_name], f)
        
        print(f"Saved checkpoint to disk: {checkpoint_file}")

    def load_checkpoint_from_disk(self, checkpoint_name):
        """Load a checkpoint from disk"""
        checkpoint_file = self.graph_path / "checkpoints" / f"{checkpoint_name}.pkl"
        if not checkpoint_file.exists():
            raise ValueError(f"Checkpoint file '{checkpoint_file}' does not exist")
        
        with open(checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self._checkpoints[checkpoint_name] = checkpoint
        print(f"Loaded checkpoint from disk: {checkpoint_file}")

    def list_checkpoints(self):
        """List available checkpoints"""
        print("Available checkpoints:")
        for name in self._checkpoints.keys():
            print(f"  - {name}")
        
        # Also check disk checkpoints
        checkpoint_dir = self.graph_path / "checkpoints"
        if checkpoint_dir.exists():
            disk_checkpoints = [f.stem for f in checkpoint_dir.glob("*.pkl")]
            if disk_checkpoints:
                print("Disk checkpoints:")
                for name in disk_checkpoints:
                    print(f"  - {name} (on disk)")

    def cleanup_checkpoint(self, checkpoint_name):
        """Remove a checkpoint from memory"""
        if checkpoint_name in self._checkpoints:
            del self._checkpoints[checkpoint_name]
            print(f"Cleaned up checkpoint: {checkpoint_name}")

