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

        # Load previous db if specified
        if self.previous_i is not None and self.previous_i >= 0:
            try:
                with open(self.graph_path / str(self.previous_i) / "db.pkl", "rb") as f:
                    db_old = pickle.load(f)
                for key, value in db_old.items():
                    self.db[key] = value
            except FileNotFoundError:
                print(
                    f"Warning: Could not find db.pkl for previous_i={self.previous_i}"
                )
            try:
                with open(
                    self.graph_path / str(self.previous_i) / "skills.json",
                    "r",
                    encoding="utf-8",
                ) as f:
                    skills_data = json.load(f)
                self.skills = skills_data["data"]
            except FileNotFoundError:
                print(
                    f"Warning: Could not find skills.json for previous_i={self.previous_i}"
                )

    def add_skill(self, skill_name, skill_data):
        """Add a skill to the skills dictionary"""
        self.skills[skill_name] = skill_data

    def next_skill(self):
        """Generate and validate a complete skill from the graph"""
        error = "Not generated yet"
        while error != "":
            evaluated = self.graph.evaluate()
            generated_code = evaluated[list(evaluated.keys())[-1]]
            functions, error = validate_code(generated_code)
            if error != "":
                print(f"Error generating: {error}")
        
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
        self.dependency_order = resolver.resolve_dependencies(target_skill_name)
        print(f"Dependency order for skill '{target_skill_name}': {self.dependency_order}")

    def write_code(self):

        nodes = self.nodes.copy()
        save_dir = self.graph_path / str(self.current_i)
        save_dir.mkdir(parents=True, exist_ok=True)

        self.current_i = len(nodes) - 1
        for i in range(len(nodes)):
            generate_validated_py(
                nodes[i],
                os.path.join(
                    self.graph_path / str(self.current_i), f"{self.current_i}.py"
                ),
                i,
            )

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

    def explain_trajectory(self, env_states, actions):
        """Explain the trajectory of the generated code"""
        try:
            self.db["example_trajectory"] = explain_trajectory(
                env_states, actions, start_state=self.current_i
            )
        except:
            breakpoint()

        self.inventory_graph.evaluate()
        return self.db["current"]["missing_resources"]

