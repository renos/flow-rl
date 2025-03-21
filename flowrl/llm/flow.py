import pickle
from flowrl.llm.craftax_classic.llm import generate_graph
from flowrl.llm.craftax_classic.generate_code import (
    validate_code,
    generate_validated_py,
    create_inventory_from_array,
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
    describe_block_environment,
    explain_trajectory,
)
import numpy as np


class Flow:
    def __init__(self, args):
        self.args = args
        self.graph_path = Path(args.graph_path)
        if args.env_name == "Craftax-Classic-Symbolic-v1":
            self.db, self.graph, self.inventory_graph = generate_graph(
                return_inventory_graph=True
            )
        else:
            assert 0, f"Not Implemented"

        self.current_i = args.current_i
        self.previous_i = args.previous_i

        self.nodes = []

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
                    self.graph_path / str(self.previous_i) / "nodes.json",
                    "r",
                    encoding="utf-8",
                ):
                    nodes = json.load(f)
                self.nodes = nodes["data"]
            except FileNotFoundError:
                print(
                    f"Warning: Could not find nodes.json for previous_i={self.previous_i}"
                )

    def add_node(self, node):
        """Add a node to the graph and update the current index"""
        self.nodes.append(node)

    def next_node(self):
        """Generate and validate code from the graph"""
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
        return functions

        # if error == "":
        #     function_code = generate_validated_py(
        #         functions, self.args.module_path, self.current_i
        #     )
        #     self.db["current"]["completed_subtasks"].append(
        #         self.db["current"]["subtask_name"]
        #     )
        #     breakpoint()
        #     return function_code
        # else:
        #     assert 0, "Error generating code"

    def write_code(self, next_node):

        nodes_plus_next = self.nodes.copy()
        nodes_plus_next.append(next_node)

        breakpoint()
        self.current_i = len(nodes_plus_next) - 1
        for i in range(len(nodes_plus_next)):
            generate_validated_py(
                nodes_plus_next[i],
                os.path.join(
                    self.graph_path / str(self.current_i), f"{self.current_i}.py"
                ),
                i,
            )

    def update_db(self, out, function_code):
        average_item = out["info"]["average_item"][0]
        average_item = np.dot(out["info"]["average_item"][0], np.arange(10))
        closest_blocks = out["info"]["closest_blocks"][0]
        if self.current_i == 0:
            self.db["past"] = {}
            self.db["past"]["inventory"] = {}
            self.db["past"]["inventory"][self.current_i] = create_inventory_from_array(
                np.zeros_like(average_item)
            )
        closest_blocks_description = describe_block_environment(
            closest_blocks, game="craftax_classic"
        )
        inventory = create_inventory_from_array(average_item)

        self.db["past"]["inventory"][self.current_i + 1] = inventory

        achievements = {
            key: value[-1]
            for key, value in out["info"].items()
            if "Achievements" in key
        }
        self.db["past_inventory"] = self.db["current"]["inventory"]
        self.db["current"]["inventory"] = inventory
        self.db["current"]["closest_blocks"] = closest_blocks_description
        self.db["current"]["achievements_completed"] = {
            key: f"{value:.1f}%" for key, value in achievements.items()
        }
        self.db["current"]["num_skills"] += not self.db["current"]["subtask"][
            "reuse_skill"
        ]
        self.db["skills"][self.db["current"]["subtask_name"]] = function_code

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
