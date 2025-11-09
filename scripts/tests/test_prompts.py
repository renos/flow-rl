"""
Test prompts using the original graph structure.
This module provides lightweight utilities for testing LLM prompts using the actual AgentKit graph.
"""

import os
import sys
from pathlib import Path

# Add the flowrl package to path if needed
sys.path.append(str(Path(__file__).parent))

from flowrl.llm.craftax_classic.llm import generate_graph
from flowrl.llm.craftax_classic.generate_code import validate_code, create_inventory_from_array
from flowrl.skill_dependency_resolver import SkillDependencyResolver
from flowrl.skill_depenency_resolver_new import SymbolicSkillDependencyResolver
from flowrl.skill_dependency_resolver_unified import UnifiedPlanningSkillResolver
import numpy as np
import json


class PromptTester:
    """Lightweight class for testing prompts using the original graph structure."""
    
    def __init__(self, env_name="Craftax-Classic-Symbolic-v1"):
        """Initialize the prompt tester with original graph structure."""
        self.env_name = env_name
        
        # Initialize the graph and database using the original method
        self.db, self.graph, self.inventory_graph, self.reuse_graph = generate_graph(
            return_inventory_graph=True
        )
        
        # Store original state for reset
        self.original_db = self._deep_copy_db(self.db)
        
    def _deep_copy_db(self, db):
        """Create a deep copy of the database for reset functionality."""
        import copy
        return copy.deepcopy(db)
        
    def reset_db(self):
        """Reset the database to original state."""
        self.db = self._deep_copy_db(self.original_db)
        
    def set_game_state(self, inventory=None, closest_blocks=None, achievements=None, status=None):
        """
        Set the current game state for testing.
        
        Args:
            inventory: Dict or Inventory object representing current inventory
            closest_blocks: String description of nearby blocks
            achievements: Dict of achievement completion percentages
            status: String representation of player status
        """
        if inventory is not None:
            if isinstance(inventory, dict):
                # Convert dict to Inventory string representation
                inv_str = f"Inventory({', '.join(f'{k}={v}' for k, v in inventory.items())})"
                self.db["current"]["inventory"] = inv_str
            else:
                self.db["current"]["inventory"] = str(inventory)
                
        if closest_blocks is not None:
            self.db["current"]["closest_blocks"] = closest_blocks
            
        if achievements is not None:
            self.db["current"]["achievements_completed"] = achievements
            
        if status is not None:
            self.db["current"]["status"] = status
    
    def add_skill(self, skill_name, skill_code):
        """
        Add a skill to the existing skills database.
        
        Args:
            skill_name: Name of the skill
            skill_code: The skill code/implementation as a string or dict
        """
        if isinstance(skill_code, str):
            # Parse JSON string if it's a string
            try:
                skill_data = json.loads(skill_code)
            except json.JSONDecodeError:
                # If it's not JSON, treat as raw code
                self.db["skills"][skill_name] = skill_code
                print(f"Added skill: {skill_name}")
                return
        else:
            skill_data = skill_code
        
        # Store in the format expected by SkillDependencyResolver
        skill_entry = {
            "skill_name": skill_name,
            "skill_with_consumption": skill_data,
            "functions": [],  # Empty for testing
            "iteration": 0
        }
        self.db["skills"][skill_name] = skill_entry
        print(f"Added skill: {skill_name}")
    
    def load_skills_from_json(self, json_file_path):
        """
        Load all skills from a JSON file.

        Supports two formats:
        1. Training format: {"data": {skill_name: skill_info, ...}}
        2. Knowledge base format: {"skills": [skill_dict, ...], "tutorial_context": ...}

        Args:
            json_file_path: Path to the skills.json or knowledgebase.json file
        """
        import json
        from pathlib import Path

        json_path = Path(json_file_path)
        assert json_path.exists(), f"Skills file not found: {json_file_path}"

        print(f"Loading skills from: {json_file_path}")

        with open(json_path, 'r') as f:
            skills_data = json.load(f)

        # Clear existing skills first
        self.db["skills"].clear()

        skills_added = 0

        # Detect format and load accordingly
        if "data" in skills_data:
            # Training format: {"data": {skill_name: skill_info}}
            print("Detected training format (with 'data' key)")
            skills = skills_data["data"]

            for skill_name, skill_info in skills.items():
                # The skill_info already has the correct format with skill_with_consumption
                self.db["skills"][skill_name] = skill_info
                skills_added += 1
                print(f"  Added: {skill_name}")

        elif "skills" in skills_data and isinstance(skills_data["skills"], list):
            # Knowledge base format: {"skills": [skill_dict, ...]}
            print("Detected knowledge base format (with 'skills' list)")
            skills_list = skills_data["skills"]

            for skill_dict in skills_list:
                skill_name = skill_dict.get("skill_name")
                assert skill_name, "Skill missing 'skill_name' field in knowledge base"

                # Convert knowledge base format to internal format
                skill_entry = {
                    "skill_name": skill_name,
                    "skill_with_consumption": {
                        "skill_name": skill_name,
                        "description": skill_dict.get("description", ""),
                        "requirements": skill_dict.get("requirements", {}),
                        "consumption": skill_dict.get("consumption", {}),
                        "gain": skill_dict.get("gain", {}),
                        "ephemeral": skill_dict.get("ephemeral", False),
                    },
                    "functions": [],  # Knowledge base skills don't have generated code yet
                    "iteration": 0
                }

                self.db["skills"][skill_name] = skill_entry
                skills_added += 1
                print(f"  Added: {skill_name}")

        else:
            raise ValueError(
                "Unknown skills JSON format. Expected either:\n"
                "  - Training format: {'data': {skill_name: skill_info, ...}}\n"
                "  - Knowledge base format: {'skills': [skill_dict, ...], 'tutorial_context': ...}"
            )

        print(f"Successfully loaded {skills_added} skills from {json_path.name}")
        return skills_added

    def load_skills_from_scheduler_state(self, scheduler_state_path):
        """
        Load all skills from a parallel scheduler_state.json file.

        Args:
            scheduler_state_path: Path to the scheduler_state.json file from parallel training
        """
        import json
        from pathlib import Path

        state_path = Path(scheduler_state_path)
        if not state_path.exists():
            raise FileNotFoundError(f"Scheduler state file not found: {scheduler_state_path}")

        print(f"Loading skills from scheduler state: {scheduler_state_path}")

        with open(state_path, 'r') as f:
            scheduler_state = json.load(f)

        # Extract skills from the "skills" key
        if "skills" not in scheduler_state:
            raise ValueError("Expected 'skills' key in scheduler state JSON file")

        skills = scheduler_state["skills"]

        # Clear existing skills first
        self.db["skills"].clear()

        # Add each skill to the database
        # In the scheduler state, each skill has a different format:
        # - It has "skill_name" and "skill_data" keys
        # - skill_data contains the actual skill information
        skills_added = 0
        for skill_key, skill_entry in skills.items():
            if "skill_data" not in skill_entry:
                print(f"  Warning: Skipping {skill_key}, no skill_data found")
                continue

            skill_data = skill_entry["skill_data"]
            skill_name = skill_data.get("skill_name", skill_key)

            # The skill_data has the right format with skill_with_consumption, functions, etc.
            self.db["skills"][skill_name] = skill_data
            skills_added += 1

            # Show status information
            status = skill_entry.get("status", "unknown")
            print(f"  Added: {skill_name} (status: {status})")

        print(f"Successfully loaded {skills_added} skills from scheduler state")
        return skills_added
        
    def remove_skill(self, skill_name):
        """
        Remove a skill from the existing skills database.
        
        Args:
            skill_name: Name of the skill to remove
        """
        if skill_name in self.db["skills"]:
            del self.db["skills"][skill_name]
            print(f"Removed skill: {skill_name}")
        else:
            print(f"Skill '{skill_name}' not found")
            
    def list_skills(self):
        """List all existing skills."""
        if not self.db["skills"]:
            print("No skills available")
        else:
            print("EXISTING SKILLS:")
            for skill_name in self.db["skills"].keys():
                print(f"  - {skill_name}")
                
    def edit_inventory_item(self, item_name, value):
        """
        Edit a specific inventory item.
        
        Args:
            item_name: Name of the inventory item (e.g., 'wood', 'stone', 'wood_pickaxe')
            value: New value for the item (0-9)
        """
        # Parse current inventory string to extract values
        current_inv = self.db["current"]["inventory"]
        
        # Basic inventory items
        inventory_items = {
            'wood': 0, 'stone': 0, 'coal': 0, 'iron': 0, 'diamond': 0, 'sapling': 0,
            'wood_pickaxe': 0, 'stone_pickaxe': 0, 'iron_pickaxe': 0,
            'wood_sword': 0, 'stone_sword': 0, 'iron_sword': 0
        }
        
        # Extract current values from the inventory string
        import re
        for item in inventory_items.keys():
            match = re.search(f'{item}=(\d+)', current_inv)
            if match:
                inventory_items[item] = int(match.group(1))
        
        # Update the specific item
        if item_name in inventory_items:
            inventory_items[item_name] = max(0, min(9, value))  # Clamp to 0-9
            
            # Reconstruct inventory string
            inv_str = f"Inventory({', '.join(f'{k}={v}' for k, v in inventory_items.items())})"
            self.db["current"]["inventory"] = inv_str
            print(f"Updated {item_name} to {inventory_items[item_name]}")
        else:
            print(f"Unknown inventory item: {item_name}")
            print(f"Available items: {list(inventory_items.keys())}")
            
    def set_inventory_from_dict(self, inventory_dict):
        """
        Set inventory from a dictionary.
        
        Args:
            inventory_dict: Dict with item names as keys and values as counts
        """
        self.set_game_state(inventory=inventory_dict)
        
    def show_current_skills_and_inventory(self):
        """Show current skills and inventory for easy editing."""
        print("CURRENT SKILLS:")
        print("=" * 30)
        if not self.db["skills"]:
            print("  No skills available")
        else:
            for skill_name, skill_data in self.db["skills"].items():
                print(f"  - {skill_name}")
                if isinstance(skill_data, dict) and "skill_with_consumption" in skill_data:
                    # New format: show requirements and gains
                    skill_info = skill_data["skill_with_consumption"]
                    requirements = skill_info.get("requirements", {})
                    gain = skill_info.get("gain", {})
                    print(f"    Requirements: {requirements}")
                    print(f"    Gain: {gain}")
                else:
                    # Old format: show code preview
                    skill_str = str(skill_data)
                    print(f"    Code preview: {skill_str[:100]}...")
        
        print("\nCURRENT INVENTORY:")
        print("=" * 30)
        print(f"  {self.db['current']['inventory']}")
        
        # Parse and show in a nicer format
        import re
        inventory_items = {}
        for match in re.finditer(r'(\w+)=(\d+)', self.db['current']['inventory']):
            item_name, value = match.groups()
            inventory_items[item_name] = int(value)
            
        print("\n  Parsed inventory:")
        for item, count in inventory_items.items():
            if count > 0:
                print(f"    {item}: {count}")
        print("=" * 30)
    
    def _find_dependencies(self, prompt_name, visited=None):
        """
        Find all dependencies for a given prompt recursively.
        
        Args:
            prompt_name: Name of the prompt to find dependencies for
            visited: Set of already visited prompts to avoid cycles
            
        Returns:
            Set of prompt names that this prompt depends on (including itself)
        """
        if visited is None:
            visited = set()
            
        if prompt_name in visited:
            return set()
            
        visited.add(prompt_name)
        dependencies = {prompt_name}
        
        # Check both prompts and temp_prompts
        prompt_info = None
        if prompt_name in self.db["prompts"]:
            prompt_info = self.db["prompts"][prompt_name]
        elif prompt_name in self.db["temp_prompts"]:
            prompt_info = self.db["temp_prompts"][prompt_name]
            
        if prompt_info is None:
            return dependencies
            
        # Add direct dependencies
        if "dep" in prompt_info:
            for dep in prompt_info["dep"]:
                dependencies.update(self._find_dependencies(dep, visited.copy()))
                
        # Add "after" dependencies
        if "after" in prompt_info:
            for dep in prompt_info["after"]:
                dependencies.update(self._find_dependencies(dep, visited.copy()))
                
        return dependencies

    def test_single_prompt(self, prompt_name, show_prompt=False):
        """
        Test a single prompt by building a minimal subgraph with only its dependencies.
        
        Args:
            prompt_name: Name of the prompt to test (e.g., "next_task", "next_subtask")
            show_prompt: Whether to print the formatted prompt
            
        Returns:
            The LLM response
        """
        if prompt_name not in self.db["prompts"] and prompt_name not in self.db["temp_prompts"]:
            available = list(self.db["prompts"].keys()) + list(self.db["temp_prompts"].keys())
            raise ValueError(f"Unknown prompt '{prompt_name}'. Available prompts: {available}")
        
        if show_prompt:
            print("="*50)
            print(f"TESTING SINGLE PROMPT: {prompt_name}")
            print("="*50)
        
        # Find all dependencies for this prompt
        dependencies = self._find_dependencies(prompt_name)
        print(f"Dependencies found: {dependencies}")
        
        # Create a minimal subgraph with only the necessary nodes
        from agentkit import Graph, SimpleDBNode
        subgraph = Graph()
        
        # Add only the nodes we need
        for dep_name in dependencies:
            if dep_name in self.db["prompts"]:
                prompt_info = self.db["prompts"][dep_name]
            elif dep_name in self.db["temp_prompts"]:
                prompt_info = self.db["temp_prompts"][dep_name]
            else:
                continue
                
            # Create the node
            node = SimpleDBNode(
                prompt_info["prompt"],
                prompt_info["prompt"],
                subgraph,
                prompt_info["query"],
                prompt_info["compose"],
                self.db,
                after_query=prompt_info.get("after_query"),
                verbose=True
            )
            subgraph.add_node(node)
        
        # Add edges between nodes in the subgraph
        for dep_name in dependencies:
            if dep_name in self.db["prompts"]:
                prompt_info = self.db["prompts"][dep_name]
            elif dep_name in self.db["temp_prompts"]:
                prompt_info = self.db["temp_prompts"][dep_name]
            else:
                continue
                
            current_prompt_text = prompt_info["prompt"]
            
            # Add dependency edges
            if "dep" in prompt_info:
                for dep in prompt_info["dep"]:
                    if dep in dependencies:
                        if dep in self.db["prompts"]:
                            dep_prompt_text = self.db["prompts"][dep]["prompt"]
                        elif dep in self.db["temp_prompts"]:
                            dep_prompt_text = self.db["temp_prompts"][dep]["prompt"]
                        else:
                            continue
                        subgraph.add_edge(dep_prompt_text, current_prompt_text)
                        
            # Add order edges
            if "after" in prompt_info:
                for dep in prompt_info["after"]:
                    if dep in dependencies:
                        if dep in self.db["prompts"]:
                            dep_prompt_text = self.db["prompts"][dep]["prompt"]
                        elif dep in self.db["temp_prompts"]:
                            dep_prompt_text = self.db["temp_prompts"][dep]["prompt"]
                        else:
                            continue
                        subgraph.add_order(dep_prompt_text, current_prompt_text)
        
        print(f"Built subgraph with {len(subgraph.nodes)} nodes")
        
        # Execute the minimal subgraph
        results = subgraph.evaluate()
        
        # Get the target prompt text
        if prompt_name in self.db["prompts"]:
            target_prompt_text = self.db["prompts"][prompt_name]["prompt"]
        else:
            target_prompt_text = self.db["temp_prompts"][prompt_name]["prompt"]
        
        # Find our target result
        target_result = results.get(target_prompt_text)
        
        if target_result is None:
            print(f"Warning: No result found for prompt '{prompt_name}'")
            print("Available results:")
            for key in results.keys():
                print(f"  - {key[:60]}...")
            return None
        
        print(f"\nLLM RESPONSE for '{prompt_name}':")
        print("-" * 30)
        if isinstance(target_result, tuple):
            print(target_result[0])
            if len(target_result) > 1:
                print(f"Token usage: {target_result[1]}")
        else:
            print(target_result)
        print("-" * 30)
        
        return target_result
    
    def test_main_workflow(self):
        """Test the main task generation workflow using graph.evaluate()."""
        print("Testing main workflow using graph.evaluate()...")
        print("="*50)
        
        # Execute the main graph - this runs the full workflow
        results = self.graph.evaluate()
        
        print("\n=== WORKFLOW RESULTS ===")
        for key, value in results.items():
            print(f"\nPrompt: {key[:100]}...")
            print("-" * 40)
            if isinstance(value, tuple):
                print(f"Response: {value[0]}")
                if len(value) > 1:
                    print(f"Token usage: {value[1]}")
            else:
                print(f"Response: {value}")
                
        return results
        
    def test_inventory_workflow(self):
        """Test the inventory analysis workflow."""
        print("Testing inventory workflow using inventory_graph.evaluate()...")
        print("="*50)
        
        results = self.inventory_graph.evaluate()
        
        print("\n=== INVENTORY WORKFLOW RESULTS ===")
        for key, value in results.items():
            print(f"\nPrompt: {key[:100]}...")
            print("-" * 40)
            if isinstance(value, tuple):
                print(f"Response: {value[0]}")
                if len(value) > 1:
                    print(f"Token usage: {value[1]}")
            else:
                print(f"Response: {value}")
                
        return results
        
    def test_single_inventory_prompt(self, prompt_name, show_prompt=False):
        """
        Test a single inventory prompt.
        
        Args:
            prompt_name: Name of the inventory prompt to test (e.g., "predict_item_count", "predict_missing_items")
            show_prompt: Whether to print the formatted prompt
            
        Returns:
            The LLM response
        """
        # Get available inventory prompts
        from flowrl.llm.craftax_classic.prompts.inventory import return_prompts
        from flowrl.llm.craftax_classic.llm import get_query, llm_name
        from functools import partial
        
        LLM_API_FUNCTION_GPT4 = partial(get_query(llm_name), max_gen=4096)
        inventory_prompts = return_prompts(LLM_API_FUNCTION_GPT4)
        
        if prompt_name not in inventory_prompts:
            available = list(inventory_prompts.keys())
            raise ValueError(f"Unknown inventory prompt '{prompt_name}'. Available prompts: {available}")
        
        if show_prompt:
            print("="*50)
            print(f"TESTING SINGLE INVENTORY PROMPT: {prompt_name}")
            print("="*50)
        
        # Find all dependencies for this prompt
        dependencies = self._find_inventory_dependencies(prompt_name, inventory_prompts)
        print(f"Dependencies found: {dependencies}")
        
        # Create a minimal subgraph with only the necessary nodes
        from agentkit import Graph, SimpleDBNode
        subgraph = Graph()
        
        # Add only the nodes we need
        for dep_name in dependencies:
            if dep_name in inventory_prompts:
                prompt_info = inventory_prompts[dep_name]
                
                # Create the node
                node = SimpleDBNode(
                    prompt_info["prompt"],
                    prompt_info["prompt"],
                    subgraph,
                    prompt_info["query"],
                    prompt_info["compose"],
                    self.db,
                    after_query=prompt_info.get("after_query"),
                    verbose=True
                )
                subgraph.add_node(node)
        
        # Add edges between nodes in the subgraph
        for dep_name in dependencies:
            if dep_name in inventory_prompts:
                prompt_info = inventory_prompts[dep_name]
                current_prompt_text = prompt_info["prompt"]
                
                # Add dependency edges
                if "dep" in prompt_info:
                    for dep in prompt_info["dep"]:
                        if dep in dependencies and dep in inventory_prompts:
                            dep_prompt_text = inventory_prompts[dep]["prompt"]
                            subgraph.add_edge(dep_prompt_text, current_prompt_text)
        
        print(f"Built inventory subgraph with {len(subgraph.nodes)} nodes")
        
        # Execute the minimal subgraph
        results = subgraph.evaluate()
        
        # Get the target prompt text
        target_prompt_text = inventory_prompts[prompt_name]["prompt"]
        
        # Find our target result
        target_result = results.get(target_prompt_text)
        
        if target_result is None:
            print(f"Warning: No result found for inventory prompt '{prompt_name}'")
            print("Available results:")
            for key in results.keys():
                print(f"  - {key[:60]}...")
            return None
        
        print(f"\nLLM RESPONSE for inventory prompt '{prompt_name}':")
        print("-" * 30)
        if isinstance(target_result, tuple):
            print(target_result[0])
            if len(target_result) > 1:
                print(f"Token usage: {target_result[1]}")
        else:
            print(target_result)
        print("-" * 30)
        
        return target_result
    
    def _find_inventory_dependencies(self, prompt_name, inventory_prompts, visited=None):
        """
        Find all dependencies for a given inventory prompt recursively.
        
        Args:
            prompt_name: Name of the prompt to find dependencies for
            inventory_prompts: Dictionary of inventory prompts
            visited: Set of already visited prompts to avoid cycles
            
        Returns:
            Set of prompt names that this prompt depends on (including itself)
        """
        if visited is None:
            visited = set()
            
        if prompt_name in visited:
            return set()
            
        visited.add(prompt_name)
        dependencies = {prompt_name}
        
        if prompt_name not in inventory_prompts:
            return dependencies
            
        prompt_info = inventory_prompts[prompt_name]
        
        # Add direct dependencies
        if "dep" in prompt_info:
            for dep in prompt_info["dep"]:
                dependencies.update(self._find_inventory_dependencies(dep, inventory_prompts, visited.copy()))
                
        return dependencies
    
    def resolve_skill_dependencies(
        self,
        skill_name,
        n=1,
        max_inventory_capacity=9,
        use_symbolic_state=False,
        use_unified_planner=False,
        initial_symbolic_state=None,
    ):
        """Resolve dependencies for a skill and return the execution order.

        Args:
            skill_name: Name of the skill to resolve dependencies for
            n: Number of times to apply this skill (default: 1)
            max_inventory_capacity: Maximum inventory capacity per item (default: 9)
            use_symbolic_state: When True, plan over the richer symbolic state representation
            use_unified_planner: When True, use the Unified Planning Library (much simpler!)
            initial_symbolic_state: Optional `SymbolicState` to seed the symbolic resolver

        Returns:
            List of nodes/skills in execution order to fulfill the target skill
        """
        if skill_name not in self.db["skills"]:
            available_skills = list(self.db["skills"].keys())
            raise ValueError(f"Skill '{skill_name}' not found. Available skills: {available_skills}")

        print(
            f"Resolving dependencies for '{skill_name}' with n={n}, max_inventory={max_inventory_capacity}, "
            f"symbolic={use_symbolic_state}, unified_planner={use_unified_planner}"
        )

        if use_unified_planner:
            resolver = UnifiedPlanningSkillResolver(
                self.db["skills"],
                max_inventory_capacity,
                initial_state=initial_symbolic_state,
            )
            execution_order = resolver.resolve_dependencies(
                skill_name, n, initial_state=initial_symbolic_state
            )
        elif use_symbolic_state:
            resolver = SymbolicSkillDependencyResolver(
                self.db["skills"],
                max_inventory_capacity,
                initial_state=initial_symbolic_state,
            )
            execution_order = resolver.resolve_dependencies(
                skill_name, n, initial_state=initial_symbolic_state
            )
        else:
            resolver = SkillDependencyResolver(
                self.db["skills"], max_inventory_capacity
            )
            execution_order = resolver.resolve_dependencies(skill_name, n)

        return execution_order

    def resolve_skill_dependencies_symbolic(
        self,
        skill_name,
        n=1,
        max_inventory_capacity=9,
        initial_symbolic_state=None,
    ):
        """Convenience wrapper that forces symbolic-state planning."""
        return self.resolve_skill_dependencies(
            skill_name,
            n,
            max_inventory_capacity,
            use_symbolic_state=True,
            initial_symbolic_state=initial_symbolic_state,
        )

    def write_execution_order_to_file(self, execution_order, output_path, module_name=None):
        """
        Write the execution order to a Python module file for use in training.
        
        Args:
            execution_order: List of (skill_name, count) tuples from resolve_skill_dependencies
            output_path: Path where to write the Python module file
            module_name: Optional base name for the module (defaults to filename without extension)
        """
        from pathlib import Path
        import os
        from flowrl.llm.craftax_classic.generate_code import generate_validated_py
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if module_name is None:
            module_name = output_path.stem
            
        print(f"Writing {len(execution_order)} execution steps to {output_path}")
        
        # Clear the file first
        if output_path.exists():
            output_path.unlink()
        
        # Generate the module file with one node per execution step
        skills_written = 0
        for task_index, (skill_name, count) in enumerate(execution_order):
            if skill_name in self.db["skills"]:
                skill_data = self.db["skills"][skill_name]
                functions = skill_data.get("functions", [])
                
                if len(functions) >= 3:
                    # generate_validated_py expects exactly 3 functions: [task_is_done, task_reward, task_network_number]
                    generate_validated_py(functions[:3], str(output_path), task_index, n=count)
                    print(f"  Added task_{task_index}: {skill_name} (n={count})")
                    skills_written += 1
                else:
                    print(f"  Warning: Skill '{skill_name}' has {len(functions)} functions, expected 3 - skipping")
            else:
                print(f"  Warning: Skill '{skill_name}' not found in skills database - skipping")
        
        print(f"Successfully wrote {skills_written}/{len(execution_order)} tasks to {output_path}")
        return skills_written
    
    def test_frontier_calculation(self, max_inventory_capacity=9):
        """
        Run the frontier calculation on loaded skills and time it.

        Args:
            max_inventory_capacity: Maximum inventory capacity for the environment

        Returns:
            Tuple of (frontier_summary, elapsed_time_seconds)
        """
        import time
        from flowrl.llm.craftax.symbolic_state import compute_frontier_summary

        print(f"\n=== FRONTIER CALCULATION TEST ===")
        print(f"Number of skills: {len(self.db['skills'])}")
        print(f"Max inventory capacity: {max_inventory_capacity}")
        print(f"Starting frontier calculation...")

        start_time = time.time()
        frontier_summary = compute_frontier_summary(self.db["skills"], max_inventory_capacity)
        elapsed_time = time.time() - start_time

        print(f"✓ Frontier calculation completed in {elapsed_time:.3f} seconds")
        print(f"\n=== FRONTIER SUMMARY ===")
        print(frontier_summary)
        print(f"=== END FRONTIER SUMMARY ===\n")

        return frontier_summary, elapsed_time

    def create_full_module(
        self,
        skill_name,
        n=1,
        output_path=None,
        max_inventory_capacity=9,
        use_symbolic_state=False,
        use_unified_planner=False,
        initial_symbolic_state=None,
        test_frontier=False,
    ):
        """
        Complete workflow: resolve dependencies and write to module file.

        Args:
            skill_name: Target skill to resolve dependencies for
            n: Number of times to apply this skill
            output_path: Path to write the module file (defaults to ./{skill_name.lower().replace(' ', '_')}.py)
            max_inventory_capacity: Maximum inventory capacity per item
            use_symbolic_state: When True, execute dependency planning with SymbolicState resolver
            use_unified_planner: When True, use the Unified Planning Library (much simpler!)
            initial_symbolic_state: Optional symbolic state override for planning context
            test_frontier: When True, also run frontier calculation and time it

        Returns:
            Tuple of (execution_order, skills_written_count)
        """
        from pathlib import Path

        # Optionally test frontier calculation
        if test_frontier:
            self.test_frontier_calculation(max_inventory_capacity)

        # Resolve dependencies
        execution_order = self.resolve_skill_dependencies(
            skill_name,
            n,
            max_inventory_capacity,
            use_symbolic_state=use_symbolic_state,
            use_unified_planner=use_unified_planner,
            initial_symbolic_state=initial_symbolic_state,
        )

        # Default output path if not provided
        if output_path is None:
            safe_name = skill_name.lower().replace(' ', '_').replace('-', '_')
            output_path = Path(f"{safe_name}_n{n}.py")
        else:
            output_path = Path(output_path)

        # Write to file
        skills_written = self.write_execution_order_to_file(execution_order, output_path)

        print(f"\n=== MODULE CREATION COMPLETE ===")
        print(f"Target skill: {skill_name} (n={n})")
        print(f"Execution steps: {len(execution_order)}")
        print(f"Skills with code written: {skills_written}")
        print(f"Output file: {output_path.absolute()}")

        return execution_order, skills_written
        
    def test_reuse_workflow(self):
        """Test the skill reuse workflow."""
        print("Testing reuse workflow using reuse_graph.evaluate()...")
        print("="*50)
        
        results = self.reuse_graph.evaluate()
        
        print("\n=== REUSE WORKFLOW RESULTS ===")
        for key, value in results.items():
            print(f"\nPrompt: {key[:100]}...")
            print("-" * 40)
            if isinstance(value, tuple):
                print(f"Response: {value[0]}")
                if len(value) > 1:
                    print(f"Token usage: {value[1]}")
            else:
                print(f"Response: {value}")
                
        return results
        
    def test_full_pipeline(self):
        """Test the complete pipeline: main -> inventory -> reuse/create."""
        print("Testing full pipeline...")
        print("="*60)
        
        # Step 1: Main workflow
        print("\n1. MAIN WORKFLOW")
        main_results = self.test_main_workflow()
        
        # Step 2: Inventory workflow
        print("\n2. INVENTORY WORKFLOW")
        inventory_results = self.test_inventory_workflow()
        
        # Step 3: Check if we should reuse or create
        should_reuse = self.db["current"].get("subtask", {}).get("reuse_skill", False)
        
        if should_reuse:
            print("\n3. REUSE WORKFLOW (Skill reuse detected)")
            reuse_results = self.test_reuse_workflow()
            return {
                "main": main_results,
                "inventory": inventory_results, 
                "reuse": reuse_results
            }
        else:
            print("\n3. CREATE WORKFLOW (New skill creation)")
            # The create workflow nodes should have been added by the main workflow
            # Check if there are any temporary nodes to evaluate
            temp_results = {}
            print("Create workflow nodes should be automatically added to main graph")
            return {
                "main": main_results,
                "inventory": inventory_results,
                "create": temp_results
            }
        
    def validate_generated_code(self, code_response):
        """
        Validate generated code for syntax and function correctness.
        
        Args:
            code_response: The LLM response containing generated code
            
        Returns:
            Tuple of (functions, error_message)
        """
        # Handle tuple response format
        if isinstance(code_response, tuple):
            code_text = code_response[0]
        else:
            code_text = code_response
            
        try:
            functions, error = validate_code(code_text)
            if error:
                print(f"Code validation failed: {error}")
                return None, error
            else:
                print("Code validation successful!")
                print(f"Generated {len(functions)} functions")
                return functions, ""
        except Exception as e:
            error_msg = f"Exception during validation: {e}"
            print(error_msg)
            return None, error_msg
            
    def simulate_inventory_update(self, average_item_array):
        """
        Simulate an inventory update based on average item array from training.
        
        Args:
            average_item_array: Numpy array of average item counts
        """
        inventory = create_inventory_from_array(average_item_array)
        self.db["current"]["inventory"] = str(inventory)
        print(f"Updated inventory: {inventory}")
        
    def show_current_state(self):
        """Display the current game state in the database."""
        print("CURRENT GAME STATE:")
        print("="*40)
        print(f"Status: {self.db['current']['status']}")
        print(f"Inventory: {self.db['current']['inventory']}")
        print(f"Achievements: {self.db['current']['achievements_completed']}")
        print(f"Closest Blocks: {self.db['current']['closest_blocks'][:200]}...")  # Truncate for readability
        if "task" in self.db["current"]:
            print(f"Current Task: {self.db['current']['task']}")
        if "subtask" in self.db["current"]:
            print(f"Current Subtask: {self.db['current']['subtask']}")
        if "skills" in self.db and self.db["skills"]:
            print(f"Available Skills: {list(self.db['skills'].keys())}")
        print("="*40)
        
    def list_available_prompts(self):
        """List all available prompts for testing."""
        print("AVAILABLE PROMPTS:")
        print("Main prompts:")
        for name in self.db["prompts"].keys():
            print(f"  - {name}")
        print("Template prompts:")
        for name in self.db["temp_prompts"].keys():
            print(f"  - {name}")
        print("Inventory prompts:")
        from flowrl.llm.craftax_classic.prompts.inventory import return_prompts
        from flowrl.llm.craftax_classic.llm import get_query, llm_name
        from functools import partial
        LLM_API_FUNCTION_GPT4 = partial(get_query(llm_name), max_gen=4096)
        inventory_prompts = return_prompts(LLM_API_FUNCTION_GPT4)
        for name in inventory_prompts.keys():
            print(f"  - {name}")
            
    def show_graph_info(self):
        """Display information about the graphs."""
        print("GRAPH INFORMATION:")
        print("="*40)
        print(f"Main graph nodes: {len(self.graph.nodes)}")
        print(f"Inventory graph nodes: {len(self.inventory_graph.nodes)}")
        print(f"Reuse graph nodes: {len(self.reuse_graph.nodes)}")
        print("="*40)

    def test_ephemeral_inlining(self):
        """
        Test the ephemeral skill inlining preprocessing with the Make Wood Pickaxe example.
        """
        print("TESTING EPHEMERAL SKILL INLINING")
        print("="*50)
        
        # Set up test skills: Collect Wood, Place Table (ephemeral), Make Wood Pickaxe
        self.reset_db()
        
        # Collect Wood skill - basic skill
        self.add_skill("Collect Wood", {
            "skill_name": "Collect Wood",
            "requirements": {},
            "consumption": {},
            "gain": {"wood": "lambda n: n"},
            "ephemeral": False
        })
        
        # Place Table skill - ephemeral skill  
        self.add_skill("Place Table", {
            "skill_name": "Place Table", 
            "requirements": {},
            "consumption": {"wood": "lambda n: 4*n"},
            "gain": {"crafting_table_placed": "lambda n: 1"},
            "ephemeral": True
        })
        
        # Make Wood Pickaxe skill - depends on ephemeral skill
        self.add_skill("Make Wood Pickaxe", {
            "skill_name": "Make Wood Pickaxe",
            "requirements": {"crafting_table_placed": "lambda _: 1"},
            "consumption": {"wood": "lambda n: 3*n"},
            "gain": {"wood_pickaxe": "lambda n: n"},
            "ephemeral": False
        })
        
        print("BEFORE INLINING:")
        print("Collect Wood requirements:", self.db["skills"]["Collect Wood"]["skill_with_consumption"]["requirements"])
        print("Collect Wood consumption:", self.db["skills"]["Collect Wood"]["skill_with_consumption"]["consumption"])
        print("Place Table requirements:", self.db["skills"]["Place Table"]["skill_with_consumption"]["requirements"])
        print("Place Table consumption:", self.db["skills"]["Place Table"]["skill_with_consumption"]["consumption"])
        print("Make Wood Pickaxe requirements:", self.db["skills"]["Make Wood Pickaxe"]["skill_with_consumption"]["requirements"])
        print("Make Wood Pickaxe consumption:", self.db["skills"]["Make Wood Pickaxe"]["skill_with_consumption"]["consumption"])
        
        # Test dependency resolution with ephemeral inlining
        try:
            execution_order = self.resolve_skill_dependencies("Make Wood Pickaxe", n=2)
            print(f"\nExecution order: {execution_order}")
            
            # Expected: [('Collect Wood', 10), ('Make Wood Pickaxe', 2)]
            # Should NOT include Place Table directly, and should collect 10 wood total
            expected_wood = 3*2 + 4*1  # 3 wood per pickaxe * 2 pickaxes + 4 wood for 1 table = 10
            print(f"Expected wood collection: {expected_wood}")
            
            # Verify the result
            collect_wood_found = False
            total_wood_collected = 0
            place_table_found = False
            
            for skill_name, count in execution_order:
                if skill_name == "Collect Wood":
                    collect_wood_found = True
                    total_wood_collected = count
                elif skill_name == "Place Table":
                    place_table_found = True
                    
            if collect_wood_found and not place_table_found:
                print("✓ SUCCESS: Ephemeral skill properly inlined")
                print(f"✓ Wood collection: {total_wood_collected} (expected: {expected_wood})")
                if total_wood_collected == expected_wood:
                    print("✓ Wood calculation correct!")
                else:
                    print(f"✗ Wood calculation incorrect: got {total_wood_collected}, expected {expected_wood}")
            else:
                print("✗ FAILED: Ephemeral skill not properly inlined")
                print(f"  Collect Wood found: {collect_wood_found}")
                print(f"  Place Table found: {place_table_found}")
                
        except Exception as e:
            print(f"Error during dependency resolution: {e}")
            import traceback
            traceback.print_exc()
            
        return execution_order
            
    def test_trajectory_analysis(self, example_trajectory, skill_data, knowledge_base=None):
        """
        Test trajectory analysis functionality to propose database updates.
        
        Args:
            example_trajectory: List of trajectory strings, e.g., ['Timestep 12: Action: DO, Gained 1 wood']
            skill_data: Dictionary with skill information including skill_with_consumption
            knowledge_base: Optional knowledge base dict, will load default if None
            
        Returns:
            Dictionary with proposed updates for both skill and knowledge base
        """
        print("TESTING TRAJECTORY ANALYSIS")
        print("="*50)
        
        # Load knowledge base if not provided
        if knowledge_base is None:
            from pathlib import Path
            import json
            kb_file = Path("resources/craftax_classic_knowledgebase_verified.json")
            if kb_file.exists():
                with open(kb_file, 'r') as f:
                    knowledge_base = json.load(f)
                print(f"Loaded knowledge base with {len(knowledge_base)} entries")
            else:
                print("Warning: Could not load knowledge base")
                knowledge_base = {}
        
        # Set up database for trajectory analysis
        self.db["example_trajectory"] = example_trajectory
        self.db["current"]["skill_with_consumption"] = skill_data.get("skill_with_consumption", {})
        self.db["knowledge_base"] = knowledge_base
        
        print(f"Current skill: {skill_data.get('skill_with_consumption', {}).get('skill_name', 'Unknown')}")
        print(f"Trajectory: {example_trajectory}")
        
        # Run the inventory graph analysis
        try:
            results = self.inventory_graph.evaluate()
            
            print("\n=== TRAJECTORY ANALYSIS RESULTS ===")
            
            # Extract skill updates
            if "skill_update_results" in self.db["current"]:
                skill_updates = self.db["current"]["skill_update_results"]
                print("\n1. SKILL UPDATES:")
                print(f"   Updated requirements: {skill_updates.get('updated_requirements', {})}")
                print(f"   Updated gain: {skill_updates.get('updated_gain', {})}")
                
            # Extract knowledge base updates applied
            if "kb_updates_applied" in self.db["current"]:
                kb_updates = self.db["current"]["kb_updates_applied"]
                print("\n2. KNOWLEDGE BASE UPDATES APPLIED:")
                for i, update in enumerate(kb_updates):
                    print(f"   Update {i+1}:")
                    print(f"     Path: {' -> '.join(update.get('path', []))}")
                    print(f"     Old: {update.get('old_requirements', [])}")
                    print(f"     New: {update.get('new_requirements', [])}")
                    print(f"     Reason: {update.get('reason', '')}")
            
            return {
                "skill_updates": self.db["current"].get("skill_update_results", {}),
                "kb_updates_applied": self.db["current"].get("kb_updates_applied", []),
                "raw_results": results
            }
            
        except Exception as e:
            print(f"Error during trajectory analysis: {e}")
            return {"error": str(e)}


# Convenience functions for Jupyter notebook usage
def create_tester():
    """Create a new PromptTester instance."""
    return PromptTester()

def quick_test_main(tester=None, inventory=None):
    """Quick test of main workflow with optional custom inventory."""
    if tester is None:
        tester = create_tester()
        
    if inventory:
        tester.set_game_state(inventory=inventory)
        
    return tester.test_main_workflow()

def quick_test_full_pipeline(tester=None):
    """Quick test of the full pipeline."""
    if tester is None:
        tester = create_tester()
        
    return tester.test_full_pipeline()

def test_trajectory_analysis_example(tester=None):
    """Test trajectory analysis with example data."""
    if tester is None:
        tester = create_tester()
    
    # Example trajectory and skill data
    example_trajectory = ['Timestep 12: Action: DO, Gained 1 wood']
    example_skill_data = {
        "skill_name": "Collect Wood",
        "skill_with_consumption": {
            "skill_name": "Collect Wood",
            "requirements": {},
            "gain": {"wood": "lambda n: n"},
            "ephemeral": False
        }
    }
    
    print("Testing with example trajectory and skill data:")
    print(f"Trajectory: {example_trajectory}")
    print(f"Skill: {example_skill_data['skill_with_consumption']['skill_name']}")
    
    return tester.test_trajectory_analysis(example_trajectory, example_skill_data)


# Example usage for Jupyter notebook:
"""
# Basic usage
tester = create_tester()
tester.show_current_state()
tester.list_available_prompts()
tester.show_graph_info()

# Test main workflow
main_results = tester.test_main_workflow()

# Test with custom game state
tester.set_game_state(
    inventory={"wood": 5, "stone": 2, "wood_pickaxe": 1},
    achievements={"Achievements/collect_wood": 95.0}
)

# Test full pipeline
pipeline_results = tester.test_full_pipeline()

# Test individual workflows
inventory_results = tester.test_inventory_workflow()
reuse_results = tester.test_reuse_workflow()

# Test trajectory analysis
trajectory_results = test_trajectory_analysis_example()

# Test trajectory analysis with custom data
custom_trajectory = ['Timestep 5: Action: MINE_TREE, Gained 2 wood', 'Timestep 6: Action: DO, Lost 1 wood']
custom_skill = {
    "skill_with_consumption": {
        "skill_name": "Make Table",
        "requirements": {"wood": "lambda n: 2*n"},
        "gain": {},
        "ephemeral": True
    }
}
custom_results = tester.test_trajectory_analysis(custom_trajectory, custom_skill)

# Test single inventory prompts
tester.test_single_inventory_prompt("predict_item_count")
tester.test_single_inventory_prompt("predict_missing_items")

# Test skill dependency resolution
execution_order = tester.resolve_skill_dependencies("Make Stone Sword", n=2)
print(f"Execution order: {execution_order}")

# Validate generated code (if any code was generated)
for key, value in pipeline_results["main"].items():
    if "coding" in key.lower():
        functions, error = tester.validate_generated_code(value)
        if functions:
            print(f"Successfully validated code from {key}")

# Reset for fresh testing
tester.reset_db()
"""

if __name__ == "__main__":
    # Example usage when run as script
    tester = create_tester()
    tester.show_current_state()
    tester.show_graph_info()
    print("\nTesting main workflow...")
    tester.test_main_workflow()

    print("\nSymbolic resolver example...")
    symbolic_order = tester.resolve_skill_dependencies_symbolic("Make Stone Sword", n=1)
    print(f"Symbolic execution order: {symbolic_order}")
