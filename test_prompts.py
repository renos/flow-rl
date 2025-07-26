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
    
    def resolve_skill_dependencies(self, skill_name, n=1, max_inventory_capacity=9):
        """
        Resolve dependencies for a skill and return the execution order.
        
        Args:
            skill_name: Name of the skill to resolve dependencies for
            n: Number of times to apply this skill (default: 1)
            max_inventory_capacity: Maximum inventory capacity per item (default: 9)
            
        Returns:
            List of nodes/skills in execution order to fulfill the target skill
        """
        if skill_name not in self.db["skills"]:
            available_skills = list(self.db["skills"].keys())
            raise ValueError(f"Skill '{skill_name}' not found. Available skills: {available_skills}")
        
        # Create resolver with current skills and inventory capacity
        resolver = SkillDependencyResolver(self.db["skills"], max_inventory_capacity)
        
        print(f"Resolving dependencies for '{skill_name}' with n={n}, max_inventory={max_inventory_capacity}")
        execution_order = resolver.resolve_dependencies(skill_name, n)
        
        return execution_order
        
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