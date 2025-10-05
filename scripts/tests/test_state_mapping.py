#!/usr/bin/env python3
"""
Test script to verify state-to-network mapping is correct.
"""

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import sys

# Add flowrl to path
sys.path.append(str(Path(__file__).parent))

def test_state_mapping_with_real_module(module_path):
    """
    Test the state-to-network mapping using a real generated module.
    
    Args:
        module_path: Path to a generated Python module (e.g., "exp/bottom_up/14/14.py")
    """
    print(f"Testing state mapping with module: {module_path}")
    print("=" * 60)
    
    # Import the module dynamically
    import importlib.util
    import os
    
    if not os.path.exists(module_path):
        print(f"ERROR: Module file not found: {module_path}")
        return
        
    module_name = "test_module"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Create the environment to get mapping info
    from craftax.craftax_env import make_craftax_flow_env_from_name
    
    env = make_craftax_flow_env_from_name(
        "Craftax-Classic-Symbolic-v1", 
        False,  # auto_reset = False (not use_optimistic_resets)
        module.__dict__
    )
    
    # Get mapping information
    heads, num_heads = env.heads_info
    task_to_skill_index = jnp.array(env.task_to_skill_index)
    num_tasks = len(task_to_skill_index)
    
    print(f"Environment info:")
    print(f"  num_heads (network heads): {num_heads}")
    print(f"  num_tasks (player states): {num_tasks}")
    print(f"  task_to_skill_index: {task_to_skill_index}")
    print(f"  heads: {list(heads)}")
    print()
    
    # Define the mapping function (same as in ppo_flow.py)
    def map_player_state_to_skill(player_state):
        player_state_one_hot = jnp.eye(num_tasks)[player_state]
        player_skill = (player_state_one_hot @ task_to_skill_index).astype(jnp.int32)
        return player_skill
    
    # Test the mapping for each player state
    print("State-to-Network Mapping:")
    print("State | Mapped->Net | Function Says | Status")
    print("-" * 45)
    
    for state in range(num_tasks):
        network_head = int(map_player_state_to_skill(state))
        
        # Try to find the corresponding task function
        task_network_func_name = f"task_{network_head}_network_number"
        task_done_func_name = f"task_{network_head}_is_done" 
        task_reward_func_name = f"task_{network_head}_reward"
        
        has_network_func = hasattr(module, task_network_func_name)
        has_done_func = hasattr(module, task_done_func_name)
        has_reward_func = hasattr(module, task_reward_func_name)
        
        if has_network_func:
            # Get the actual network number from the function
            network_func = getattr(module, task_network_func_name)
            try:
                actual_network = network_func()
                if actual_network == network_head:
                    status = "✓ MATCH"
                else:
                    status = "✗ MISMATCH"
                
                print(f"  {state:2d}  |      {network_head:2d}     |      {actual_network:2d}       | {status}")
                
            except Exception as e:
                print(f"  {state:2d}  |      {network_head:2d}     |    ERROR    | ✗ CALL FAILED")
        else:
            print(f"  {state:2d}  |      {network_head:2d}     |   MISSING   | ✗ NO FUNCTION")
    
    print()
    print("Verification Summary:")
    print("=" * 30)
    
    # Check for consistency
    all_correct = True
    for state in range(num_tasks):
        network_head = int(map_player_state_to_skill(state))
        task_network_func_name = f"task_{network_head}_network_number"
        
        if hasattr(module, task_network_func_name):
            network_func = getattr(module, task_network_func_name)
            try:
                expected_network = network_func()
                if expected_network != network_head:
                    print(f"INCONSISTENCY: State {state} maps to head {network_head}, but function returns {expected_network}")
                    all_correct = False
            except Exception as e:
                print(f"ERROR: Cannot call {task_network_func_name}: {e}")
                all_correct = False
        else:
            print(f"MISSING: No function {task_network_func_name} for state {state}")
            all_correct = False
    
    if all_correct:
        print("✅ All mappings are consistent!")
    else:
        print("❌ Found mapping inconsistencies!")
        
    return all_correct

def test_mapping_logic():
    """Test the mapping logic with synthetic data."""
    print("Testing mapping logic with synthetic data:")
    print("=" * 50)
    
    # Example mapping: states 0,1,2 -> networks 0,1,2
    task_to_skill_index = jnp.array([0, 1, 2])
    num_tasks = len(task_to_skill_index)
    
    def map_player_state_to_skill(player_state):
        player_state_one_hot = jnp.eye(num_tasks)[player_state]
        player_skill = (player_state_one_hot @ task_to_skill_index).astype(jnp.int32)
        return player_skill
    
    print("task_to_skill_index:", task_to_skill_index)
    for state in range(num_tasks):
        network = map_player_state_to_skill(state)
        print(f"State {state} -> Network {network}")
    
    print()
    
    # Test batch mapping
    batch_states = jnp.array([0, 1, 2, 1, 0])
    batch_networks = jax.vmap(map_player_state_to_skill)(batch_states)
    print("Batch test:")
    print(f"States: {batch_states}")
    print(f"Networks: {batch_networks}")
    print()

if __name__ == "__main__":
    # Test basic logic first
    test_mapping_logic()
    
    # Test with real module if path provided
    if len(sys.argv) > 1:
        module_path = sys.argv[1]
        test_state_mapping_with_real_module(module_path)
    else:
        print("To test with a real module, run:")
        print("python test_state_mapping.py path/to/your/module.py")
        print("\nExample:")
        print("python test_state_mapping.py exp/bottom_up/14/14.py")