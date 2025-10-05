#!/usr/bin/env python3
"""
Extract trajectory data from .pbz2 file and save in a clean format
that doesn't depend on the original Env classes.
"""

import pickle
import bz2
import os
import numpy as np
import jax.numpy as jnp
from pathlib import Path


def load_compressed_pickle(file_path):
    """Load a compressed pickle file (.pbz2)"""
    with bz2.BZ2File(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pickle(data, file_path):
    """Save data as pickle file (.pkl)"""
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def save_compressed_pickle(data, file_path):
    """Save data as compressed pickle file (.pbz2)"""
    with bz2.BZ2File(file_path, 'wb') as f:
        pickle.dump(data, f)


def extract_env_state_data(env_state):
    """Extract raw data from an EnvState object without class dependencies"""
    
    # Convert JAX arrays to regular numpy arrays for better compatibility
    def jax_to_numpy(x):
        if hasattr(x, '__array__'):
            return np.array(x)
        return x
    
    extracted_data = {
        # Map data
        'map': jax_to_numpy(env_state.map),  # (num_floors, height, width)
        'item_map': jax_to_numpy(env_state.item_map),
        'mob_map': jax_to_numpy(env_state.mob_map),
        'light_map': jax_to_numpy(env_state.light_map),
        
        # Level structure
        'down_ladders': jax_to_numpy(env_state.down_ladders),
        'up_ladders': jax_to_numpy(env_state.up_ladders),
        'chests_opened': jax_to_numpy(env_state.chests_opened),
        'monsters_killed': jax_to_numpy(env_state.monsters_killed),
        
        # Player state
        'player_position': jax_to_numpy(env_state.player_position),  # [x, y]
        'player_level': jax_to_numpy(env_state.player_level),  # floor number
        'player_direction': jax_to_numpy(env_state.player_direction),
        
        # Player intrinsics
        'player_health': jax_to_numpy(env_state.player_health),
        'player_food': jax_to_numpy(env_state.player_food),
        'player_drink': jax_to_numpy(env_state.player_drink),
        'player_energy': jax_to_numpy(env_state.player_energy),
        'player_mana': jax_to_numpy(env_state.player_mana),
        'is_sleeping': jax_to_numpy(env_state.is_sleeping),
        'is_resting': jax_to_numpy(env_state.is_resting),
        
        # Player secondary stats
        'player_recover': jax_to_numpy(env_state.player_recover),
        'player_hunger': jax_to_numpy(env_state.player_hunger),
        'player_thirst': jax_to_numpy(env_state.player_thirst),
        'player_fatigue': jax_to_numpy(env_state.player_fatigue),
        'player_recover_mana': jax_to_numpy(env_state.player_recover_mana),
        
        # Player attributes
        'player_xp': jax_to_numpy(env_state.player_xp),
        'player_dexterity': jax_to_numpy(env_state.player_dexterity),
        'player_strength': jax_to_numpy(env_state.player_strength),
        'player_intelligence': jax_to_numpy(env_state.player_intelligence),
        
        # Inventory (extract individual fields to avoid class dependency)
        'inventory': {
            'wood': jax_to_numpy(env_state.inventory.wood),
            'stone': jax_to_numpy(env_state.inventory.stone),
            'coal': jax_to_numpy(env_state.inventory.coal),
            'iron': jax_to_numpy(env_state.inventory.iron),
            'diamond': jax_to_numpy(env_state.inventory.diamond),
            'sapling': jax_to_numpy(env_state.inventory.sapling),
            'pickaxe': jax_to_numpy(env_state.inventory.pickaxe),
            'sword': jax_to_numpy(env_state.inventory.sword),
            'bow': jax_to_numpy(env_state.inventory.bow),
            'arrows': jax_to_numpy(env_state.inventory.arrows),
            'armour': jax_to_numpy(env_state.inventory.armour),
            'torches': jax_to_numpy(env_state.inventory.torches),
            'ruby': jax_to_numpy(env_state.inventory.ruby),
            'sapphire': jax_to_numpy(env_state.inventory.sapphire),
            'potions': jax_to_numpy(env_state.inventory.potions),
            'books': jax_to_numpy(env_state.inventory.books),
        },
        
        # Mob data (simplified)
        'melee_mobs': {
            'position': jax_to_numpy(env_state.melee_mobs.position),
            'health': jax_to_numpy(env_state.melee_mobs.health),
            'mask': jax_to_numpy(env_state.melee_mobs.mask),
            'attack_cooldown': jax_to_numpy(env_state.melee_mobs.attack_cooldown),
            'type_id': jax_to_numpy(env_state.melee_mobs.type_id),
        },
        'passive_mobs': {
            'position': jax_to_numpy(env_state.passive_mobs.position),
            'health': jax_to_numpy(env_state.passive_mobs.health),
            'mask': jax_to_numpy(env_state.passive_mobs.mask),
            'attack_cooldown': jax_to_numpy(env_state.passive_mobs.attack_cooldown),
            'type_id': jax_to_numpy(env_state.passive_mobs.type_id),
        },
        'ranged_mobs': {
            'position': jax_to_numpy(env_state.ranged_mobs.position),
            'health': jax_to_numpy(env_state.ranged_mobs.health),
            'mask': jax_to_numpy(env_state.ranged_mobs.mask),
            'attack_cooldown': jax_to_numpy(env_state.ranged_mobs.attack_cooldown),
            'type_id': jax_to_numpy(env_state.ranged_mobs.type_id),
        },
        
        # Other game state
        'achievements': jax_to_numpy(env_state.achievements),
        'timestep': jax_to_numpy(env_state.timestep),
        'light_level': jax_to_numpy(env_state.light_level),
        
        # Simplified other fields
        'learned_spells': jax_to_numpy(env_state.learned_spells),
        'sword_enchantment': jax_to_numpy(env_state.sword_enchantment),
        'bow_enchantment': jax_to_numpy(env_state.bow_enchantment),
        'armour_enchantments': jax_to_numpy(env_state.armour_enchantments),
        'boss_progress': jax_to_numpy(env_state.boss_progress),
    }
    
    return extracted_data


def extract_trajectory(input_file, output_file, use_compression=False):
    """
    Extract trajectory data and save in clean format.
    
    Args:
        input_file: Input .pbz2 file
        output_file: Output file (.pkl or .pbz2)  
        use_compression: If True, save as .pbz2, else save as .pkl
    """
    print(f"Loading trajectory from: {input_file}")
    
    # Load original trajectory
    trajectory_data = load_compressed_pickle(input_file)
    
    print(f"Original trajectory keys: {trajectory_data.keys()}")
    print(f"Number of timesteps: {len(trajectory_data['state'])}")
    
    # Extract clean data
    print("Extracting state data...")
    clean_states = []
    
    for i, state in enumerate(trajectory_data['state']):
        clean_state = extract_env_state_data(state)
        clean_states.append(clean_state)
        
        if i % 1000 == 0:
            print(f"  Processed {i}/{len(trajectory_data['state'])} states")
    
    # Create clean trajectory data
    clean_trajectory = {
        'states': clean_states,  # List of clean state dictionaries
        'actions': [int(a) for a in trajectory_data['action']],  # Convert to regular ints
        'rewards': [float(r) for r in trajectory_data['reward']],  # Convert to regular floats
        'dones': [bool(d) for d in trajectory_data['done']],  # Convert to regular bools
        'metadata': {
            'num_timesteps': len(clean_states),
            'map_shape': clean_states[0]['map'].shape,
            'player_start_pos': clean_states[0]['player_position'].tolist(),
            'player_start_level': int(clean_states[0]['player_level']),
            'total_reward': sum(float(r) for r in trajectory_data['reward']),
        }
    }
    
    print(f"Saving clean trajectory to: {output_file}")
    if use_compression:
        save_compressed_pickle(clean_trajectory, output_file)
    else:
        save_pickle(clean_trajectory, output_file)
    
    print(f"✅ Extraction complete!")
    print(f"  Original file size: {os.path.getsize(input_file) / (1024*1024):.1f} MB")
    print(f"  Clean file size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
    print(f"  Timesteps: {clean_trajectory['metadata']['num_timesteps']}")
    print(f"  Total reward: {clean_trajectory['metadata']['total_reward']:.4f}")
    print(f"  Map shape: {clean_trajectory['metadata']['map_shape']}")
    
    return clean_trajectory


def test_load_clean_trajectory(clean_file):
    """
    Test loading and accessing the clean trajectory data.
    """
    print(f"Testing clean trajectory loading from: {clean_file}")
    
    clean_data = load_compressed_pickle(clean_file)
    
    print(f"Clean trajectory keys: {clean_data.keys()}")
    print(f"Metadata: {clean_data['metadata']}")
    
    # Test accessing first state
    first_state = clean_data['states'][0]
    print(f"First state keys: {list(first_state.keys())}")
    print(f"Player position: {first_state['player_position']}")
    print(f"Player level: {first_state['player_level']}")
    print(f"Inventory wood: {first_state['inventory']['wood']}")
    
    # Test accessing some actions/rewards
    print(f"First 10 actions: {clean_data['actions'][:10]}")
    print(f"First 10 rewards: {clean_data['rewards'][:10]}")
    
    return clean_data


def main():
    """Main extraction function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract clean trajectory data")
    parser.add_argument("--input_file", help="Input .pbz2 trajectory file")
    parser.add_argument("-o", "--output", help="Output file path", default=None)
    parser.add_argument("--compress", action="store_true", help="Use compression (.pbz2)")
    
    args = parser.parse_args()
    
    # Check if input exists
    if not os.path.exists(args.input_file):
        print(f"Input file not found: {args.input_file}")
        return
    
    # Generate output filename
    if args.output is None:
        input_path = Path(args.input_file)
        if args.compress:
            output_file = input_path.parent / f"{input_path.stem}_clean.pbz2"
        else:
            output_file = input_path.parent / f"{input_path.stem}_clean.pkl"
    else:
        output_file = args.output
    
    # Extract trajectory
    clean_trajectory = extract_trajectory(args.input_file, str(output_file), args.compress)
    
    # Test loading the clean file
    print("\n" + "="*50)
    print("TESTING CLEAN FILE")
    print("="*50)
    
    if args.compress:
        test_clean_data = test_load_clean_trajectory(str(output_file))
    else:
        # Load and test .pkl file
        with open(output_file, 'rb') as f:
            test_clean_data = pickle.load(f)
        print(f"Clean trajectory keys: {test_clean_data.keys()}")
        print(f"Metadata: {test_clean_data['metadata']}")
    
    print(f"\n✅ Success! Clean trajectory saved to: {output_file}")
    if args.compress:
        print("This file can now be loaded with modern code without class dependencies.")
    else:
        print("This .pkl file will load much faster than .pbz2 format!")


if __name__ == "__main__":
    # First load and explore the original data structure
    main()
    # trajectory_file = "/home/renos/flow-rl/resources/people/run1.pbz2"
    
    # if os.path.exists(trajectory_file):
    #     print("Loading original trajectory for exploration...")
    #     trajectory_data = load_compressed_pickle(trajectory_file)
        
    #     print("Original data structure:")
    #     print(f"  Keys: {list(trajectory_data.keys())}")
    #     print(f"  Timesteps: {len(trajectory_data['state'])}")
    #     print(f"  First state type: {type(trajectory_data['state'][0])}")
    #     print(f"  First action: {trajectory_data['action'][0]}")
    #     print(f"  First reward: {trajectory_data['reward'][0]}")
        
    #     print("\nReady to extract clean data...")
    #     print("Available in debugger:")
    #     print("  trajectory_data - original data")
    #     print("  To extract: run main()")
        
    #     breakpoint()
        
    #     # Uncomment to extract clean data
    #     # main()
    # else:
    #     print(f"Trajectory file not found: {trajectory_file}")