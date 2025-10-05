#!/usr/bin/env python3
"""
Script to load and explore trajectory data from a .pbz2 file.
Usage: python explore_trajectory.py
"""

import pickle
import bz2
import os
from pathlib import Path


def load_compressed_pickle(file_path):
    """Load a compressed pickle file (.pbz2)"""
    with bz2.BZ2File(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def explore_trajectory():
    """Load trajectory data and explore its structure"""
    
    # Path to the trajectory file
    trajectory_file = "/home/renos/flow-rl/resources/people/run1.pbz2"
    
    # Check if file exists
    if not os.path.exists(trajectory_file):
        print(f"File not found: {trajectory_file}")
        print("Available files in resources/people/:")
        people_dir = Path("/Users/renos/Documents/flow-rl/resources/people")
        if people_dir.exists():
            for file in people_dir.iterdir():
                print(f"  {file.name}")
        return
    
    print(f"Loading trajectory from: {trajectory_file}")
    
    try:
        # Load the trajectory data
        trajectory_data = load_compressed_pickle(trajectory_file)
        
        print(f"Successfully loaded trajectory data!")
        print(f"Type: {type(trajectory_data)}")
        
        # Basic exploration
        if hasattr(trajectory_data, '__len__'):
            print(f"Length: {len(trajectory_data)}")
        
        if isinstance(trajectory_data, dict):
            print("Keys:", list(trajectory_data.keys()))
        elif isinstance(trajectory_data, (list, tuple)):
            print(f"First few elements types: {[type(x) for x in trajectory_data[:5]]}")
        
        # Drop into breakpoint for manual exploration
        print("\nDropping into breakpoint for exploration...")
        print("Available variables:")
        print("  trajectory_data - the loaded data")
        print("  trajectory_file - path to the file")
        print("\nTry exploring with:")
        print("  type(trajectory_data)")
        print("  len(trajectory_data) if applicable")
        print("  trajectory_data.keys() if dict")
        print("  trajectory_data[0] if list/array")
        
        breakpoint()
        
    except Exception as e:
        print(f"Error loading file: {e}")
        print("This might not be a standard pickle file.")
        
        # Try alternative loading methods
        print("Trying alternative loading methods...")
        
        try:
            # Try loading as regular pickle
            with open(trajectory_file, 'rb') as f:
                trajectory_data = pickle.load(f)
            print("Loaded as regular pickle file")
            breakpoint()
        except:
            pass
        
        try:
            # Try loading with different compression
            import gzip
            with gzip.open(trajectory_file, 'rb') as f:
                trajectory_data = pickle.load(f)
            print("Loaded as gzip compressed file")
            breakpoint()
        except:
            pass
        
        print("Could not load the file with standard methods.")
        print("The file might use a custom format or be corrupted.")


if __name__ == "__main__":
    explore_trajectory()