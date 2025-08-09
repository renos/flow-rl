#!/usr/bin/env python3
"""
Test script to load and explore the knowledge base
"""

import json
from pathlib import Path 

def load_knowledge_base():
    """Load the verified Craftax knowledge base file"""
    kb_file = "resources/craftax_classic_knowledgebase_verified.json"
    
    kb_path = Path(kb_file)
    if kb_path.exists():
        with open(kb_path, 'r') as f:
            kb_data = json.load(f)
        print(f"Loaded {kb_file}: {len(kb_data)} entries")
        return kb_data
    else:
        print(f"Error: {kb_file} not found")
        return None

def explore_knowledge_base(kb_data, name):
    """Explore structure of knowledge base"""
    print(f"\n=== Exploring {name} ===")
    
    if isinstance(kb_data, dict):
        print(f"Top-level keys: {list(kb_data.keys())}")
        
        for key, value in kb_data.items():
            print(f"\n{key}:")
            if isinstance(value, dict):
                print(f"  Type: dict with {len(value)} entries")
                print(f"  Sample keys: {list(value.keys())[:5]}")
            elif isinstance(value, list):
                print(f"  Type: list with {len(value)} items")
                if value and isinstance(value[0], dict):
                    print(f"  Sample item keys: {list(value[0].keys())}")
            else:
                print(f"  Type: {type(value).__name__}")
                print(f"  Value: {str(value)[:100]}...")

def show_sample_entries(kb_data, name, max_entries=3):
    """Show sample entries from knowledge base"""
    print(f"\n=== Sample entries from {name} ===")
    
    if isinstance(kb_data, dict):
        for i, (key, value) in enumerate(kb_data.items()):
            if i >= max_entries:
                break
            print(f"\nEntry: {key}")
            print(f"Content: {json.dumps(value, indent=2)}")

def main():
    print("Knowledge Base Test Script")
    print("=" * 40)
    
    # Load knowledge base
    kb_data = load_knowledge_base()
    breakpoint()
    
    if kb_data:
        explore_knowledge_base(kb_data, "verified knowledge base")
        show_sample_entries(kb_data, "verified knowledge base")

if __name__ == "__main__":
    main()