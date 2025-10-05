# Knowledge Base Processing

This folder contains scripts for generating and verifying knowledge bases for both Craftax Classic and full Craftax environments.

## Files

### For Craftax Classic
- `process_craftax_classic_kb.py` - Generates knowledge base for Craftax Classic
- `verify_craftax_classic_kb.py` - Verifies and annotates the generated knowledge base

### For Full Craftax  
- `process_craftax_kb.py` - Generates knowledge base for full Craftax environment
- `verify_craftax_kb.py` - Verifies and annotates the generated knowledge base

## Usage

### 1. Generate Knowledge Base

For Craftax Classic:
```bash
cd /path/to/flow-rl
python knowledge_base_processing/process_craftax_classic_kb.py
```

For full Craftax:
```bash
cd /path/to/flow-rl
python knowledge_base_processing/process_craftax_kb.py
```

This will create a knowledge base JSON file in the `resources/` directory.

### 2. Verify Knowledge Base

For Craftax Classic:
```bash
cd /path/to/flow-rl
python knowledge_base_processing/verify_craftax_classic_kb.py
```

For full Craftax:
```bash
cd /path/to/flow-rl
python knowledge_base_processing/verify_craftax_kb.py
```

This will create a verified knowledge base with each requirement marked as either:
- `VERIFIED:` - Can be confirmed from game mechanics
- `ASSUMPTION:` - Educated guess that needs validation

## Prerequisites

Make sure the flowrl package is installed:
```bash
pip install -e .
```

## Output Files

The scripts will generate files in the `resources/` directory:
- `craftax_classic_knowledgebase.json` - Raw knowledge base for Classic
- `craftax_classic_knowledgebase_verified.json` - Verified knowledge base for Classic
- `craftax_knowledgebase.json` - Raw knowledge base for full Craftax  
- `craftax_knowledgebase_verified.json` - Verified knowledge base for full Craftax

## Key Differences

### Craftax Classic
- Simpler inventory (12 items with max 9 each)
- Basic block types (17 types)
- Limited actions (17 actions)
- 22 achievements

### Full Craftax
- Extended inventory (16+ items including arrays for armour/potions)
- Many more block types (37 types including dungeons, gems, etc.)
- Extended actions (43 actions including spells, enchanting, etc.) 
- 67 achievements including advanced gameplay