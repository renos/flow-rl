"""
Generate Craftax knowledge base from tutorial markdown file.
This replaces the old action-focused knowledge base with a strategic progression-focused one.

Usage:
    python knowledge_base_processing/process_craftax_kb_from_tutorial.py
"""

import os
import sys
import json
from functools import partial
import agentkit
from agentkit import Graph, SimpleDBNode
from flowrl.llm.compose_prompts import ComposeBasePrompt
from flowrl.llm.craftax.after_queries import *
import agentkit.llm_api

try:
    import llm_api
    get_query = llm_api.get_query
    llm_name = "yintat-gpt-4o"
except:
    get_query = agentkit.llm_api.get_query
    llm_name = "gpt-5"


class SaveKnowledgeBaseAfterQuery(aq.JsonAfterQuery):
    """Parse JSON response and save to database."""
    def __init__(self):
        super().__init__()
        self.type = dict

    def post_process(self):
        self.node.db["knowledgebase"] = self.parse_json()[-1]


def generate_graph(db=None):
    """Build agentkit graph for processing tutorial."""
    LLM_API_FUNCTION_GPT4 = partial(get_query(llm_name), max_gen=16384)

    def build_graph(prompts, database={}):
        graph = Graph()
        edge_list = []
        order_list = []

        for _, node_info in prompts.items():
            key = node_prompt = node_info["prompt"]
            node = SimpleDBNode(
                key,
                node_prompt,
                graph,
                node_info["query"],
                node_info["compose"],
                database,
                after_query=(
                    node_info["after_query"]
                    if "after_query" in node_info.keys()
                    else None
                ),
                verbose=True,
            )
            graph.add_node(node)

            for dependency in node_info["dep"]:
                dependency_name = prompts[dependency]["prompt"]
                edge_list.append((dependency_name, key))

            if "after" in node_info.keys():
                for dependency in node_info["after"]:
                    dependency_name = prompts[dependency]["prompt"]
                    order_list.append((dependency_name, key))

        for edge in edge_list:
            graph.add_edge(*edge)
        for order in order_list:
            graph.add_order(*order)

        return graph

    prompts = return_prompts(LLM_API_FUNCTION_GPT4)
    graph = build_graph(prompts, db)
    return graph


def return_prompts(LLM_API_FUNCTION_GPT4):
    """Define prompts for knowledge base generation."""
    kb_prompts = {}

    kb_prompts["convert_tutorial"] = {
        "prompt": """
Convert the following Craftax game tutorial into a structured JSON knowledge base.
The knowledge base will be used by an RL agent to decide which skills to learn next.

# Symbolic State Space

The agent tracks progress through these state types:

**Inventory items** (type: "inventory"):
- wood, stone, coal, iron, diamond, sapling, arrows, torches, ruby, sapphire, books
- pickaxe (tier: 0=none, 1=wood, 2=stone, 3=iron, 4=diamond)
- sword (tier: 0=none, 1=wood, 2=stone, 3=iron, 4=diamond)
- bow (0=none, 1=has bow)
- armour (array of 4 pieces, each 0-2 representing tier)
- potions (array of 6 colors)

**Achievements** (type: "achievement", prefix: "achievement:"):
COLLECT_WOOD, PLACE_TABLE, EAT_COW, COLLECT_SAPLING, COLLECT_DRINK, MAKE_WOOD_PICKAXE, MAKE_WOOD_SWORD,
PLACE_PLANT, DEFEAT_ZOMBIE, COLLECT_STONE, PLACE_STONE, EAT_PLANT, DEFEAT_SKELETON, MAKE_STONE_PICKAXE,
MAKE_STONE_SWORD, WAKE_UP, PLACE_FURNACE, COLLECT_COAL, COLLECT_IRON, COLLECT_DIAMOND, MAKE_IRON_PICKAXE,
MAKE_IRON_SWORD, MAKE_ARROW, MAKE_TORCH, PLACE_TORCH, MAKE_DIAMOND_PICKAXE, MAKE_DIAMOND_SWORD,
MAKE_IRON_ARMOUR, MAKE_DIAMOND_ARMOUR, ENTER_GNOMISH_MINES, ENTER_DUNGEON, ENTER_SEWERS, ENTER_VAULT,
ENTER_TROLL_MINES, ENTER_FIRE_REALM, ENTER_ICE_REALM, ENTER_GRAVEYARD, DEFEAT_GNOME_WARRIOR, DEFEAT_GNOME_ARCHER,
DEFEAT_ORC_SOLIDER, DEFEAT_ORC_MAGE, DEFEAT_LIZARD, DEFEAT_KOBOLD, DEFEAT_KNIGHT, DEFEAT_ARCHER, DEFEAT_TROLL,
DEFEAT_DEEP_THING, DEFEAT_PIGMAN, DEFEAT_FIRE_ELEMENTAL, DEFEAT_FROST_TROLL, DEFEAT_ICE_ELEMENTAL,
DAMAGE_NECROMANCER, DEFEAT_NECROMANCER, EAT_BAT, EAT_SNAIL, FIND_BOW, FIRE_BOW, LEARN_FIREBALL, CAST_FIREBALL,
LEARN_ICEBALL, CAST_ICEBALL, COLLECT_SAPPHIRE, COLLECT_RUBY, OPEN_CHEST, DRINK_POTION, ENCHANT_SWORD, ENCHANT_ARMOUR

**Levels/Stats** (type: "level", prefix: "level:"):
- level:player_level (0-8 for floors)
- level:player_dexterity, level:player_strength, level:player_intelligence (1-5)

**Monsters killed per floor** (type: "stat", prefix: "stat:monsters_killed:<floor_index>"):
- stat:monsters_killed:0, stat:monsters_killed:1, etc.
- Used to track if ladder can open (8 kills required)

**Ephemeral** (type: "ephemeral"):
- Transient capabilities not directly observable in inventory (e.g., "near crafting table")

# Gain Schema

Each skill has gains expressed as:
```json
"gain_key": {
  "type": "inventory|achievement|level|stat|ephemeral",
  "expression": "lambda n: ...",
  "description": "optional"
}
```

Expression patterns:
- **Count-based** (wood, stone, arrows): `"lambda n: n"` (gain n items)
- **Tier-based** (pickaxe, sword): `"lambda n: 2"` for stone tier, `"lambda n: 3"` for iron tier
- **Achievements**: `"lambda n: 1"` (binary)
- **Levels**: `"lambda n: target_value"` (e.g., `"lambda n: 1"` for reaching floor 1)
- **Monster kills**: `"lambda n: n"` (additive)

# Tutorial

```markdown
$db.tutorial_text$
```

# Task

Generate a JSON knowledge base with:

1. **tutorial_context**: Preserve all tutorial information (mechanics, floors, tips)

2. **skills**: List of all skills necessary to progress through the game.

   Consider each floor (0-8) and identify what skills are needed to progress to the next floor.
   Each skill has ONE primary goal. Do not combine multiple distinct goals into a single skill.

   Return as a flat list, not grouped by floor.

   For each skill, provide:
   - skill_name: brief descriptive name
   - description: what the skill does
   - floor_availability: list of floor numbers derived from tutorial information
     - For resource gathering: which floors does the tutorial say this resource exists on?
     - For actions with location requirements: which floors can this action be performed on?
     - Empty list [] only if the tutorial indicates no floor restrictions
   - gain: what inventory/achievements/levels the skill provides
     - Use proper gain keys (inventory items, "achievement:X", "level:X", "stat:monsters_killed:X")
     - expression: "lambda n: n" for counts, "lambda n: value" for tiers/fixed values

Return ONLY valid JSON.
        """,
        "dep": [],
        "compose": ComposeBasePrompt(),
        "query": LLM_API_FUNCTION_GPT4,
        "after_query": SaveKnowledgeBaseAfterQuery(),
    }

    for _, v in kb_prompts.items():
        v["prompt"] = v["prompt"].strip()

    return kb_prompts


def main():
    """Main function to generate knowledge base from tutorial."""

    # Get project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Paths
    tutorial_path = os.path.join(project_root, "resources", "tutorial.md")
    output_json_path = os.path.join(project_root, "resources", "craftax_knowledgebase.json")
    output_txt_path = os.path.join(project_root, "resources", "craftax_knowledgebase.txt")

    print(f"Loading tutorial from: {tutorial_path}")

    if not os.path.exists(tutorial_path):
        print(f"ERROR: Tutorial file not found at {tutorial_path}")
        sys.exit(1)

    # Load tutorial
    with open(tutorial_path, 'r') as f:
        tutorial_text = f.read()
    print(f"Loaded tutorial ({len(tutorial_text)} characters)")

    # Setup database with tutorial text
    db = {"tutorial_text": tutorial_text}

    # Generate graph and evaluate
    print("Generating structured knowledge base from tutorial using LLM...")
    graph = generate_graph(db)
    answer = graph.evaluate()

    # Save as JSON
    print(f"Saving knowledge base to: {output_json_path}")
    with open(output_json_path, 'w') as f:
        json.dump(db["knowledgebase"], f, indent=2)

    # Save as pretty text
    print(f"Saving human-readable version to: {output_txt_path}")
    with open(output_txt_path, 'w') as f:
        f.write("CRAFTAX KNOWLEDGE BASE (Generated from Tutorial)\n")
        f.write("=" * 80 + "\n\n")

        def write_dict(d, indent=0):
            """Recursively write dictionary as formatted text."""
            prefix = "  " * indent
            for key, value in d.items():
                if isinstance(value, dict):
                    f.write(f"{prefix}{key}:\n")
                    write_dict(value, indent + 1)
                elif isinstance(value, list):
                    f.write(f"{prefix}{key}:\n")
                    for item in value:
                        if isinstance(item, dict):
                            write_dict(item, indent + 1)
                            f.write("\n")
                        else:
                            f.write(f"{prefix}  - {item}\n")
                else:
                    f.write(f"{prefix}{key}: {value}\n")

        write_dict(db["knowledgebase"])

    print("\nKnowledge base generation complete!")
    print(f"JSON: {output_json_path}")
    print(f"Text: {output_txt_path}")

    # Print summary
    print("\nKnowledge Base Summary:")
    for section, content in db["knowledgebase"].items():
        if isinstance(content, dict):
            print(f"  - {section}: {len(content)} entries")
        elif isinstance(content, list):
            print(f"  - {section}: {len(content)} items")
        else:
            print(f"  - {section}: present")


if __name__ == "__main__":
    main()
