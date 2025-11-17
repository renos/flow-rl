"""
Phase 3: Generate code and reward functions for skills in the knowledge base.

This reads craftax_knowledgebase_verified.json (output from add_prerequisites_craftax_kb.py)
and generates the task_is_done, task_reward, and task_network_number functions for each skill.

Output: craftax_knowledgebase_verified_with_code.json

Usage:
    python knowledge_base_processing/generate_code_craftax_kb.py
    python knowledge_base_processing/generate_code_craftax_kb.py --skill-index 0
    python knowledge_base_processing/generate_code_craftax_kb.py --skill-index 0 --save-single
"""

import os
import sys
import json
import argparse
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


class SaveDenseRewardAfterQuery(aq.JsonAfterQuery):
    """Parse JSON response and save dense reward info."""
    def __init__(self, skill_index):
        super().__init__()
        self.type = dict
        self.skill_index = skill_index
        self.required_keys = [
            "sparse_reward_only_function",
            "dense_reward_function",
        ]

    def post_process(self):
        print(f"\n{'='*80}")
        print(f"DENSE REWARD RESPONSE for skill {self.skill_index}:")
        print(f"{'='*80}")
        print(self.node.result)
        print(f"{'='*80}\n")

        parsed_answer = self.parse_json()[-1]

        # Store directly in skill-specific db key for template access
        self.node.db[f"reward_{self.skill_index}"] = parsed_answer["sparse_reward_only_function"]
        self.node.db[f"dense_reward_{self.skill_index}"] = parsed_answer["dense_reward_function"]


class SaveCodeAfterQuery(aq.BaseAfterQuery):
    """Save raw code response."""
    def __init__(self, skill_index):
        super().__init__()
        self.skill_index = skill_index

    def post_process(self):
        print(f"\n{'='*80}")
        print(f"CODE GENERATION RESPONSE for skill {self.skill_index}:")
        print(f"{'='*80}")
        print(self.node.result)
        print(f"{'='*80}\n")

        # Save the raw response in a tracking dict
        if "generated_code" not in self.node.db:
            self.node.db["generated_code"] = {}

        self.node.db["generated_code"][self.skill_index] = self.node.result


def generate_graph(db=None, skill_index=None):
    """Build agentkit graph for code generation."""
    LLM_API_FUNCTION_GPT4 = partial(get_query(llm_name), max_gen=8192)

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

    prompts = return_prompts(LLM_API_FUNCTION_GPT4, db, skill_index=skill_index)
    graph = build_graph(prompts, db)
    return graph


def return_prompts(LLM_API_FUNCTION_GPT4, db, skill_index=None):
    """Define prompts for generating code for each skill."""
    code_gen_prompts = {}

    skills_list = db.get("skills", [])

    # If skill_index is specified, only process that one skill
    if skill_index is not None:
        indices_to_process = [skill_index]
    else:
        indices_to_process = range(len(skills_list))

    # Create prompts for each skill
    for i in indices_to_process:
        skill = skills_list[i]
        skill_name = skill.get("skill_name", f"skill_{i}")

        # Prepare current skill context
        db[f"current_{i}"] = {
            "skill_name": skill_name,
            "skill_with_consumption": json.dumps(skill, indent=2),
            "num_skills": len(skills_list),
        }

        # Prompt 1: Dense reward reasoning
        code_gen_prompts[f"dense_reward_{i}"] = {
            "prompt": f"""
# All factors

Environment definitions:
```python
class BlockType(Enum):
    INVALID = 0
    OUT_OF_BOUNDS = 1
    GRASS = 2
    WATER = 3
    STONE = 4
    TREE = 5
    WOOD = 6
    PATH = 7
    COAL = 8
    IRON = 9
    DIAMOND = 10
    CRAFTING_TABLE = 11
    FURNACE = 12
    SAND = 13
    LAVA = 14
    PLANT = 15
    RIPE_PLANT = 16
    WALL = 17
    DARKNESS = 18
    WALL_MOSS = 19
    STALAGMITE = 20
    SAPPHIRE = 21
    RUBY = 22
    CHEST = 23
    FOUNTAIN = 24
    FIRE_GRASS = 25
    ICE_GRASS = 26
    GRAVEL = 27
    FIRE_TREE = 28
    ICE_SHRUB = 29
    ENCHANTMENT_TABLE_FIRE = 30
    ENCHANTMENT_TABLE_ICE = 31
    NECROMANCER = 32
    GRAVE = 33
    GRAVE2 = 34
    GRAVE3 = 35
    NECROMANCER_VULNERABLE = 36

# Inventory has much larger capacity than Classic version
@struct.dataclass
class Inventory:
    wood: int
    stone: int
    coal: int
    iron: int
    diamond: int
    sapling: int
    pickaxe: int  # Tier indicator: 0=none, 1=wood, 2=stone, 3=iron, 4=diamond
    sword: int  # Tier indicator: 0=none, 1=wood, 2=stone, 3=iron, 4=diamond
    bow: int
    arrows: int
    armour: jnp.ndarray  # Array of 4 armor pieces: [helmet, chestplate, leggings, boots], each 0-2 representing tier
    torches: int
    ruby: int
    sapphire: int
    potions: jnp.ndarray  # Array of 6 potion types: [red, green, blue, pink, cyan, yellow], each storing count
    books: int

class Achievement(Enum):
    COLLECT_WOOD = 0
    PLACE_TABLE = 1
    EAT_COW = 2
    COLLECT_SAPLING = 3
    COLLECT_DRINK = 4
    MAKE_WOOD_PICKAXE = 5
    MAKE_WOOD_SWORD = 6
    PLACE_PLANT = 7
    DEFEAT_ZOMBIE = 8
    COLLECT_STONE = 9
    PLACE_STONE = 10
    EAT_PLANT = 11
    DEFEAT_SKELETON = 12
    MAKE_STONE_PICKAXE = 13
    MAKE_STONE_SWORD = 14
    WAKE_UP = 15
    PLACE_FURNACE = 16
    COLLECT_COAL = 17
    COLLECT_IRON = 18
    COLLECT_DIAMOND = 19
    MAKE_IRON_PICKAXE = 20
    MAKE_IRON_SWORD = 21
    MAKE_ARROW = 22
    MAKE_TORCH = 23
    PLACE_TORCH = 24
    COLLECT_SAPPHIRE = 54
    COLLECT_RUBY = 59
    MAKE_DIAMOND_PICKAXE = 60
    MAKE_DIAMOND_SWORD = 25
    MAKE_IRON_ARMOUR = 26
    MAKE_DIAMOND_ARMOUR = 27
    ENTER_GNOMISH_MINES = 28
    ENTER_DUNGEON = 29
    ENTER_SEWERS = 30
    ENTER_VAULT = 31
    ENTER_TROLL_MINES = 32
    ENTER_FIRE_REALM = 33
    ENTER_ICE_REALM = 34
    ENTER_GRAVEYARD = 35
    DEFEAT_GNOME_WARRIOR = 36
    DEFEAT_GNOME_ARCHER = 37
    DEFEAT_ORC_SOLIDER = 38
    DEFEAT_ORC_MAGE = 39
    DEFEAT_LIZARD = 40
    DEFEAT_KOBOLD = 41
    DEFEAT_KNIGHT = 65
    DEFEAT_ARCHER = 66
    DEFEAT_TROLL = 42
    DEFEAT_DEEP_THING = 43
    DEFEAT_PIGMAN = 44
    DEFEAT_FIRE_ELEMENTAL = 45
    DEFEAT_FROST_TROLL = 46
    DEFEAT_ICE_ELEMENTAL = 47
    DAMAGE_NECROMANCER = 48
    DEFEAT_NECROMANCER = 49
    EAT_BAT = 50
    EAT_SNAIL = 51
    FIND_BOW = 52
    FIRE_BOW = 53
    LEARN_FIREBALL = 55
    CAST_FIREBALL = 56
    LEARN_ICEBALL = 57
    CAST_ICEBALL = 58
    OPEN_CHEST = 61
    DRINK_POTION = 62
    ENCHANT_SWORD = 63
    ENCHANT_ARMOUR = 64
```
The reward function is calculated independently at each timestep using these available factors:

- inventory_diff (Inventory): The change in the player's inventory between the current and previous timesteps. For count fields (wood, stone, coal, etc.), this is +1 per item gained, -1 per item used. For tier fields (pickaxe, sword), this is the tier increase (e.g., +1 when upgrading from wood to stone pickaxe).
- closest_bocks_changes (numpy.ndarray): The changes in distance to closest blocks of each type from the last timestep to the current timestep. Decreases in distance are positive. If an item has moves from being unseen to seen, the default will be 30-current_distance. E.g. if a table is placed in front of the player, the distance diff will be 29.
- player_level_diff (int): The change in dungeon level (e.g., +1 when descending from floor 0 to floor 1, -1 when ascending).
- monsters_killed_diff (jnp.ndarray): A vector of per-floor monster kill differences. Use monsters_killed_diff[floor_number] to get kills on a specific floor.
- player_intrinsics_diff (jnp.ndarray): The changes in current intrinsic values (health, food, drink, energy, mana) from the last timestep to the current timestep.


# Other Information
- This reward function is called independently at each timestep
- Each timestep's reward is calculated using only information from the current and previous timestep
- The reward at timestep t cannot access information from timestep t-2 or earlier
- The completion criteria is a separate function; do not worry about implementing it
- No state can be stored between timesteps - each reward calculation must be independent

# Skill
Given the following skill, design the reward function for the Skill `{skill_name}`
```
$db.current_{i}.skill_with_consumption$
```

# Design Approach
Return the raw factor value that measures the skill's gain. Requirements and consumption are enforced elsewhere.

1. Identify the factor measuring the gain
2. Sparse reward: return the raw factor (can be positive, negative, or zero)
3. Dense reward: optional raw factor that provides denser feedback (can be positive, negative, or zero) toward the gain, scaled by coefficient ≤ 0.01, or "NA"

```json
{{
"sparse_reward_only_function": # (str) "return <factor>"
"dense_reward_function": # (str) "return <coefficient> * <dense_factor>", or "NA"
}}
```
            """,
            "dep": [],
            "compose": ComposeBasePrompt(),
            "query": LLM_API_FUNCTION_GPT4,
            "after_query": SaveDenseRewardAfterQuery(i),
        }

        # Prompt 2: Code generation
        code_gen_prompts[f"code_gen_{i}"] = {
            "prompt": f"""
```python
class BlockType(Enum):
    INVALID = 0
    OUT_OF_BOUNDS = 1
    GRASS = 2
    WATER = 3
    STONE = 4
    TREE = 5
    WOOD = 6
    PATH = 7
    COAL = 8
    IRON = 9
    DIAMOND = 10
    CRAFTING_TABLE = 11
    FURNACE = 12
    SAND = 13
    LAVA = 14
    PLANT = 15
    RIPE_PLANT = 16
    WALL = 17
    DARKNESS = 18
    WALL_MOSS = 19
    STALAGMITE = 20
    SAPPHIRE = 21
    RUBY = 22
    CHEST = 23
    FOUNTAIN = 24
    FIRE_GRASS = 25
    ICE_GRASS = 26
    GRAVEL = 27
    FIRE_TREE = 28
    ICE_SHRUB = 29
    ENCHANTMENT_TABLE_FIRE = 30
    ENCHANTMENT_TABLE_ICE = 31
    NECROMANCER = 32
    GRAVE = 33
    GRAVE2 = 34
    GRAVE3 = 35
    NECROMANCER_VULNERABLE = 36

# Inventory has much higher capacities than Classic version
@struct.dataclass
class Inventory:
    wood: int
    stone: int
    coal: int
    iron: int
    diamond: int
    sapling: int
    pickaxe: int  # Tier indicator: 0=none, 1=wood, 2=stone, 3=iron, 4=diamond
    sword: int  # Tier indicator: 0=none, 1=wood, 2=stone, 3=iron, 4=diamond
    bow: int
    arrows: int
    armour: jnp.ndarray  # Array of 4 armor pieces: [helmet, chestplate, leggings, boots], each 0-2 representing tier
    torches: int
    ruby: int
    sapphire: int
    potions: jnp.ndarray  # Array of 6 potion types: [red, green, blue, pink, cyan, yellow], each storing count
    books: int

class Achievement(Enum):
    COLLECT_WOOD = 0
    PLACE_TABLE = 1
    EAT_COW = 2
    COLLECT_SAPLING = 3
    COLLECT_DRINK = 4
    MAKE_WOOD_PICKAXE = 5
    MAKE_WOOD_SWORD = 6
    PLACE_PLANT = 7
    DEFEAT_ZOMBIE = 8
    COLLECT_STONE = 9
    PLACE_STONE = 10
    EAT_PLANT = 11
    DEFEAT_SKELETON = 12
    MAKE_STONE_PICKAXE = 13
    MAKE_STONE_SWORD = 14
    WAKE_UP = 15
    PLACE_FURNACE = 16
    COLLECT_COAL = 17
    COLLECT_IRON = 18
    COLLECT_DIAMOND = 19
    MAKE_IRON_PICKAXE = 20
    MAKE_IRON_SWORD = 21
    MAKE_ARROW = 22
    MAKE_TORCH = 23
    PLACE_TORCH = 24
    COLLECT_SAPPHIRE = 54
    COLLECT_RUBY = 59
    MAKE_DIAMOND_PICKAXE = 60
    MAKE_DIAMOND_SWORD = 25
    MAKE_IRON_ARMOUR = 26
    MAKE_DIAMOND_ARMOUR = 27
    ENTER_GNOMISH_MINES = 28
    ENTER_DUNGEON = 29
    ENTER_SEWERS = 30
    ENTER_VAULT = 31
    ENTER_TROLL_MINES = 32
    ENTER_FIRE_REALM = 33
    ENTER_ICE_REALM = 34
    ENTER_GRAVEYARD = 35
    DEFEAT_GNOME_WARRIOR = 36
    DEFEAT_GNOME_ARCHER = 37
    DEFEAT_ORC_SOLIDER = 38
    DEFEAT_ORC_MAGE = 39
    DEFEAT_LIZARD = 40
    DEFEAT_KOBOLD = 41
    DEFEAT_KNIGHT = 65
    DEFEAT_ARCHER = 66
    DEFEAT_TROLL = 42
    DEFEAT_DEEP_THING = 43
    DEFEAT_PIGMAN = 44
    DEFEAT_FIRE_ELEMENTAL = 45
    DEFEAT_FROST_TROLL = 46
    DEFEAT_ICE_ELEMENTAL = 47
    DAMAGE_NECROMANCER = 48
    DEFEAT_NECROMANCER = 49
    EAT_BAT = 50
    EAT_SNAIL = 51
    FIND_BOW = 52
    FIRE_BOW = 53
    LEARN_FIREBALL = 55
    CAST_FIREBALL = 56
    LEARN_ICEBALL = 57
    CAST_ICEBALL = 58
    OPEN_CHEST = 61
    DRINK_POTION = 62
    ENCHANT_SWORD = 63
    ENCHANT_ARMOUR = 64

#when indexing an enum make sure to use .value

Note on BlockType and closest_blocks:
- The first dimension of `closest_blocks` has length `len(BlockType) + 3`.
- Indices `[0 .. len(BlockType)-1]` correspond to BlockType terrain channels (use `BlockType.X.value`).
- In Craftax (current enum ends at `NECROMANCER_VULNERABLE = 36`), the appended ladder channels are fixed to:
  - `37` = LADDER_DOWN
  - `38` = LADDER_UP
  - `39` = LADDER_DOWN_BLOCKED

#Here are example docstrings:

def task_is_done(inventory, inventory_diff, closest_blocks, closest_blocks_prev, player_level, monsters_killed, player_intrinsics, player_intrinsics_diff, achievements, n):
    \"\"\"
    Determines whether Task `{skill_name}` is complete by checking the primary gain from the gain dictionary.
    Do not call external functions or make any assumptions beyond the information given to you.

    Args:
        inventory (Inventory): The player's current inventory, defined in the above struct
        inventory_diff (Inventory): The change in the player's inventory between the current and previous timesteps, same struct as above.
        closest_blocks (numpy.ndarray): A 3D tensor of shape (len(BlockType)+3, 2, K). The first len(BlockType) channels are terrain blocks; the last three channels append item overlay ladders: indices [len(BlockType)+0, +1, +2] map to [LADDER_DOWN, LADDER_UP, LADDER_DOWN_BLOCKED]. Default values are (30, 30) for unseen entries.
        #default of 30,30 if less then k seen, ordered by distance (so :,:,0 would be the closest of each block type.
        # to get the l2 distance of the agent from the closest diamond for example would be jnp.linalg.norm(closest_blocks[BlockType.DIAMOND.value, :, 0]), closest_bocks_changes = l2dist(closest_blocks_prev) - l2dist(closest_blocks)
        closest_blocks_prev (numpy.ndarray): A 3D array of shape (len(BlockType), 2, K) representing the K closest blocks of each type in the previous timestep. Default values are (30, 30) for unseen blocks.
        #default of 30,30 if less then k seen, ordered by distance (so :,:,0 would be the closest of each block type
        player_level (int): The current dungeon level index (0-based)
        monsters_killed (jnp.ndarray): Vector (len=StaticEnvParams.num_levels, usually 9) with counts per floor; use monsters_killed[player_level] to check ladder open state vs LADDER_DOWN_BLOCKED
        player_intrinsics (jnp.ndarray): An len 5 array representing the player's health, food, drink, energy, and mana levels
        player_intrinsics_diff (jnp.ndarray): An len 5 array representing the change in the player's health, food, drink, energy, and mana levels
        achievements (jnp.ndarray): A 1D array (67,) of achievements, where each element is an boolean indicating the corresponding achievement has been completed.
        n (int): The count parameter (use for count-based gains like collecting wood; ignore for tier-based gains or achievements that have fixed targets).

    Returns:
        bool: True if the primary gain condition is satisfied, False otherwise.
    \"\"\"
    return TODO


def task_reward(inventory_diff, closest_blocks, closest_blocks_prev, player_level_diff, monsters_killed_diff, player_intrinsics_diff, achievements_diff, health_penalty):
    \"\"\"
    Calculates the reward for Task `{skill_name}` based on changes in inventory and other factors.
    Do not call external functions or make any assumptions beyond the information given to you.

    Args:
        inventory_diff (Inventory): The change in the player's inventory between the current and previous timesteps, same struct as above.
        closest_blocks (numpy.ndarray): A 3D array of shape (len(BlockType)+3, 2, K): block channels followed by ladder overlays (LADDER_DOWN, LADDER_UP, LADDER_DOWN_BLOCKED),
        #default of 30,30 if less then k seen, ordered by distance (so :,:,0 would be the closest of each block type.
        #Since the environment is a 2d gridworld, an object next to the player will have a distance of 1.
        # to get the l2 distance of the agent from the closest diamond for example would be jnp.linalg.norm(closest_blocks[BlockType.DIAMOND.value, :, 0]), closest_bocks_changes = l2dist(closest_blocks_prev) - l2dist(closest_blocks)
        closest_blocks_prev (numpy.ndarray): A 3D array of shape (len(BlockType), 2, K) representing the K closest blocks of each type in the previous timestep. Default values are (30, 30) for unseen blocks.
        player_level_diff (int): Change in dungeon level this step (e.g., -1, 0, +1)
        monsters_killed_diff (jnp.ndarray): Per-floor differences this step (vector)
        #default of 30,30 if less then k seen, ordered by distance (so :,:,0 would be the closest of each block type
        health_penalty (float): The penalty for losing health. Negative when loosing health and positive when regaining health.
        player_intrinsics_diff (jnp.ndarray): An len 5 array representing the change in the player's health, food, drink, energy, and mana levels
        achievements_diff (jnp.ndarray): A 1D array (67,) of achievements, where each element is an boolean indicating whether the achievement was completed in the last timestep. If the achievement was already completed previously, it will not indicate the achievement was completed again.

    Returns:
        float: Reward for RL agent

    Note:
        The task reward should be two parts:
          1. Sparse reward
          2. Dense reward
        Make sure to disable (2) if (1) is triggered, e.g. sparse_reward +  (sparse_reward == 0.0) * dense_reward

    \"\"\"
    return TODO + health_penalty
```

def task_network_number():
    \"\"\"
    Returns the network index corresponding to the nodes associated skill
    Returns:
        int: Network index

    \"\"\"
    return TODO

Given the above documentations, implement the `task_is_done`, `task_reward`, and task_network_number function for the subtask `{skill_name}` with the following details:
```json
$db.current_{i}.skill_with_consumption$
```

The sparse and dense rewards are:
```
Sparse: $db.reward_{i}$
Dense: $db.dense_reward_{i}$
```

The current number of skills is:
```json
$db.current_{i}.num_skills$
```

## Implementation Guidelines:

For `task_is_done`:
- Identify the main gain entry from the structured "gain" dictionary (the entry representing the primary objective, typically the one whose expression evaluates to `n` for the produced item or predicate)
- Check if the current inventory amount of that main gain item is >= n (the target amount)
- Return True when the target amount is reached, False otherwise
- Use inventory.{{item_name}} to access inventory amounts (e.g., inventory.wood, inventory.stone)
- If a skill is ephemeral, the inventory does not suffice, so for completion criteria you can use closest_blocks or if that doesn't work achievements.

For `task_reward` and `task_network_number`:
- Follow the existing reward structure and network numbering as before

The task network number should be num_skills since we're creating a new skill and the networks are zero indexed.
Do not change the function signature or the docstrings. Do not make any assumptions beyond the information given to you.
The code you write should be able to be jax compiled, no if statements.
No need to retype BlockType, Inventory, and Achievement they will be provided in the environment.
No need to add coefficents to rewards, for example, no need for 10 * inventory_diff.*, just use the raw values.
Return all three functions in a single code block, don't seperate it into 3.
No need to return the docstrings.
Your code will be pasted into a file that already has the following imports. Do not add any additional imports.
from craftax.craftax.constants import *
from craftax.craftax.craftax_state import Inventory
import jax
            """,
            "dep": [],
            "after": [f"dense_reward_{i}"],
            "compose": ComposeBasePrompt(),
            "query": LLM_API_FUNCTION_GPT4,
            "after_query": SaveCodeAfterQuery(i),
        }

    for _, v in code_gen_prompts.items():
        v["prompt"] = v["prompt"].strip()

    return code_gen_prompts


def main():
    """Main function to generate code for skills."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate code for Craftax knowledge base skills')
    parser.add_argument('--skill-index', type=int, default=None,
                        help='Process only the skill at this index')
    parser.add_argument('--save-single', action='store_true',
                        help='When used with --skill-index, save only that skill into the verified knowledge base')
    args = parser.parse_args()

    # Get project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Paths
    input_json_path = os.path.join(project_root, "resources", "craftax_knowledgebase_verified.json")
    output_json_path = os.path.join(project_root, "resources", "craftax_knowledgebase_verified_with_code.json")
    output_txt_path = os.path.join(project_root, "resources", "craftax_knowledgebase_verified_with_code.txt")

    print(f"Loading verified knowledge base from: {input_json_path}")

    if not os.path.exists(input_json_path):
        print(f"ERROR: Verified knowledge base file not found at {input_json_path}")
        print("Please run add_prerequisites_craftax_kb.py first")
        sys.exit(1)

    # Load existing knowledge base
    with open(input_json_path, 'r') as f:
        kb = json.load(f)

    skills_list = kb.get("skills", [])

    print(f"Loaded {len(skills_list)} skills")

    # Validate skill index if provided
    if args.skill_index is not None:
        if args.skill_index < 0 or args.skill_index >= len(skills_list):
            print(f"ERROR: Invalid skill index {args.skill_index}. Valid range: 0-{len(skills_list)-1}")
            sys.exit(1)
        print(f"Processing only skill {args.skill_index}: '{skills_list[args.skill_index]['skill_name']}'")

    # Setup database
    db = {
        "skills": skills_list,
        "generated_code": {},
    }

    # Generate graph and evaluate
    if args.skill_index is not None:
        print(f"Generating code for skill {args.skill_index} using LLM...")
    else:
        print("Generating code for all skills using LLM...")
    graph = generate_graph(db, skill_index=args.skill_index)
    graph.evaluate()

    # Update skills with generated code
    for i, code in db["generated_code"].items():
        skills_list[i]["functions"] = code
        print(f"Updated skill '{skills_list[i]['skill_name']}' with generated code")

    # If debugging single skill, print details and exit without saving
    if args.skill_index is not None and not args.save_single:
        print("\n" + "="*80)
        print(f"DEBUG OUTPUT for skill {args.skill_index}:")
        print("="*80)
        print(json.dumps(skills_list[args.skill_index], indent=2))
        print("\nNot saving (debug mode). Re-run with --save-single to update only this skill, or run without --skill-index to save all skills.")
        return

    # If saving a single skill, update only that entry in the output KB
    if args.skill_index is not None and args.save_single:
        target_idx = args.skill_index
        updated_skill = skills_list[target_idx]
        target_name = updated_skill.get('skill_name', f'skill_{target_idx}')

        # Load existing output KB if present, else start from the input kb we loaded
        output_kb = None
        if os.path.exists(output_json_path):
            try:
                with open(output_json_path, 'r') as f:
                    output_kb = json.load(f)
                print(f"Loaded existing output KB from: {output_json_path}")
            except Exception as e:
                print(f"Warning: Could not load output KB ({e}); starting from input KB")

        if not output_kb:
            output_kb = kb

        output_skills = output_kb.get('skills', [])

        # Find by name first to avoid index drift
        found_j = None
        for j, s in enumerate(output_skills):
            if s.get('skill_name') == target_name:
                found_j = j
                break

        if found_j is not None:
            output_skills[found_j].update(updated_skill)
            print(f"Updated output KB skill by name match: index {found_j} -> '{target_name}'")
        elif 0 <= target_idx < len(output_skills):
            output_skills[target_idx].update(updated_skill)
            print(f"Updated output KB skill by index: {target_idx} -> '{target_name}'")
        else:
            output_skills.append(updated_skill)
            print(f"Appended skill to output KB (no name/index match): '{target_name}'")

        output_kb['skills'] = output_skills

        # Save updated output KB (JSON)
        print(f"Saving updated single-skill output knowledge base to: {output_json_path}")
        with open(output_json_path, 'w') as f:
            json.dump(output_kb, f, indent=2)

        # Also regenerate the pretty text file
        print(f"Saving human-readable version to: {output_txt_path}")
        with open(output_txt_path, 'w') as f:
            f.write("CRAFTAX KNOWLEDGE BASE WITH CODE\n")
            f.write("=" * 80 + "\n\n")

            def write_dict(d, indent=0):
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

            write_dict(output_kb)

        print("\nSingle-skill code generation complete!")
        print(f"Skill: {target_name}")
        print(f"JSON: {output_json_path}")
        print(f"Text: {output_txt_path}")
        return

    # Save updated knowledge base with generated code
    kb["skills"] = skills_list

    print(f"Saving knowledge base with generated code to: {output_json_path}")
    with open(output_json_path, 'w') as f:
        json.dump(kb, f, indent=2)

    # Save as pretty text
    print(f"Saving human-readable version to: {output_txt_path}")
    with open(output_txt_path, 'w') as f:
        f.write("CRAFTAX KNOWLEDGE BASE WITH CODE\n")
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

        write_dict(kb)

    print("\nCode generation complete!")
    print(f"JSON: {output_json_path}")
    print(f"Text: {output_txt_path}")


if __name__ == "__main__":
    main()
