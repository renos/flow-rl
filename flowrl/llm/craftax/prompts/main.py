from flowrl.llm.compose_prompts import ComposeReasoningPrompt
from flowrl.llm.craftax.after_queries import *


def return_prompts(LLM_API_FUNCTION_GPT4):
    prompts = {}

    prompts["next_task"] = {
        "prompt": """

Environment Details:
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
# Max inventory values are higher in Craftax vs Craftax Classic

# ENUMS
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

class Action(Enum):
    NOOP = 0  #
    LEFT = 1  # a
    RIGHT = 2  # d
    UP = 3  # w
    DOWN = 4  # s
    DO = 5  # space
    SLEEP = 6  # tab
    PLACE_STONE = 7  # r
    PLACE_TABLE = 8  # t
    PLACE_FURNACE = 9  # f
    PLACE_PLANT = 10  # p
    MAKE_WOOD_PICKAXE = 11  # 1
    MAKE_STONE_PICKAXE = 12  # 2
    MAKE_IRON_PICKAXE = 13  # 3
    MAKE_WOOD_SWORD = 14  # 4
    MAKE_STONE_SWORD = 15  # 5
    MAKE_IRON_SWORD = 16  # 6
    REST = 17  # e
    DESCEND = 18  # >
    ASCEND = 19  # <
    MAKE_DIAMOND_PICKAXE = 20  # 4
    MAKE_DIAMOND_SWORD = 21  # 8
    MAKE_IRON_ARMOUR = 22  # y
    MAKE_DIAMOND_ARMOUR = 23  # u
    SHOOT_ARROW = 24  # i
    MAKE_ARROW = 25  # o
    CAST_FIREBALL = 26  # g
    CAST_ICEBALL = 27  # h
    PLACE_TORCH = 28  # j
    DRINK_POTION_RED = 29  # z
    DRINK_POTION_GREEN = 30  # x
    DRINK_POTION_BLUE = 31  # c
    DRINK_POTION_PINK = 32  # v
    DRINK_POTION_CYAN = 33  # b
    DRINK_POTION_YELLOW = 34  # n
    READ_BOOK = 35  # m
    ENCHANT_SWORD = 36  # k
    ENCHANT_ARMOUR = 37  # l
    MAKE_TORCH = 38  # [
    LEVEL_UP_DEXTERITY = 39  # ]
    LEVEL_UP_STRENGTH = 40  # -
    LEVEL_UP_INTELLIGENCE = 41  # =
    ENCHANT_BOW = 42  # ;

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

Knowledgebase:
```
$db.knowledge_base$
```

Existing Skills:
```
$db.skills_without_code$
```

Current Symbolic Frontier:
```
$db.frontier_summary$
```

# Instruction
Consider the knowledgebase, existing skills, and current frontier. Identify the next skill that should be learned. Pay special attention to the task requirements and action prerequisites from the knowledgebase.

**CRITICAL: FRONTIER-GUIDED SKILL PROPOSAL**
- The frontier summary shows the current reachable symbolic states (inventory items, achievements, levels) based on existing skills
- You MUST propose a skill that extends this frontier by achieving NEW symbolic states not currently reachable
- Verify that your proposed skill's preconditions are FEASIBLE (can be satisfied by current frontier)
- Verify that your proposed skill's gains are NOVEL (not already achievable by existing skills)

Fill out the following sections explicitly before arriving at the final formatted output.

## Review Existing Skills
In a few sentences, review existing skills.

## Frontier Analysis
Analyze the current symbolic frontier:
- What inventory items, achievements, and levels are currently reachable?
- What are the obvious gaps or next logical extensions to this frontier?
- Which symbolic states would be most valuable to reach next?

## Future Objectives
List 2-3 concrete next skills that extend the current frontier. For each, provide ONE sentence covering:
- Novel gains (new inventory/achievements/levels not in frontier)
- Feasibility (requirements satisfied by frontier)
Keep this section brief and focused.

## Immediate Objective
State the single best next skill and justify it in 1-2 sentences based on frontier analysis.

CRITICAL CHECKS:
- Skill does NOT already exist in existing skills
- Requirements CAN be fulfilled by current frontier
- Gains are NEW (extend beyond current frontier)

# Gain Schema
Each skill has ONE primary goal. Include all symbolic state changes that occur from achieving that goal.

Format: `"key": {"type": "inventory|achievement|level|ephemeral", "expression": "lambda n: ...", "description": "..."}`

Expression patterns:
- Count-based gains (wood, stone, arrows, etc.): `"expression": "lambda n: n"`
- Tier-based gains (pickaxe, sword): `"expression": "lambda n: tier_value"` (e.g., `lambda n: 2` for stone pickaxe)
- Achievements: `"expression": "lambda n: 1"`

Use `achievement:` and `level:` prefixes where applicable.


# Formatting
Finally, complete the following Json dictionary as your output.
```json
{
"skill_name": # name of the objective
"description": # (string) 1-line description of the objective
"gain": { # structured gains following the schema above
  "gain_key": {
    "type": "inventory | achievement | level | ephemeral",
    "expression": "lambda n: ...",
    "description": "optional context"
  }
}
}
```
        """,
        "dep": [],
        "compose": ComposeReasoningPrompt(),
        "query": LLM_API_FUNCTION_GPT4,
        "after_query": TaskAfterQuery(),
    }

    prompts["next_subtask"] = {
        "prompt": """
Consider the Knowledgebase and existing skills.

Knowledgebase:
```
$db.knowledge_base$
```

Existing Skills
```
$db.skills$
```

Skill to Learn
```
$db.current.skill$
```

# Instruction


## Analyze Knowledgebase
Identify and draw connections between the skill to learn and any relevant existing knowledge.

## Task Analysis
Explicitly analyze the current skill:
- What is the core objective?
- What are the specific requirements? 
- What resources are consumed when applying the skill.

## Previous Skill Analysis
In a bulleted list, write what each skill gains using the structured gain schema (key + type + expression). The requirements and consumption dictionaries for the current skill must be written solely in terms of the gain keys produced by existing skills.

## Ephemeral Analysis
Determine if this skill is ephemeral. A skill is ephemeral if the gain itself is not observable in the inventory. 

# Note
- Distance/adjaceny CANNOT be directly verified or quantified, but you can use the closest blocks as a proxy.
- Skills should be explicit and complete on their own, and should convey a clear quantifiable goal. The purpose of requirements/consumption is to later parse what other skills need to be performed before applying the skill.
- Requirements are a SUPERSET of consumption: requirements include everything needed (both consumed and non-consumed resources), while consumption only includes what gets used up.
- Each value in requirements/consumption should be written as a Python lambda function string that takes n and returns the amount needed, in the form: "lambda n: a*n + b", where:
  - a = amount of resource consumed PER unit of gain (scales with n)
  - b = amount of resource required but NOT consumed (fixed amount regardless of n)
  - Ask yourself: "Does this requirement scale with the number of times I apply the skill?"
    - If YES (scales with n): use "lambda n: a*n + 0" format
    - If NO (fixed amount): use "lambda n: 0*n + b" format
- Requirements do not support 'or'
- Each key in requirements/consumption must exactly match a gain key from an existing skill (including prefixes like `achievement:` or `level:` when applicable). 
- Gains for the current skill must use the structured schema: each key maps to `{ "type": ..., "expression": ..., "description": ... }` following the same conventions as the proposal step.
  
  
# Formatting
Finally, complete the following Json dictionary as your output.
```json
{
"skill_name": , # name of the current skill
"requirements": , # (dict) total amount needed available using "lambda n: a*n + b" format. Each key must exactly match the key of a gain of a previous skill.
"consumption": , # (dict) amount consumed using "lambda n: a*n + b" format. Each key must exactly match the key of a gain of a previous skill.
"gain": { # (dict) structured gains; for count-based gains the expression evaluates to n, for tier/achievement gains use lambda n: target_value
  "gain_key": {
    "type": "inventory | achievement | level | ephemeral",
    "expression": "lambda n: ...",  # For counts: n; For tiers: specific tier value; For achievements: 1
    "description": "optional"
  }
}
"ephemeral": , # (bool) true if the gain itself is not observable in the inventory, false if the gain appears directly in the inventory
}
```
        """,
        "dep": [],
        "after": ["next_task"],
        "compose": ComposeReasoningPrompt(),
        "query": LLM_API_FUNCTION_GPT4,
        "after_query": SubtaskAfterQuery(),
    }

    # temp_prompts["densify_reward_reasoning"]
    prompts["create_skill_densify_reward_reasoning"] = {
        "prompt": """
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
- player_intrinsics (jnp.ndarray): The intrinsic values
- player_intrinsics_diff (jnp.ndarray): The changes in current intrinsic values from the last timestep to the current timestep.


# Other Information
- This reward function is called independently at each timestep
- Each timestep's reward is calculated using only information from the current and previous timestep
- The reward at timestep t cannot access information from timestep t-2 or earlier
- The completion criteria is a separate function; do not worry about implementing it
- No state can be stored between timesteps - each reward calculation must be independent

# Skill
Given the following skill, design the reward function for the Skill `$db.current.skill_name$`
```
$db.current.skill_with_consumption$
```

# Design Approach
Return the raw factor value that measures the skill's gain. Requirements and consumption are enforced elsewhere.

1. Identify the factor measuring the gain
2. Sparse reward: return the raw factor (can be positive, negative, or zero)
3. Dense reward: optional raw factor that provides denser feedback (can be positive, negative, or zero) toward the gain, scaled by coefficient ≤ 0.01, or "NA"

```json
{
"sparse_reward_only_function": # (str) "return <factor>"
"dense_reward_function": # (str) "return <coefficient> * <dense_factor>", or "NA"
}
```
        """,
        "dep": [],
        "after": ["next_subtask"],
        "compose": ComposeReasoningPrompt(),
        "query": LLM_API_FUNCTION_GPT4,
        "after_query": DensifyAfterQuery(),
    }

    prompts["create_skill_coding"] = {
        "prompt": """
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

#Here are example docstrings:

def task_is_done(inventory, inventory_diff, closest_blocks, closest_blocks_prev, player_intrinsics, player_intrinsics_diff, achievements, n):
    \"\"\"
    Determines whether Task `$db.current.skill_name$` is complete by checking the primary gain from the gain dictionary.
    Do not call external functions or make any assumptions beyond the information given to you.

    Args:
        inventory (Inventory): The player's current inventory, defined in the above struct
        inventory_diff (Inventory): The change in the player's inventory between the current and previous timesteps, same struct as above.
        closest_blocks (numpy.ndarray): A 3D tensor of shape (len(BlockType), 2, K) representing the K closest blocks of each type. Default values are (30, 30) for unseen blocks.
        #default of 30,30 if less then k seen, ordered by distance (so :,:,0 would be the closest of each block type.
        # to get the l2 distance of the agent from the closest diamond for example would be jnp.linalg.norm(closest_blocks[BlockType.DIAMOND.value, :, 0]), closest_bocks_changes = l2dist(closest_blocks_prev) - l2dist(closest_blocks)
        closest_blocks_prev (numpy.ndarray): A 3D array of shape (len(BlockType), 2, K) representing the K closest blocks of each type in the previous timestep. Default values are (30, 30) for unseen blocks.
        #default of 30,30 if less then k seen, ordered by distance (so :,:,0 would be the closest of each block type
        player_intrinsics (jnp.ndarray): An len 5 array representing the player's health, food, drink, energy, and mana levels
        player_intrinsics_diff (jnp.ndarray): An len 5 array representing the change in the player's health, food, drink, energy, and mana levels
        achievements (jnp.ndarray): A 1D array (67,) of achievements, where each element is an boolean indicating the corresponding achievement has been completed.
        n (int): The count parameter (use for count-based gains like collecting wood; ignore for tier-based gains or achievements that have fixed targets).

    Returns:
        bool: True if the primary gain condition is satisfied, False otherwise.
    \"\"\"
    return TODO


def task_reward(inventory_diff, closest_blocks, closest_blocks_prev, player_intrinsics_diff, achievements_diff, health_penalty):
    \"\"\"
    Calculates the reward for Task `$db.current.skill_name$` based on changes in inventory and other factors.
    Do not call external functions or make any assumptions beyond the information given to you.

    Args:
        inventory_diff (Inventory): The change in the player's inventory between the current and previous timesteps, same struct as above.
        closest_blocks (numpy.ndarray): A 3D array of shape (len(BlockType), 2, K) representing the K closest blocks of each type, 
        #default of 30,30 if less then k seen, ordered by distance (so :,:,0 would be the closest of each block type.
        #Since the environment is a 2d gridworld, an object next to the player will have a distance of 1.
        # to get the l2 distance of the agent from the closest diamond for example would be jnp.linalg.norm(closest_blocks[BlockType.DIAMOND.value, :, 0]), closest_bocks_changes = l2dist(closest_blocks_prev) - l2dist(closest_blocks)
        closest_blocks_prev (numpy.ndarray): A 3D array of shape (len(BlockType), 2, K) representing the K closest blocks of each type in the previous timestep. Default values are (30, 30) for unseen blocks.
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

Given the above documentations, implement the `task_is_done`, `task_reward`, and task_network_number function for the subtask `$db.current.skill_name$` with the following details:
```json
$db.current.skill_with_consumption$
$db.current.reward$
```
The dense reward to include is:
```json
$db.current.dense_reward_factor$
```

The current number of skills is: 
```json
$db.current.num_skills$
```

## Implementation Guidelines:

For `task_is_done`: 
- Identify the main gain entry from the structured "gain" dictionary (the entry representing the primary objective, typically the one whose expression evaluates to `n` for the produced item or predicate)
- Check if the current inventory amount of that main gain item is >= n (the target amount)
- Return True when the target amount is reached, False otherwise
- Use inventory.{item_name} to access inventory amounts (e.g., inventory.wood, inventory.stone)
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
        "after": [
            "next_subtask",
            "create_skill_densify_reward_reasoning",
        ],
        "compose": ComposeReasoningPrompt(),
        "query": LLM_API_FUNCTION_GPT4,
        # TODO: Add after_query
    }
    return prompts, {}
