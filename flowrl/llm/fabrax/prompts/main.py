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
    wood_pickaxe: int
    stone_pickaxe: int
    iron_pickaxe: int
    wood_sword: int
    stone_sword: int
    iron_sword: int
    copper: int
    tin: int
    sand: int
    clay: int
    limestone: int
    leather: int
    iron_bar: int
    steel_bar: int
    bronze_bar: int
    glass: int
    brick: int
    lime: int
    tar: int
    bottle: int
    lens: int
    telescope: int
    mortar: int
    fertilizer: int
    flux: int
    steel_pickaxe: int
    bronze_pickaxe: int
    steel_sword: int
    bronze_sword: int
    tonic_basic: int
    tonic_stoneskin: int

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
    COPPER = 17
    TIN = 18
    CLAY = 19
    LIMESTONE = 20
    ANVIL = 21
    KILN = 22
    COMPOSTER = 23
    ALCHEMY_BENCH = 24
    WINDOW = 25
    WALL_MASONRY = 26

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
    PLACE_ANVIL = 17
    PLACE_KILN = 18
    PLACE_COMPOSTER = 19
    PLACE_ALCHEMY_BENCH = 20
    MAKE_BELLOWS = 21
    SMELT_IRON_BAR = 22
    MAKE_STEEL_BAR = 23
    MAKE_BRONZE_BAR = 24
    FORGE_STEEL_PICKAXE = 25
    FORGE_BRONZE_PICKAXE = 26
    FORGE_STEEL_SWORD = 27
    FORGE_BRONZE_SWORD = 28
    HARDEN_EDGE = 29
    SMELT_GLASS = 30
    FIRE_BRICK = 31
    MAKE_GLASS_BOTTLE = 32
    MAKE_LENS = 33
    MAKE_TELESCOPE = 34
    MAKE_LIME = 35
    MAKE_MORTAR = 36
    PLACE_WINDOW = 37
    PLACE_WALL_MASONRY = 38
    MAKE_TAR = 39
    MAKE_FERTILIZER = 40
    BREW_TONIC = 41
    BREW_STONE_SKIN = 42
    MAKE_FLUX = 43

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
    PLACE_ANVIL = 22
    MAKE_BELLOWS = 23
    SMELT_IRON_BAR = 24
    COLLECT_COPPER = 25
    COLLECT_TIN = 26
    MAKE_STEEL_BAR = 27
    MAKE_BRONZE_BAR = 28
    FORGE_STEEL_PICKAXE = 29
    FORGE_BRONZE_PICKAXE = 30
    HARDEN_EDGE = 31
    FORGE_STEEL_SWORD = 32
    FORGE_BRONZE_SWORD = 33
    COLLECT_SAND = 34
    PLACE_KILN = 35
    SMELT_GLASS = 36
    COLLECT_CLAY = 37
    FIRE_BRICK = 38
    MAKE_GLASS_BOTTLE = 39
    MAKE_LENS = 40
    COLLECT_LIMESTONE = 41
    MAKE_LIME = 42
    MAKE_TELESCOPE = 43
    MAKE_MORTAR = 44
    PLACE_WINDOW = 45
    PLACE_WALL_MASONRY = 46
    PLACE_COMPOSTER = 47
    MAKE_TAR = 48
    MAKE_FERTILIZER = 49
    PLACE_ALCHEMY_BENCH = 50
    BREW_TONIC = 51
    BREW_STONE_SKIN = 52
    MAKE_FLUX = 53
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
- The frontier summary shows the current reachable symbolic states (inventory items and achievements) based on existing skills
- You MUST propose a skill that extends this frontier by achieving NEW symbolic states not currently reachable
- Verify that your proposed skill's preconditions are FEASIBLE (can be satisfied by current frontier)
- Verify that your proposed skill's gains are NOVEL (not already achievable by existing skills)

Fill out the following sections explicitly before arriving at the final formatted output.

## Review Existing Skills
In a few sentences, review existing skills.

## Frontier Analysis
Analyze the current symbolic frontier:
- What inventory items and achievements are currently reachable?
- What are the obvious gaps or next logical extensions to this frontier?
- Which symbolic states would be most valuable to reach next?

## Future Objectives
List 2-3 concrete next skills that extend the current frontier. For each, provide ONE sentence covering:
- Novel gains (new inventory or achievements not in frontier)
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

Format: `"key": {"type": "inventory|achievement|ephemeral", "expression": "lambda n: ...", "description": "..."}`

Expression patterns:
- Count-based gains (wood, stone, etc.): `"expression": "lambda n: n"`
- Achievements: `"expression": "lambda n: 1"`

Use `achievement:` prefixes where applicable.


# Formatting
Finally, complete the following Json dictionary as your output.
```json
{
"skill_name": ,              # name of the objective
"description": ,            # (string) 1-line description
"gain": {                   # structured gains following the schema above
  "gain_key": {
    "type": "inventory | achievement | ephemeral",
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
$db.skills_without_code$
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
- Each key in requirements/consumption must exactly match a gain key from an existing skill (including prefixes like `achievement:` when applicable). 
- Gains for the current skill must use the structured schema: each key maps to `{ "type": ..., "expression": ..., "description": ... }` following the same conventions as the proposal step.
  
  
# Formatting
Finally, complete the following Json dictionary as your output.
```json
{
"skill_name": , # name of the current skill
"requirements": , # (dict) total amount needed available using "lambda n: a*n + b" format. Each key must exactly match the key of a gain of a previous skill.
"consumption": , # (dict) amount consumed using "lambda n: a*n + b" format. Each key must exactly match the key of a gain of a previous skill.
"gain": { # (dict) structured gains; for count-based gains the expression evaluates to n, for achievements use lambda n: 1
  "gain_key": {
    "type": "inventory | achievement | ephemeral",
    "expression": "lambda n: ...",  # For counts: n; For achievements: 1
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

    prompts["continue_training_decision"] = {
        "prompt": """
Analyze whether to CONTINUE TRAINING an existing prerequisite skill instead of proceeding with the proposed skill.

Existing Skills (metrics may appear under each entry):
```
$db.skills_without_code$
```

Current Skill:
```
$db.current.skill_with_consumption$
```

# Instruction
- Map requirement/consumption keys to the skills that produce those gains.
- If a prerequisite skill shows low success consider continuing its training first. Treat success_rate < 0.5 as low success.
- Only recommend continuation when it blocks meaningful progress on the current skill.
- Skills are trained initially for up to 10 million timesteps. Propose an increment of 10 million for further training. if extra timesteps is needed.

# Output JSON
```json
{
  "continue_training": ,
  "skill_name": ,
  "extra_timesteps": 0,
  "reason": ""
}
```
        """,
        "dep": [],
        "after": ["next_subtask"],
        "compose": ComposeReasoningPrompt(),
        "query": LLM_API_FUNCTION_GPT4,
        "after_query": ContinueTrainingDecisionAfterQuery(),
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
    COPPER = 17
    TIN = 18
    CLAY = 19
    LIMESTONE = 20
    ANVIL = 21
    KILN = 22
    COMPOSTER = 23
    ALCHEMY_BENCH = 24
    WINDOW = 25
    WALL_MASONRY = 26

@struct.dataclass
class Inventory:
    wood: int
    stone: int
    coal: int
    iron: int
    diamond: int
    sapling: int
    wood_pickaxe: int
    stone_pickaxe: int
    iron_pickaxe: int
    wood_sword: int
    stone_sword: int
    iron_sword: int
    copper: int
    tin: int
    sand: int
    clay: int
    limestone: int
    leather: int
    iron_bar: int
    steel_bar: int
    bronze_bar: int
    glass: int
    brick: int
    lime: int
    tar: int
    bottle: int
    lens: int
    telescope: int
    mortar: int
    fertilizer: int
    flux: int
    steel_pickaxe: int
    bronze_pickaxe: int
    steel_sword: int
    bronze_sword: int
    tonic_basic: int
    tonic_stoneskin: int

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
    PLACE_ANVIL = 22
    MAKE_BELLOWS = 23
    SMELT_IRON_BAR = 24
    COLLECT_COPPER = 25
    COLLECT_TIN = 26
    MAKE_STEEL_BAR = 27
    MAKE_BRONZE_BAR = 28
    FORGE_STEEL_PICKAXE = 29
    FORGE_BRONZE_PICKAXE = 30
    HARDEN_EDGE = 31
    FORGE_STEEL_SWORD = 32
    FORGE_BRONZE_SWORD = 33
    COLLECT_SAND = 34
    PLACE_KILN = 35
    SMELT_GLASS = 36
    COLLECT_CLAY = 37
    FIRE_BRICK = 38
    MAKE_GLASS_BOTTLE = 39
    MAKE_LENS = 40
    COLLECT_LIMESTONE = 41
    MAKE_LIME = 42
    MAKE_TELESCOPE = 43
    MAKE_MORTAR = 44
    PLACE_WINDOW = 45
    PLACE_WALL_MASONRY = 46
    PLACE_COMPOSTER = 47
    MAKE_TAR = 48
    MAKE_FERTILIZER = 49
    PLACE_ALCHEMY_BENCH = 50
    BREW_TONIC = 51
    BREW_STONE_SKIN = 52
    MAKE_FLUX = 53
```
The reward function is calculated independently at each timestep using these available factors:

- inventory_diff (Inventory): The change in the player's inventory between the current and previous timesteps. For count fields (wood, stone, coal, wood_pickaxe, etc.), this is +1 per item gained, -1 per item used.
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
        "after": ["next_subtask", "continue_training_decision"],
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
    COPPER = 17
    TIN = 18
    CLAY = 19
    LIMESTONE = 20
    ANVIL = 21
    KILN = 22
    COMPOSTER = 23
    ALCHEMY_BENCH = 24
    WINDOW = 25
    WALL_MASONRY = 26

@struct.dataclass
class Inventory:
    wood: int
    stone: int
    coal: int
    iron: int
    diamond: int
    sapling: int
    wood_pickaxe: int
    stone_pickaxe: int
    iron_pickaxe: int
    wood_sword: int
    stone_sword: int
    iron_sword: int
    copper: int
    tin: int
    sand: int
    clay: int
    limestone: int
    leather: int
    iron_bar: int
    steel_bar: int
    bronze_bar: int
    glass: int
    brick: int
    lime: int
    tar: int
    bottle: int
    lens: int
    telescope: int
    mortar: int
    fertilizer: int
    flux: int
    steel_pickaxe: int
    bronze_pickaxe: int
    steel_sword: int
    bronze_sword: int
    tonic_basic: int
    tonic_stoneskin: int

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
    PLACE_ANVIL = 22
    MAKE_BELLOWS = 23
    SMELT_IRON_BAR = 24
    COLLECT_COPPER = 25
    COLLECT_TIN = 26
    MAKE_STEEL_BAR = 27
    MAKE_BRONZE_BAR = 28
    FORGE_STEEL_PICKAXE = 29
    FORGE_BRONZE_PICKAXE = 30
    HARDEN_EDGE = 31
    FORGE_STEEL_SWORD = 32
    FORGE_BRONZE_SWORD = 33
    COLLECT_SAND = 34
    PLACE_KILN = 35
    SMELT_GLASS = 36
    COLLECT_CLAY = 37
    FIRE_BRICK = 38
    MAKE_GLASS_BOTTLE = 39
    MAKE_LENS = 40
    COLLECT_LIMESTONE = 41
    MAKE_LIME = 42
    MAKE_TELESCOPE = 43
    MAKE_MORTAR = 44
    PLACE_WINDOW = 45
    PLACE_WALL_MASONRY = 46
    PLACE_COMPOSTER = 47
    MAKE_TAR = 48
    MAKE_FERTILIZER = 49
    PLACE_ALCHEMY_BENCH = 50
    BREW_TONIC = 51
    BREW_STONE_SKIN = 52
    MAKE_FLUX = 53

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
        player_intrinsics (jnp.ndarray): An len 4 array representing the player's health, food, drink, and energy levels
        player_intrinsics_diff (jnp.ndarray): An len 4 array representing the change in the player's health, food, drink, and energy levels
        achievements (jnp.ndarray): A 1D array (54,) of achievements, where each element is an boolean indicating the corresponding achievement has been completed.
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
        closest_blocks_prev (numpy.ndarray): A 3D array of shape (len(BlockType), 2, K) representing the K closest blocks of each type in the previous timestep. Default values are (30, 30) for unseen blocks.
        #default of 30,30 if less then k seen, ordered by distance (so :,:,0 would be the closest of each block type
        health_penalty (float): The penalty for losing health. Negative when loosing health and positive when regaining health.
        player_intrinsics_diff (jnp.ndarray): An len 4 array representing the change in the player's health, food, drink, and energy levels
        achievements_diff (jnp.ndarray): A 1D array (54,) of achievements, where each element is an boolean indicating whether the achievement was completed in the last timestep. If the achievement was already completed previously, it will not indicate the achievement was completed again.

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
from craftax.fabrax.constants import *
from craftax.fabrax.envs.craftax_state import Inventory
import jax
        """,
        "dep": [],
        "after": [
            "next_subtask",
            "continue_training_decision",
            "create_skill_densify_reward_reasoning",
        ],
        "compose": ComposeReasoningPrompt(),
        "query": LLM_API_FUNCTION_GPT4,
        # TODO: Add after_query
    }
    return prompts, {}
