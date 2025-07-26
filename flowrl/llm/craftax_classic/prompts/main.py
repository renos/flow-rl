from flowrl.llm.compose_prompts import ComposeReasoningPrompt
from flowrl.llm.craftax_classic.after_queries import *


def return_prompts(LLM_API_FUNCTION_GPT4):
    prompts = {}

    prompts["next_task"] = {
        "prompt": """

Environment Details:
@struct.dataclass
class Inventory:
    wood: int = 0
    stone: int = 0
    coal: int = 0
    iron: int = 0
    diamond: int = 0
    sapling: int = 0
    wood_pickaxe: int = 0
    stone_pickaxe: int = 0
    iron_pickaxe: int = 0
    wood_sword: int = 0
    stone_sword: int = 0
    iron_sword: int = 0
#max inventory size is 9 for each item

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

Knowledgebase:
```
$db.knowledge_base$
```

Existing Skills:
```
$db.skills$
```

# Instruction
Consider the knowledgebase, and existing skills. Identify the next skill that should be learned. Pay special attention to the task requirements and action prerequisites from the knowledgebase.
Fill out the following sections explicitly before arriving at the final formatted output.

## Future Objectives
List up to 3 potential future objectives that the player could work toward next. For each objective, briefly discuss the necessity, benefits, requirements.

## Review Existing Skills
In a few sentences, review existing skills. Do not propose a skill which has already been learned.

## Immediate Objective
Identify the next skill the player should learn based on your analysis. CRITICAL: Do NOT propose any skill that already exists in the existing skills list. You should only propose NEW skills whose requirements can be fulfilled by preexisting skills. 

# Note
- Distance/adjaceny cannot be directly tracked, but you can use the closest blocks as a proxy.
- Write gain in terms of n, the number of times a skill will be performed. For each requirement, write it in terms of n if the skill consumes that requirement.


# Formatting
Finally, complete the following Json dictionary as your output.
```json
{
"skill_name": # name of the objective
"description": # (string) 1-line description of the objective
"requirements": # (string) requirements for the objective.
"gain": # (str) what the player will gain after applying the skill. 
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
In a bulleted list, write what each skill gains. The requirements/consumption dictionarys for the current skill must be written soley in terms of the gains of existing skills.

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
- Each key in requirements/consumption must be a key in the gain of an existing skill. 
  
# Formatting
Finally, complete the following Json dictionary as your output.
```json
{
"skill_name": , # name of the current skill
"requirements": , # (dict) a dictionary requirements needed before the player can apply the skill. Each key must exactly match the key of a gain of a previous skill.
"consumption": , # (dict): a dictionary of what resources are consumed by applying the skill. Each key must exactly match the key of a gain of a previous skill.
"gain": , # (dict) a dictionary of what is gained by applying the skill. The gain for the skill goal should be n.
"completion_criteria": , # (string) description of when this skill is considered complete
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
# Max inventory value is 9, max player intrinsics values are also 9 
@struct.dataclass
class Inventory:
    wood: int = 0
    stone: int = 0
    coal: int = 0
    iron: int = 0
    diamond: int = 0
    sapling: int = 0
    wood_pickaxe: int = 0
    stone_pickaxe: int = 0
    iron_pickaxe: int = 0
    wood_sword: int = 0
    stone_sword: int = 0
    iron_sword: int = 0

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
```
The reward function is calculated independently at each timestep using these available factors:

- inventory_diff (Inventory): The change in the player's inventory between the current and previous timesteps (-1 for each item used and +1 for each item gained).
- closest_bocks_changes (numpy.ndarray): The changes in distance to closest blocks of each type from the last timestep to the current timestep. Decreases in distance are positive. If an item has moves from being unseen to seen, the default will be 30-current_distance. E.g. if a table is placed in front of the player, the distanc diff will be 29.
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

# Steps
Explicitly complete the following steps before arriving at your final formatted output
0. Analyze Skill Gains and identify appropriate reward factors:
   - What is the core objective of this subtask?
   - What specific behaviors or outcomes need to be rewarded?
   - For each available factor, determine if it can provide meaningful feedback for the required behaviors
   - Remove any factors that are irrelevant to the subtask objectives or should not be used.
   - Assume all requirements for the skill has been met before the skill is applied.
   List the remaining factors that will be analyzed in subsequent steps.
1. Analyze each factor's per-timestep behavior, responding to each question explicitly:
   - How does the raw factor behave at each individual timestep?
   - What does a positive vs negative value mean at a single timestep?
   - What is measured when we use this raw factor as a direct reward?
   - Write out a sequence of timestep values for a potential reward hacking attempt. Sum these values.
   - Based on the sequence sum: Does the reward cycling result in positive net reward? If so, state reward hacking is possible since the agent can repeat this cycle indefinitely for unbounded reward. If the cycle results in zero or negative net reward, state the raw factor naturally prevents reward hacking since repeating the cycle cannot generate unbounded reward.
   - Write the exact transformation: If we concluded the raw factor naturally prevents reward hacking, write the factor name exactly as it appears in the available factors (e.g., "Transform = inventory_diff"). Otherwise, write the minimal equation required and explain why it's necessary based on the sequence analysis.

2. Filter out factors with no obvious non-hackable reward functions or those that are not relevant to the task.

3. Classify the remaining factors in to dense and sparse rewards. The chosen sparse reward should be a single factor that best represents the main objective of the subtask. Justify your choice.
4. Design a minimalistic sparse reward formula:
   - Use the raw factor directly if it was shown to naturally prevent reward hacking
   - Include only the minimum operations needed for the reward signal
   - Verify the formula matches your timestep sequence analysis from step 1

5. Design a dense reward formula:
   - For each factor proven safe in step 1, include it directly 
   - If multiple factors are valid, combine them through simple addition
   - No additional transformations beyond what was proven necessary in step 1
   - For each factor included, include a coefficient between 0.0 and 1.0 such that the the magnitude of sparse reward over-powers dense reward and output in the requested format.
   - Write "NA" if no dense reward is needed

6. Write both rewards into mathematical formula, and double-check for redundancy

# Note
- The optimization stops when completion critiera is met, so no more rewards will be provided after completion.

If no dense reward function is possible or needed for this task, simply state NA.

```json
{
"sparse_reward_only_function": # (str) Minimal reward pseudocode
"dense_reward_function": # (str) Dense reward pseudocode, "NA" if not available
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
# Max inventory value is 9, max player intrinsics values are also 9 
@struct.dataclass
class Inventory:
    wood: int = 0
    stone: int = 0
    coal: int = 0
    iron: int = 0
    diamond: int = 0
    sapling: int = 0
    wood_pickaxe: int = 0
    stone_pickaxe: int = 0
    iron_pickaxe: int = 0
    wood_sword: int = 0
    stone_sword: int = 0
    iron_sword: int = 0

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

#when indexing an enum make sure to use .value

#Here are example docstrings:

def task_is_done(inventory, inventory_diff, closest_blocks, closest_blocks_prev, player_intrinsics, player_intrinsics_diff, achievements, n):
    \"\"\"
    Determines whether Task `$db.current.skill_name$` is complete.
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
        achievements (jnp.ndarray): A 1D array (22,) of achievements, where each element is an boolean indicating the corresponding achievement has been completed.
        n (int): The number of times this skill has been applied/attempted

    Returns:
        bool: True if complete (i.e., $db.current.skill_with_consumption.completion_criteria$), False otherwise.
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
        achievements_diff (jnp.ndarray): A 1D array (22,) of achievements, where each element is an boolean indicating whether the achievement was completed in the last timestep. If the achievement was already completed previously, it will not indicate the achievement was completed again.

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
The task network number should be num_skills since we're creating a new skill and the networks are zero indexed.
Do not change the function signature or the docstrings. Do not make any assumptions beyond the information given to you. 
The code you write should be able to be jax compiled, no if statements.
No need to retype BlockType, Inventory, and Achievement they will be provided in the environment.
No need to add coefficents to rewards, for example, no need for 10 * inventory_diff.*, just use the raw values.
Return all three functions in a single code block, don't seperate it into 3.
No need to return the docstrings.
Your code will be pasted into a file that already has the following imports. Do not add any additional imports.
from craftax.craftax_classic.constants import *
from craftax.craftax_classic.envs.craftax_state import Inventory
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
