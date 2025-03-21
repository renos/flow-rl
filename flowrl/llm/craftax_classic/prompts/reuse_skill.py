from flowrl.llm.compose_prompts import ComposeReasoningPrompt
from flowrl.llm.craftax_classic.after_queries import *


def return_prompts(LLM_API_FUNCTION_GPT4):
    inventory_prompts = {}
    inventory_prompts["reuse_or_generate"] = {
        "prompt": """
Consider the missing items and List of existing skills


Missing Resources
```
$db.missing_resource$
```

Existing Skills
```
$db.skills$
```


## Review Skills
In plain text, review existing skills and determine whether one exists that can complete the current subtask. A skill should only be reused if it fits the current subtask exactly.

# Formatting
Finally, complete the following Json dictionary as your output.
```json
{
    "reuse_skill": # (bool) true if we are reusing a skill to complete the task and false if we are making a new skill to complete the task.
    "reused_skill": # (string) name of the skill we are reusing from the existing skills, write NA if no skill is being reused.
}
```
        """,
        "dep": [],
        "compose": ComposeReasoningPrompt(),
        "query": LLM_API_FUNCTION_GPT4,
        "after_query": ReuseSkill(),
    }
    inventory_prompts["reuse_skill_task_reasoning"] = {
        "prompt": """
    Given the following information,
    Missing Resources
    ```
    $db.missing_resource$
    ```
    Skill to Reuse:
    ```
    $db.current.skill_reused$
    ```

    ## Review requirements
    Determine whether the task completion function needs to change for the current subtask.

    Finally, complete the following Json dictionary for your output.
    ```json
    {
    "completion_changes": # (string) List how the completion function of the skill should change.
    }
    ```
        """,
        "dep": [],
        "after": ["reuse_or_generate"],
        "compose": ComposeReasoningPrompt(),
        "query": LLM_API_FUNCTION_GPT4,
        "after_query": UpdatedDoneCondAfterQuery(),
    }

    inventory_prompts["reuse_skill_coding"] = {
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

Skill to be reused:
```json
$db.current.skill_reused$
```

Given the above documentations and existing skills, reuse the `task_is_done`, `task_reward`, and task_network_number function for the subtask `$db.current.skill_reused_name$` with the following details:
Missing Resources
```
$db.missing_resource$
```
```json
$db.current.reward$
```

The task network number should be the same as the old one since we're reusing an old skill and so the networks will be the same.
The reward function should not change since we're reusing the skill, just copy over the one given.
Do not change the function signature or the docstrings. Do not make any assumptions beyond the information given to you. 
The code you write should be able to be jax compiled, no if statements.
No need to retype BlockType, Inventory, and Achievement they will be provided in the environment.
Return all three functions in a single code block, don't seperate it into 3.
Return the functions with the 
        """,
        "dep": [],
        "after": ["reuse_or_generate", "reuse_skill_task_reasoning"],
        "compose": ComposeReasoningPrompt(),
        "query": LLM_API_FUNCTION_GPT4,
        # TODO: Add after_query
    }
    for _, v in inventory_prompts.items():
        v["prompt"] = v["prompt"].strip()
    return inventory_prompts
