from flowrl.llm.compose_prompts import ComposeReasoningPrompt
from flowrl.llm.craftax.after_queries import *


def return_prompts(LLM_API_FUNCTION_GPT4):
    inventory_prompts = {}

    inventory_prompts["update_skill_from_trajectory"] = {
        "prompt": """
You need to update a skill based on its execution trajectory.

Current Skill:
```
$db.current.skill_with_consumption$
```

Existing Skills (for requirements validation):
```
$db.skills_without_code$
```

Trajectory Data (Multiple Successful Executions):
```
$db.example_trajectories$
```

## Task

Analyze the multiple trajectories to determine what the skill actually required, consumed, and gained, then express this in terms of `n` (the number of times the skill is applied).

Each trajectory shows a specific instance (e.g. n=1), but you need to infer the general pattern. Only include gains that appear consistently across all trajectories.

**HOW TO READ THE TRAJECTORY:**
The trajectory contains:
1. **ACTION SUMMARY**: Shows the count of each action type performed during the entire trajectory
2. **TIMESTEPS WITH CHANGES**: Shows only the specific moments where inventory, achievements, or levels changed

**YOUR ROLE IN TRAJECTORY ANALYSIS:**
Your job is to **correct factual errors** in requirements/consumption based on what actually happened in the trajectory, NOT to judge whether requirements are useful.

- **DO update** if the trajectory DISPROVES a requirement/consumption amount:
  - Example: Requirement says "wood: lambda n: 4*n" but trajectory shows crafting table only consumed 1 wood → UPDATE to "lambda n: 1*n"
  - Example: Consumption says "stone: lambda n: 5*n" but trajectory shows only 3 stone was consumed → UPDATE to "lambda n: 3*n"

- **DO NOT remove** requirements just because they weren't visibly used in this single trajectory:
  - Example: Requirement includes "wood_pickaxe" but trajectory doesn't explicitly show pickaxe usage → KEEP the requirement (it may still be useful)
  - The initial skill definition already judged what's useful - trust that unless the trajectory proves it wrong

- **DO update** gains to match what was actually achieved:
  - If trajectory shows achievement unlocked, level gained, or items obtained, ensure gains reflect this

**IMPORTANT CONSTRAINTS:**
- Requirements are a SUPERSET of consumption: requirements include everything needed (both consumed and non-consumed resources), while consumption only includes what gets used up.
- Each value in requirements/consumption should be written as a Python lambda function string that takes n and returns the amount needed, in the form: "lambda n: a*n + b", where:
  - a = amount of resource consumed PER unit of gain (scales with n)
  - b = amount of resource required but NOT consumed (fixed amount regardless of n)
  - Ask yourself: "Does this requirement scale with the number of times I apply the skill?"
    - If YES (scales with n): use "lambda n: a*n + 0" format
    - If NO (fixed amount): use "lambda n: 0*n + b" format
  - **EXCEPTION**: "level:player_level" (floor constraints) use a list format like [0] or [1], NOT a lambda expression
- Requirements do not support 'or'
- Each key in requirements/consumption must be a key in the gain of an existing skill.

Update the skill's requirements and gain based on what the trajectory revealed.
- Requirements/consumption follow the lambda format above
- Gains must use the structured schema: each key maps to an object with `type`, `expression`, and optional `description`
- Use `type="inventory"` for tangible items, `type="achievement"` with keys like `achievement:PLACE_FURNACE`, `type="level"` with keys like `level:player_strength`, and `type="ephemeral"` for transient capabilities
- For tiered tools (`pickaxe`, `sword`) treat the gain as an absolute tier index rather than a count; set the expression to the resulting tier (`lambda n: 2` for stone, `lambda n: 3` for iron) and avoid scaling with `n`
- **Special case**: `level:player_level` in requirements is a list `[0]`, `[1]`, etc. indicating which floors the skill can be performed on. Empty list `[]` means all floors. If the trajectory shows the skill executed on a floor not in the list, add that floor to the list. Otherwise, keep the list unchanged.

# Formatting
```json
{
"skill_name": "", # name of the skill
"updated_requirements": {},  # total amount needed available using "lambda n: a*n + b" format. Each key must exactly match the key of a gain of a previous skill.
"updated_consumption": {}, # amount consumed using "lambda n: a*n + b" format. Each key must exactly match the key of a gain of a previous skill.
"updated_gain": { # structured gains following the same schema as new skills
  "gain_key": {
    "type": "inventory | achievement | level | stat | ephemeral",
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
        "after_query": SkillUpdateAfterQuery(),
    }

    for _, v in inventory_prompts.items():
        v["prompt"] = v["prompt"].strip()
    return inventory_prompts
