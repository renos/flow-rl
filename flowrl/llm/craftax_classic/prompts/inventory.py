from flowrl.llm.compose_prompts import ComposeReasoningPrompt
from flowrl.llm.craftax_classic.after_queries import *


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
$db.skills$
```

Trajectory Data:
```
$db.example_trajectory$
```

## Task

Analyze the trajectory to determine what the skill actually required, consumed, and gained, then express this in terms of `n` (the number of times the skill is applied).

The trajectory shows a specific instance (e.g. n=1), but you need to infer the general pattern.

**IMPORTANT CONSTRAINTS:**
- Requirements are a SUPERSET of consumption: requirements include everything needed (both consumed and non-consumed resources), while consumption only includes what gets used up.
- Each value in requirements/consumption should be written as a Python lambda function string that takes n and returns the amount needed, in the form: "lambda n: a*n + b", where:
  - a = amount of resource consumed PER unit of gain (scales with n)
  - b = amount of resource required but NOT consumed (fixed amount regardless of n)
  - Ask yourself: "Does this requirement scale with the number of times I apply the skill?"
    - If YES (scales with n): use "lambda n: a*n + 0" format
    - If NO (fixed amount): use "lambda n: 0*n + b" format
- Requirements do not support 'or'
- Each key in requirements/consumption must be a key in the gain of an existing skill. 
  
Update the skill's requirements and gain as lambda functions based on what the trajectory revealed.

# Formatting
```json
{
"skill_name": "", # name of the skill
"updated_requirements": {},  # total amount needed available using "lambda n: a*n + b" format. Each key must exactly match the key of a gain of a previous skill.
"updated_consumption": {}, # amount consumed using "lambda n: a*n + b" format. Each key must exactly match the key of a gain of a previous skill.
"updated_gain": {} # a dictionary of what is gained by applying the skill. The gain for the skill goal should be n.
}
```

        """,
        "dep": [],
        "compose": ComposeReasoningPrompt(),
        "query": LLM_API_FUNCTION_GPT4,
        "after_query": SkillUpdateAfterQuery(),
    }
    
    inventory_prompts["propose_knowledge_base_updates"] = {
        "prompt": """
You need to propose which parts of the knowledge base should be updated based on trajectory analysis.

Knowledge Base:
```
$db.knowledge_base$
```

Trajectory Data:
```
$db.example_trajectory$
```

Current Skill:
```
$db.current.skill_with_consumption$
```

## Task

Look at the knowledge base structure and propose which specific entries/fields should be updated based on what was VERIFIED from the trajectory execution.

The knowledge base contains requirement lists with items marked as "ASSUMPTION" or "VERIFIED". Based on the trajectory, propose updates where:

1. ASSUMPTIONS that were confirmed by the trajectory become "VERIFIED: condition"
2. ASSUMPTIONS that were proven FALSE by the trajectory should be REMOVED
3. ASSUMPTIONS that cannot be verified from this trajectory MUST remain "ASSUMPTION: condition" 
4. New requirements discovered from the trajectory are added as "VERIFIED: condition"

**CRITICAL**: Only make changes when you have clear evidence from the trajectory:
- Change ASSUMPTION to VERIFIED if trajectory confirms it's true
- REMOVE assumptions if trajectory proves they're false
- KEEP assumptions unchanged if trajectory provides no evidence either way

Requirements should be in the format: "VERIFIED: condition" or "ASSUMPTION: condition"

# Formatting
```json
{
"proposed_updates": [
    {
        "path": ["key1", "subkey", "field"], # Path to the field in the knowledge base
        "updated_requirements": [], # Complete updated list of requirements (verified + remaining assumptions)
        "reason_for_update": "" # What the trajectory showed that confirms, disproves, or leaves unchanged
    }
]
}
```

        """,
        "dep": ["update_skill_from_trajectory"],
        "compose": ComposeReasoningPrompt(),
        "query": LLM_API_FUNCTION_GPT4,
        "after_query": KnowledgeBaseUpdateAfterQuery(),
    }

    for _, v in inventory_prompts.items():
        v["prompt"] = v["prompt"].strip()
    return inventory_prompts
