from flowrl.llm.craftax_classic.compose_prompts import ComposeReasoningPrompt
from flowrl.llm.craftax_classic.post_processing import *





def return_prompts(LLM_API_FUNCTION_GPT4):
    inventory_prompts = {}
    inventory_prompts["predict_item_count"] = {
        "prompt": """
        
Player Status:
```
$db.current.status$
```

Instruction Manual:
```
$db.manual$
```

Inventory Before Task:
```
$db.past_inventory$
```

State Information
```
$db.example_trajectory$
```


Subtask Completed:
```
$db.current.subtask_name$
```

Carefully review the trajectory data and count:

The EXACT total amount of each resource CONSUMED (all "Lost" items) during this subtask.

Count each resource separately for each action
Add them up to get the total for each resource type
Double-check your math


The EXACT total amount of each resource GAINED during this subtask.

Count each resource separately for each action
Add them up to get the total for each resource type
Double-check your math



When counting resources, create a running tally for each resource type and verify the sum before providing your final answer.

# Formatting
Finally, complete the following Json dictionary as your output.
```json
{
"subtask_name": , # name of the subtask
"resources_required": {}, # Dictionary mapping item names to quantities consumed, including only items listed as requirements in the manual entry for this subtask
}
```

        """,
        "dep": [],
        "compose": ComposeReasoningPrompt(),
        "query": LLM_API_FUNCTION_GPT4,
        "after_query": InventoryAfterQuery(),
    }
    inventory_prompts["predict_missing_items"] = {
        "prompt": """
        
Player Status:
```
$db.current.status$
```

Instruction Manual:
```
$db.manual$
```

Subtask Completed:
```
$db.current.subtask_name$
```

Inventory Before Task:
```
$db.current.inventory$
```

Resources Required:
```
$db.current.resources_required$
```



Carefully review the inventory before the task and the resources required for the subtask. Then, answer the following questions:

Are any resources required for the subtask missing from the inventory before the task? If so, which resources are missing and how many of each are needed?


# Formatting
Finally, complete the following Json dictionary as your output.
```json
{
"subtask_name": , # name of the subtask
"missing_resources": {}, # Dictionary mapping item names to number missing, including only items listed as requirements in the manual entry for this subtask
}
```

        """,
        "dep": ["predict_item_count"],
        "compose": ComposeReasoningPrompt(),
        "query": LLM_API_FUNCTION_GPT4,
        "after_query": MissingItemsAfterQuery(),
    }

    for _, v in inventory_prompts.items():
        v["prompt"] = v["prompt"].strip()
    return inventory_prompts
