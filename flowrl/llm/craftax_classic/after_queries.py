from agentkit import after_query as aq
from agentkit import Graph, SimpleDBNode
from flowrl.llm.compose_prompts import ComposeReasoningPrompt
import json


class TaskAfterQuery(aq.JsonAfterQuery):

    def __init__(self):
        super().__init__()
        self.type = dict
        self.required_keys = [
            "skill_name",
            "description",
            "requirements",
            "gain",
            "completion_criteria",
        ]
        # self.length = len(self.keys)

    def post_process(self):

        parsed_answer = self.parse_json()[-1]

        self.node.db["current"]["skill_name"] = parsed_answer["skill_name"]
        self.node.db["current"]["task"] = parsed_answer


class SubtaskAfterQuery(aq.JsonAfterQuery):

    def __init__(self):
        super().__init__()
        self.type = dict
        self.required_keys = [
            "skill_name",
            "requirements",
            "consumption",
            "gain",
        ]
        # self.length = len(self.keys)

    def post_process(self):

        parsed_answer = self.parse_json()[-1]

        self.node.db["current"]["skill_with_consumption"] = parsed_answer


class ReuseSkill(aq.JsonAfterQuery):

    def __init__(self):
        super().__init__()
        self.type = dict
        self.required_keys = [
            "reuse_skill",
            "reused_skill",
        ]
        # self.length = len(self.keys)

    def post_process(self):

        parsed_answer = self.parse_json()[-1]

        self.node.db["current"]["subtask"]["reuse_skill"] = parsed_answer["reuse_skill"]

        # add the relevant branch
        if parsed_answer["reuse_skill"]:
            # self.node.graph.skip_nodes_temporary([self.node.db["prompts"]["create_skill_reward_reasoning"]["prompt"], self.node.db["prompts"]["create_skill_coding"]["prompt"]])
            # self.node.graph.add_temporary_node(SimpleDBNode(questions, questions, self.node.graph, self.node.query_llm, ComposePlannerPrompt(), self.node.db))
            # self.node.graph.add_temporary_node(SimpleDBNode(questions, questions, self.node.graph, self.node.query_llm, ComposePlannerPrompt(), self.node.db))
            query = "reuse"
            self.node.db["current"]["skill_reused"] = self.node.db["skills"][
                parsed_answer["reused_skill"]
            ]
            self.node.db["current"]["skill_reused_name"] = parsed_answer["reused_skill"]


class ReuseOrGenerateNewSkill(aq.JsonAfterQuery):

    def __init__(self):
        super().__init__()
        self.type = dict
        self.required_keys = [
            "reuse_skill",
            "reused_skill",
        ]
        # self.length = len(self.keys)

    def post_process(self):

        parsed_answer = self.parse_json()[-1]

        self.node.db["current"]["subtask"]["reuse_skill"] = parsed_answer["reuse_skill"]

        # add the relevant branch
        if parsed_answer["reuse_skill"]:
            # self.node.graph.skip_nodes_temporary([self.node.db["prompts"]["create_skill_reward_reasoning"]["prompt"], self.node.db["prompts"]["create_skill_coding"]["prompt"]])
            # self.node.graph.add_temporary_node(SimpleDBNode(questions, questions, self.node.graph, self.node.query_llm, ComposePlannerPrompt(), self.node.db))
            # self.node.graph.add_temporary_node(SimpleDBNode(questions, questions, self.node.graph, self.node.query_llm, ComposePlannerPrompt(), self.node.db))
            query = "reuse"
            self.node.db["current"]["skill_reused"] = self.node.db["skills"][
                parsed_answer["reused_skill"]
            ]
            self.node.db["current"]["skill_reused_name"] = parsed_answer["reused_skill"]
        else:
            # self.node.db["prompts"]["reflection_skip_questions"]
            # self.node.graph.skip_nodes_temporary([self.node.db["prompts"]["reuse_skill_task_reasoning"]["prompt"], self.node.db["prompts"]["reuse_skill_coding"]["prompt"]])
            query = "create"
        prompts_to_add = {}
        for key, value in self.node.db["temp_prompts"].items():
            if query in key:
                prompts_to_add[key] = value

        temp_edge_list = []
        for _, node_info in prompts_to_add.items():
            key = node_prompt = node_info["prompt"]
            node = SimpleDBNode(
                key,
                node_prompt,
                self.node.graph,
                node_info["query"],
                node_info["compose"],
                self.node.db,
                after_query=(
                    node_info["after_query"]
                    if "after_query" in node_info.keys()
                    else None
                ),
                verbose=True,
            )
            self.node.graph.add_temporary_node(node)

            if "shorthand" in node_info.keys() and node_info["shorthand"] is not None:
                self.node.db["shorthands"][key] = node_info["shorthand"]

            for dependency in node_info["after"]:
                if dependency in prompts_to_add:
                    dependency_name = prompts_to_add[dependency]["prompt"]
                elif dependency in self.node.db["prompts"]:
                    dependency_name = self.node.db["prompts"][dependency]["prompt"]
                temp_edge_list.append((dependency_name, key))

        for edge in temp_edge_list:
            print(edge)
            self.node.graph.add_edge_temporary(*edge)
        return True


class FactorsAfterQuery(aq.JsonAfterQuery):

    def __init__(self):
        super().__init__()
        self.type = dict
        # self.required_keys = [
        #     "factors",
        # ]
        # self.length = len(self.keys)

    def post_process(self):

        parsed_answer = self.parse_json()[-1]

        self.node.db["current"]["factors"] = parsed_answer


class InventoryAfterQuery(aq.JsonAfterQuery):

    def __init__(self):
        super().__init__()
        self.type = dict
        self.required_keys = ["subtask_name", "resources_required"]
        # self.length = len(self.keys)

    def post_process(self):

        parsed_answer = self.parse_json()[-1]
        subtask_name = parsed_answer["subtask_name"]
        resources_required = parsed_answer["resources_required"]

        self.node.db["knowledge_base"][subtask_name] = {
            "resources_required": resources_required
        }
        self.node.db["current"]["resources_required"] = resources_required


class MissingItemsAfterQuery(aq.JsonAfterQuery):
    def __init__(self):
        super().__init__()
        self.type = dict
        self.required_keys = ["subtask_name", "missing_resources"]

    def post_process(self):
        parsed_answer = self.parse_json()[-1]
        subtask_name = parsed_answer["subtask_name"]
        missing_resources = parsed_answer["missing_resources"]
        self.node.db["current"]["missing_resources"] = missing_resources


class DensifyAfterQuery(aq.JsonAfterQuery):

    def __init__(self):
        super().__init__()
        self.type = dict
        self.required_keys = [
            "sparse_reward_only_function",
            "dense_reward_function",
        ]
        # self.length = len(self.keys)

    def post_process(self):

        parsed_answer = self.parse_json()[-1]

        self.node.db["current"]["dense_reward_factor"] = parsed_answer[
            "dense_reward_function"
        ]
        self.node.db["current"]["reward"] = parsed_answer["sparse_reward_only_function"]


class UpdatedDoneCondAfterQuery(aq.JsonAfterQuery):

    def __init__(self):
        super().__init__()
        self.type = dict
        self.required_keys = [
            "completion_changes",
        ]
        # self.length = len(self.keys)

    def post_process(self):

        parsed_answer = self.parse_json()[-1]

        self.node.db["current"]["completion_changes"] = parsed_answer


# Obsolete
class AdaptiveAfterQuery(aq.JsonAfterQuery):

    def __init__(self):
        super().__init__()
        self.type = dict

    def post_process(self):

        parsed_answer = self.parse_json()[-1]

        for skill, info in parsed_answer.items():
            requirements = info["req"]
            for r in requirements:
                print(parsed_answer.keys())
                assert (
                    r in parsed_answer.keys()
                ), f"Requirement {r} not found in parsed answer"

        prompt = """
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
#Max inventory value is 9, max player intrinsics values are also 9 
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

#Here are example docstrings:

def task_is_done(inventory, inventory_diff, closest_blocks, closest_blocks_prev, player_intrinsics, achievements):
    \"\"\"
    Determines whether Task is complete.
    Do not call external functions or make any assumptions beyond the information given to you.

    Args:
        inventory (Inventory): The player's current inventory, defined in the above struct
        inventory_diff (Inventory): The player's difference in inventory, defined in the above struct
        closest_blocks (numpy.ndarray): A 3D array of shape (len(BlockType), 2, K) representing the K closest blocks of each type. Default values are (30, 30) for unseen blocks.
        #default of 30,30 if less then k seen, ordered by distance (so :,:,0 would be the closest of each block type
        closest_blocks_prev (numpy.ndarray): A 3D array of shape (len(BlockType), 2, K) representing the K closest blocks of each type from the previous timestep, 
        #default of 30,30 if less then k seen, ordered by distance (so :,:,0 would be the closest of each block type
        player_intrinsics (jnp.ndarray): An len 4 array representing the player's health, food, drink, and energy levels
        achievements_diff (jnp.ndarray): A 1D array (22,) of achievements, where each element is a boolean indicating whether the corresponding achievement has been completed in the last timestep.

    Returns:
        bool: True if Task is complete (i.e., if the inventory contains at least 8 units of wood), False otherwise.
    \"\"\"
    return TODO


def task_reward(inventory_diff, closest_blocks, player_intrinsics_diff, achievements_diff, health_penalty):
    \"\"\"
    Calculates the reward for Task based on changes in inventory and other factors.
    Do not call external functions or make any assumptions beyond the information given to you.

    Args:
        inventory_diff (Inventory): The change in the player's inventory between the current and previous timesteps, same struct as above.
        closest_blocks (numpy.ndarray): A 3D array of shape (len(BlockType), 2, K) representing the K closest blocks of each type, 
        #default of 30,30 if less then k seen, ordered by distance (so :,:,0 would be the closest of each block type. The 2 corresponds to the x and y coordinates of the block.
        closest_blocks_prev (numpy.ndarray): A 3D array of shape (len(BlockType), 2, K) representing the K closest blocks of each type from the previous timestep, 
        #default of 30,30 if less then k seen, ordered by distance (so :,:,0 would be the closest of each block type. The 2 corresponds to the x and y coordinates of the block.
        health_penalty (float): The penalty for losing health. Negative when loosing health and positive when regaining health.
        player_intrinsics_diff (jnp.ndarray): An len 4 array representing the change in the player's health, food, drink, and energy levels
        achievements_diff (jnp.ndarray): A 1D array (22,) of achievements, where each element is a boolean indicating whether the corresponding achievement has been completed in the last timestep.

    Returns:
        float: The reward for Task, calculated as a combination of the change in wood inventory and health penalty.
    \"\"\"
    return TODO + health_penalty
```

Given the above documentations, implement the `task_is_done` and `task_reward` function for the subtask `{}` with the following details:
```json
{}
```
Do not add edit the function signature or the docstrings. Do not make any assumptions beyond the information given to you. 
The code you write should be able to be jax compiled, no if statements.
No need to retype BlockType, Inventory, and Achievement they will be provided in the environment.
No need to add coefficents to rewards, for example, no need for 10 * inventory_diff.*, just use the raw values.
Keep the reward simple, do not try to over optimize. The reward can be as simple as the completion criteria.
""".strip()

        for subtask, info in parsed_answer.items():
            node_prompt = prompt.format(subtask, json.dumps(info, indent=2))
            self.node.graph.add_temporary_node(
                SimpleDBNode(
                    node_prompt,
                    node_prompt,
                    self.node.graph,
                    self.node.query_llm,
                    ComposeReasoningPrompt(),
                    self.node.db,
                    verbose=True,
                )
            )
            self.node.graph.add_edge_temporary(self.node.key, node_prompt)
