from agentkit import after_query as aq
from agentkit import Graph, SimpleDBNode
from flowrl.llm.compose_prompts import ComposeReasoningPrompt
import json


ALLOWED_GAIN_TYPES = {"inventory", "achievement", "level", "ephemeral"}


def normalize_gain_schema(raw_gain: dict, error_cls, context: str = "gain"):
    """Convert gain entries into a normalized structured format.

    Args:
        raw_gain: Dict mapping gain key -> value (string or dict)
        error_cls: Exception class to raise on validation errors
        context: Human-readable context for error messages

    Returns:
        Dict where each value is `{type, expression, description}`
    """
    if raw_gain is None:
        raise error_cls(f"{context}: expected a dictionary of gains, received None")

    if not isinstance(raw_gain, dict):
        raise error_cls(f"{context}: expected gains as a dictionary, received {type(raw_gain).__name__}")

    normalized = {}
    for key, value in raw_gain.items():
        if not isinstance(key, str) or not key:
            raise error_cls(f"{context}: gain keys must be non-empty strings, found {repr(key)}")

        entry_type = "inventory"
        expression = None
        description = None

        if isinstance(value, dict):
            entry_type = value.get("type", "inventory")
            expression = value.get("expression")
            description = value.get("description")
        else:
            expression = value

        if isinstance(entry_type, str):
            entry_type = entry_type.lower()
        else:
            raise error_cls(f"{context}: gain '{key}' has non-string type {entry_type}")

        if entry_type not in ALLOWED_GAIN_TYPES:
            raise error_cls(
                f"{context}: gain '{key}' has invalid type '{entry_type}'. Allowed: {sorted(ALLOWED_GAIN_TYPES)}"
            )

        if expression is None or expression == "":
            raise error_cls(f"{context}: gain '{key}' must define an expression")

        if not isinstance(expression, str):
            expression = str(expression)

        if entry_type == "achievement" and not key.startswith("achievement:"):
            raise error_cls(
                f"{context}: achievement gains must use 'achievement:NAME' keys (got '{key}')"
            )
        if entry_type == "level" and not key.startswith("level:"):
            raise error_cls(
                f"{context}: level gains must use 'level:STAT' keys (got '{key}')"
            )

        normalized[key] = {
            "type": entry_type,
            "expression": expression,
        }
        if description:
            normalized[key]["description"] = description

    if not normalized:
        raise error_cls(f"{context}: at least one gain must be provided")

    return normalized


class TaskAfterQuery(aq.JsonAfterQuery):

    def __init__(self):
        super().__init__()
        self.type = dict
        self.required_keys = [
            "skill_name",
            "description",
            #"requirements",
            "gain",
        ]
        # self.length = len(self.keys)

    def validate_unique_gains(self, parsed_answer):
        """
        Validate that at least one gain key is not already produced by existing skills.
        Returns (is_valid, conflicting_gains, existing_gains)
        """
        # Get existing skills from database
        existing_skills = self.node.db.get("skills", {})
        
        # Collect all existing gains from previous skills
        existing_gains = set()
        for skill_name, skill_data in existing_skills.items():
            skill_with_consumption = skill_data.get("skill_with_consumption", {})
            gain = skill_with_consumption.get("gain", {})
            if isinstance(gain, dict):
                existing_gains.update(gain.keys())
            elif isinstance(gain, str) and gain:  # Handle legacy string gains
                existing_gains.add(gain)
        
        # Check new skill's gains
        new_gain = parsed_answer.get("gain", {})
        if isinstance(new_gain, dict):
            new_gains = set(new_gain.keys())
        elif isinstance(new_gain, str):
            new_gains = {new_gain} if new_gain else set()
        else:
            new_gains = set()
        
        # Find overlapping gains
        conflicting_gains = new_gains & existing_gains
        
        # Valid if there's at least one unique gain
        is_valid = len(new_gains - existing_gains) > 0
        
        return is_valid, conflicting_gains, existing_gains

    def post_process(self):

        parsed_answer = self.parse_json()[-1]

        # Normalize gain schema to structured format
        try:
            parsed_answer["gain"] = normalize_gain_schema(
                parsed_answer.get("gain", {}),
                RestartGraphException,
                context="next_task.gain",
            )
        except RestartGraphException as exc:
            raise RestartGraphException(str(exc))

        # Validate that at least one gain is unique
        is_valid, conflicting_gains, existing_gains = self.validate_unique_gains(parsed_answer)

        if not is_valid:
            print(f"All gains already exist in previous skills: {conflicting_gains}")
            print(f"Existing gains from previous skills: {existing_gains}")
            print("Restarting graph processing to generate a skill with novel gains...")
            
            # Raise custom exception to restart the entire graph
            raise RestartGraphException(
                f"All gains {list(conflicting_gains)} already exist in previous skills. Need at least one unique gain. Existing gains: {list(existing_gains)}. The skill you proposed already exists, propose another one."
            )

        self.node.db["current"]["skill_name"] = parsed_answer["skill_name"]
        self.node.db["current"]["skill"] = parsed_answer


class RestartGraphException(Exception):
    """Custom exception to signal that the graph should be restarted"""
    pass


class SubtaskAfterQuery(aq.JsonAfterQuery):

    def __init__(self):
        super().__init__()
        self.type = dict
        self.required_keys = [
            "skill_name",
            "requirements",
            "gain",
        ]
        # self.length = len(self.keys)

    def validate_requirements_against_existing_gains(self, parsed_answer):
        """
        Validate that all keys in requirements/consumption exist as gains in previous skills.
        Returns (is_valid, invalid_keys, available_gains)
        """
        # Get existing skills from database
        existing_skills = self.node.db.get("skills", {})
        
        # Collect all available gains from existing skills
        available_gains = set()
        for skill_name, skill_data in existing_skills.items():
            skill_with_consumption = skill_data.get("skill_with_consumption", {})
            gain = skill_with_consumption.get("gain", {})
            if isinstance(gain, dict):
                available_gains.update(gain.keys())
            elif isinstance(gain, str) and gain:
                available_gains.add(gain)

        # Check requirements
        requirements = parsed_answer.get("requirements", {})
        consumption = parsed_answer.get("consumption", {})
        
        # Find invalid keys
        invalid_req_keys = set(requirements.keys()) - available_gains
        invalid_cons_keys = set(consumption.keys()) - available_gains
        
        all_invalid_keys = invalid_req_keys | invalid_cons_keys
        
        is_valid = len(all_invalid_keys) == 0
        
        return is_valid, all_invalid_keys, available_gains

    def verify_skill_frontier_compliance(self, parsed_answer):
        """
        Verify that the skill is both novel and feasible using symbolic state verification.
        Only applies to environments with frontier-based prompts enabled.
        Returns (is_novel, is_feasible, verification_message)
        """
        # Check if this environment uses frontier-based prompts
        try:
            # Look for indication of frontier-based prompts in the database
            if "frontier_summary" not in self.node.db:
                return True, True, "Frontier verification disabled for this environment"

            from flowrl.llm.craftax.symbolic_state import verify_skill

            # Get existing skills from database
            existing_skills = self.node.db.get("skills", {})

            # Determine inventory capacity based on frontier summary content
            frontier_summary = self.node.db.get("frontier_summary", "")
            if "disabled" in frontier_summary.lower():
                return True, True, "Frontier verification disabled"

            # Default to 99 for full Craftax, could be refined based on environment detection
            max_capacity = 99

            # Extract skill name from the current skill data
            skill_name = parsed_answer.get("skill_name", "unknown_skill")

            # Use the existing verify_skill function
            is_novel, is_feasible = verify_skill(skill_name, parsed_answer, existing_skills, max_capacity)

            # Generate verification message
            if is_feasible and is_novel:
                message = f"✓ Skill '{skill_name}' is feasible and novel - ready for training"
            elif is_feasible and not is_novel:
                message = f"✗ Skill '{skill_name}' is feasible but NOT novel - already achievable by existing skills"
            elif not is_feasible and is_novel:
                message = f"⚠ Skill '{skill_name}' is novel but NOT feasible - preconditions cannot be satisfied {skill_name, parsed_answer, existing_skills, max_capacity}"
            else:
                message = f"✗ Skill '{skill_name}' is neither feasible nor novel - should not be trained"

            return is_novel, is_feasible, message

        except Exception as e:
            return True, True, f"Frontier verification error: {str(e)} - proceeding with training"

    def post_process(self):
        parsed_answer = self.parse_json()[-1]

        # Normalize gain schema before validation
        try:
            parsed_answer["gain"] = normalize_gain_schema(
                parsed_answer.get("gain", {}),
                RestartGraphException,
                context="next_subtask.gain",
            )
        except RestartGraphException as exc:
            raise RestartGraphException(str(exc))

        # Validate requirements/consumption against existing skill gains
        is_valid, invalid_keys, available_gains = self.validate_requirements_against_existing_gains(parsed_answer)

        if not is_valid:
            print(f"Invalid keys found in requirements/consumption: {invalid_keys}")
            print(f"Available gains from previous skills: {available_gains}")
            print("Restarting graph processing...")

            # Raise custom exception to restart the entire graph
            raise RestartGraphException(
                f"Keys {list(invalid_keys)} in requirements/consumption are not available as gains from previous skills. Available gains: {list(available_gains)}"
            )

        # Verify skill novelty and feasibility using symbolic state system
        is_novel, is_feasible, verification_msg = self.verify_skill_frontier_compliance(parsed_answer)
        print(f"Frontier verification: {verification_msg}")

        # For now, we'll log the verification but not block training
        # In the future, could add logic to restart if skill is not novel/feasible
        # if not (is_novel and is_feasible):
        #     print("Skill failed frontier verification - restarting...")
        #     raise RestartGraphException(verification_msg)

        # Store verification results
        parsed_answer["frontier_verification"] = {
            "is_novel": is_novel,
            "is_feasible": is_feasible,
            "message": verification_msg
        }

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
    pickaxe: int
    sword: int
    bow: int
    arrows: int
    armour: jnp.ndarray
    torches: int
    ruby: int
    sapphire: int
    potions: jnp.ndarray
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

#Here are example docstrings:

def task_is_done(inventory, inventory_diff, closest_blocks, closest_blocks_prev, player_intrinsics, player_intrinsics_diff, achievements, n):
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
        player_intrinsics (jnp.ndarray): An len 5 array representing the player's health, food, drink, energy, and mana levels
        player_intrinsics_diff (jnp.ndarray): An len 5 array representing the change in the player's health, food, drink, energy, and mana levels
        achievements (jnp.ndarray): A 1D array (67,) of achievements, where each element is an boolean indicating the corresponding achievement has been completed.
        n (int): The number of times this skill has been applied/attempted

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
        player_intrinsics_diff (jnp.ndarray): An len 5 array representing the change in the player's health, food, drink, energy, and mana levels
        achievements_diff (jnp.ndarray): A 1D array (67,) of achievements, where each element is a boolean indicating whether the corresponding achievement has been completed in the last timestep.

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


class SkillUpdateAfterQuery(aq.JsonAfterQuery):
    """Handles results from update_skill_from_trajectory prompt"""
    
    def __init__(self):
        super().__init__()
        self.type = dict
        self.required_keys = ["skill_name", "updated_requirements", "updated_gain"]

    def validate_requirements_against_existing_gains(self, parsed_answer):
        """
        Validate that all keys in updated_requirements/updated_consumption exist as gains in previous skills.
        Returns (is_valid, invalid_keys, available_gains)
        """
        # Get existing skills from database
        existing_skills = self.node.db.get("skills", {})
        
        # Collect all available gains from existing skills
        available_gains = set()
        for skill_name, skill_data in existing_skills.items():
            skill_with_consumption = skill_data.get("skill_with_consumption", {})
            gain = skill_with_consumption.get("gain", {})
            if isinstance(gain, dict):
                available_gains.update(gain.keys())
            elif isinstance(gain, str) and gain:
                available_gains.add(gain)
        
        # Check updated requirements and consumption
        updated_requirements = parsed_answer.get("updated_requirements", {})
        updated_consumption = parsed_answer.get("updated_consumption", {})
        
        # Find invalid keys
        invalid_req_keys = set(updated_requirements.keys()) - available_gains
        invalid_cons_keys = set(updated_consumption.keys()) - available_gains
        
        all_invalid_keys = invalid_req_keys | invalid_cons_keys
        
        is_valid = len(all_invalid_keys) == 0
        
        return is_valid, all_invalid_keys, available_gains

    def post_process(self):
        parsed_answer = self.parse_json()[-1]
        
        # Normalize gain schema for updates
        try:
            parsed_answer["updated_gain"] = normalize_gain_schema(
                parsed_answer.get("updated_gain", {}),
                aq.AfterQueryError,
                context="update_skill_from_trajectory.updated_gain",
            )
        except aq.AfterQueryError as exc:
            raise aq.AfterQueryError("Invalid gain schema", str(exc))

        # Validate updated requirements/consumption against existing skill gains
        is_valid, invalid_keys, available_gains = self.validate_requirements_against_existing_gains(parsed_answer)
        
        if not is_valid:
            print(f"Invalid keys found in updated requirements/consumption: {invalid_keys}")
            print(f"Available gains from previous skills: {available_gains}")
            print("Restarting update_skill_from_trajectory node...")
            
            # Raise AfterQueryError to restart just this node
            raise aq.AfterQueryError(
                "Invalid updated requirements/consumption keys", 
                f"Keys {list(invalid_keys)} in updated requirements/consumption are not available as gains from previous skills. Available gains: {list(available_gains)}"
            )
        
        # Store the results for use by flow.py
        self.node.db["current"]["skill_update_results"] = parsed_answer


class KnowledgeBaseUpdateAfterQuery(aq.JsonAfterQuery):
    """Handles results from propose_knowledge_base_updates prompt"""
    
    def __init__(self):
        super().__init__()
        self.type = dict
        self.required_keys = ["proposed_updates"]

    def validate_knowledge_base_paths(self, parsed_answer):
        """Validate that all paths in proposed_updates are valid knowledge base paths"""
        kb = self.node.db.get("knowledge_base", {})
        
        for update in parsed_answer.get("proposed_updates", []):
            path = update.get("path", [])
            if not path:
                raise aq.AfterQueryError("Invalid path", "Empty path in knowledge base update proposal")
            
            # Navigate through the knowledge base using the path
            current = kb
            for i, key in enumerate(path):
                if not isinstance(current, dict) or key not in current:
                    raise aq.AfterQueryError("Invalid path", f"Path {path} is invalid at key '{key}' (position {i})")
                current = current[key]
            
            # Check that the final item is a list (since KB entries are requirement lists)
            if not isinstance(current, list):
                raise aq.AfterQueryError("Invalid path", f"Path {path} does not point to a list, got {type(current)}")

    def apply_knowledge_base_updates(self, parsed_answer):
        """Apply validated updates directly to the knowledge base"""
        kb = self.node.db.get("knowledge_base", {})
        updates_applied = []
        
        for update in parsed_answer.get("proposed_updates", []):
            path = update.get("path", [])
            updated_requirements = update.get("updated_requirements", [])
            reason = update.get("reason_for_update", "")
            
            # Navigate to the target location in the knowledge base
            current = kb
            for key in path[:-1]:  # All keys except the last one
                current = current[key]
            
            # Get the final key and update the requirements list
            final_key = path[-1]
            old_requirements = current[final_key].copy()  # Keep a copy of old requirements
            current[final_key] = updated_requirements
            
            updates_applied.append({
                "path": path,
                "old_requirements": old_requirements,
                "new_requirements": updated_requirements,
                "reason": reason
            })
            
            print(f"Updated KB at {' -> '.join(path)}")
            print(f"  Old: {old_requirements}")
            print(f"  New: {updated_requirements}")
            print(f"  Reason: {reason}")
        
        return updates_applied

    def post_process(self):
        parsed_answer = self.parse_json()[-1]
        
        # Validate knowledge base paths
        self.validate_knowledge_base_paths(parsed_answer)
        
        # Apply updates directly to the knowledge base
        updates_applied = self.apply_knowledge_base_updates(parsed_answer)
        
        # Store record of what was updated
        self.node.db["current"]["kb_updates_applied"] = updates_applied
