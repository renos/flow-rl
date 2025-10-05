"""Symbolic abstraction helpers for the Fabrax environment."""

from dataclasses import dataclass, field
from typing import Dict, Set, List, Optional, Tuple

from craftax.fabrax.constants import Achievement


# Fabrax uses explicit tool items (wood_pickaxe, stone_pickaxe, etc.) instead of tier indices.
TIERED_INVENTORY_ITEMS: Set[str] = set()


ACHIEVEMENT_NAME_TO_ID = {
    achievement.name.lower(): achievement.value for achievement in Achievement
}


def _achievement_key_to_id(key: str) -> Optional[int]:
    """Extract achievement id from keys like 'achievement:PLACE_ANVIL'."""
    if not isinstance(key, str):
        return None

    if ":" in key:
        _, suffix = key.split(":", 1)
    else:
        suffix = key

    return ACHIEVEMENT_NAME_TO_ID.get(suffix.lower())


@dataclass
class SymbolicState:
    """Symbolic representation of Fabrax game state."""

    inventory: Dict[str, int] = field(default_factory=dict)
    achievements: Set[int] = field(default_factory=set)

    def __hash__(self):
        return hash(
            (
                tuple(sorted(self.inventory.items())),
                tuple(sorted(self.achievements)),
            )
        )

    def __eq__(self, other):
        if not isinstance(other, SymbolicState):
            return False
        return (
            self.inventory == other.inventory
            and self.achievements == other.achievements
        )

    def copy(self):
        return SymbolicState(
            inventory=self.inventory.copy(),
            achievements=self.achievements.copy(),
        )


class SymbolicPredicate:
    """Base class for symbolic predicates over states."""

    def satisfied(self, state: SymbolicState) -> bool:
        raise NotImplementedError

    def apply_positive(self, state: SymbolicState):
        raise NotImplementedError

    def apply_negative(self, state: SymbolicState):
        raise NotImplementedError


class HasPredicate(SymbolicPredicate):
    """Predicate representing inventory availability."""

    def __init__(self, item: str, count: int):
        self.item = item
        self.count = count

    def satisfied(self, state: SymbolicState) -> bool:
        return state.inventory.get(self.item, 0) >= self.count

    def apply_positive(self, state: SymbolicState):
        current = state.inventory.get(self.item, 0)
        if self.item in TIERED_INVENTORY_ITEMS:
            state.inventory[self.item] = max(current, self.count)
        else:
            state.inventory[self.item] = current + self.count

    def apply_negative(self, state: SymbolicState):
        current = state.inventory.get(self.item, 0)
        state.inventory[self.item] = max(0, current - self.count)
        if state.inventory[self.item] == 0:
            state.inventory.pop(self.item, None)

    def __str__(self):
        return f"has({self.item}, {self.count})"

    def __hash__(self):
        return hash((self.item, self.count))

    def __eq__(self, other):
        return (
            isinstance(other, HasPredicate)
            and self.item == other.item
            and self.count == other.count
        )


class AchievedPredicate(SymbolicPredicate):
    """Predicate representing completed achievements."""

    def __init__(self, achievement_id: int):
        self.achievement_id = achievement_id

    def satisfied(self, state: SymbolicState) -> bool:
        return self.achievement_id in state.achievements

    def apply_positive(self, state: SymbolicState):
        state.achievements.add(self.achievement_id)

    def apply_negative(self, state: SymbolicState):
        state.achievements.discard(self.achievement_id)

    def __str__(self):
        return f"achieved({self.achievement_id})"

    def __hash__(self):
        return hash(self.achievement_id)

    def __eq__(self, other):
        return (
            isinstance(other, AchievedPredicate)
            and self.achievement_id == other.achievement_id
        )


@dataclass
class SkillOperator:
    """Symbolic operator describing a skill's preconditions and effects."""

    name: str
    preconditions: Set[SymbolicPredicate] = field(default_factory=set)
    effects_positive: Set[SymbolicPredicate] = field(default_factory=set)
    effects_negative: Set[SymbolicPredicate] = field(default_factory=set)

    def can_apply(self, state: SymbolicState) -> bool:
        return all(predicate.satisfied(state) for predicate in self.preconditions)

    def apply(self, state: SymbolicState) -> SymbolicState:
        new_state = state.copy()

        for effect in self.effects_negative:
            effect.apply_negative(new_state)

        for effect in self.effects_positive:
            effect.apply_positive(new_state)

        return new_state


def abstract_state(env_state) -> SymbolicState:
    """Convert raw Fabrax environment state to symbolic representation."""

    symbolic = SymbolicState()

    inventory_fields = [
        # Core Craftax Classic materials/tools
        "wood",
        "stone",
        "coal",
        "iron",
        "diamond",
        "sapling",
        "wood_pickaxe",
        "stone_pickaxe",
        "iron_pickaxe",
        "wood_sword",
        "stone_sword",
        "iron_sword",
        # Fabrax materials and products
        "copper",
        "tin",
        "sand",
        "clay",
        "limestone",
        "leather",
        "iron_bar",
        "steel_bar",
        "bronze_bar",
        "glass",
        "brick",
        "lime",
        "tar",
        "bottle",
        "lens",
        "telescope",
        "mortar",
        "fertilizer",
        "flux",
        "steel_pickaxe",
        "bronze_pickaxe",
        "steel_sword",
        "bronze_sword",
        "tonic_basic",
        "tonic_stoneskin",
    ]

    for field_name in inventory_fields:
        if hasattr(env_state.inventory, field_name):
            count = getattr(env_state.inventory, field_name)
            if hasattr(count, "sum"):
                count = int(count.sum())
            else:
                count = int(count)
            if count > 0:
                symbolic.inventory[field_name] = count

    if hasattr(env_state, "achievements"):
        for idx, achieved in enumerate(env_state.achievements):
            if achieved:
                symbolic.achievements.add(int(idx))

    return symbolic


def _parse_gain_entries(gain: Dict) -> List[Tuple[str, str, str, Optional[str]]]:
    entries: List[Tuple[str, str, str, Optional[str]]] = []
    if not isinstance(gain, dict):
        return entries

    for key, value in gain.items():
        gain_type = "inventory"
        expression = None
        description = None

        if isinstance(value, dict):
            gain_type = value.get("type", "inventory") or "inventory"
            expression = value.get("expression")
            description = value.get("description")
        else:
            expression = value

        if isinstance(gain_type, str):
            gain_type = gain_type.lower()

        if expression is None:
            expression = "lambda n: 0"

        if not isinstance(expression, str):
            expression = str(expression)

        entries.append((key, gain_type, expression, description))

    return entries


def _predicate_from_key(key: str, amount: int) -> Optional[SymbolicPredicate]:
    if key.startswith("achievement:"):
        achievement_id = _achievement_key_to_id(key)
        if achievement_id is None:
            print(f"Warning: Unknown achievement key '{key}' in requirements")
            return None
        return AchievedPredicate(achievement_id)

    if key.startswith("level:"):
        print(f"Warning: Ignoring level requirement '{key}' (levels unsupported in Fabrax)")
        return None

    if amount <= 0:
        return None
    return HasPredicate(key, amount)


def convert_skill_to_operator(
    skill_data: Dict,
    processed_skills: Dict = None,
    inline_ephemeral: bool = True,
) -> SkillOperator:
    """Convert current skill format to symbolic operator."""

    if inline_ephemeral and processed_skills and skill_data["skill_name"] in processed_skills:
        skill_spec = processed_skills[skill_data["skill_name"]]["skill_with_consumption"]
    else:
        skill_spec = skill_data["skill_with_consumption"]

    requirements = skill_spec.get("requirements", {}) or {}
    consumption = skill_spec.get("consumption", {}) or {}
    gain = skill_spec.get("gain", {}) or {}
    ephemeral = skill_spec.get("ephemeral", False)

    gain_entries = _parse_gain_entries(gain)
    if not gain_entries and isinstance(gain, dict) and gain:
        for key, value in gain.items():
            gain_entries.append((key, "inventory", str(value), None))

    if not ephemeral:
        ephemeral = any(entry_type == "ephemeral" for _, entry_type, _, _ in gain_entries)

    preconditions: Set[SymbolicPredicate] = set()
    for item, lambda_str in requirements.items():
        count = eval_lambda(lambda_str, n=1)
        predicate = _predicate_from_key(item, count)
        if predicate:
            preconditions.add(predicate)

    effects_positive: Set[SymbolicPredicate] = set()
    effects_negative: Set[SymbolicPredicate] = set()

    achievement_mapping = {name.lower(): name_enum.value for name, name_enum in Achievement.__members__.items()}
    # Manual aliases for common alternate phrasings (e.g., craft vs make)
    achievement_mapping.update(
        {
            "place_crafting_table": Achievement.PLACE_TABLE.value,
            "craft_wood_pickaxe": Achievement.MAKE_WOOD_PICKAXE.value,
            "craft_wood_sword": Achievement.MAKE_WOOD_SWORD.value,
            "craft_stone_pickaxe": Achievement.MAKE_STONE_PICKAXE.value,
            "craft_stone_sword": Achievement.MAKE_STONE_SWORD.value,
            "craft_iron_pickaxe": Achievement.MAKE_IRON_PICKAXE.value,
            "craft_iron_sword": Achievement.MAKE_IRON_SWORD.value,
            "brew_tonic": Achievement.BREW_TONIC.value,
            "brew_stone_skin": Achievement.BREW_STONE_SKIN.value,
        }
    )

    skill_name_lower = skill_data["skill_name"].lower().replace(" ", "_")
    if skill_name_lower in achievement_mapping:
        achievement_id = achievement_mapping[skill_name_lower]
        effects_positive.add(AchievedPredicate(achievement_id))

    for gain_key, gain_type, expression, _ in gain_entries:
        gain_type = gain_type or "inventory"

        if gain_type == "inventory":
            amount = eval_lambda(expression, n=1)
            if amount > 0:
                effects_positive.add(HasPredicate(gain_key, amount))
        elif gain_type == "achievement":
            achievement_id = _achievement_key_to_id(gain_key)
            if achievement_id is not None:
                effects_positive.add(AchievedPredicate(achievement_id))
            else:
                print(f"Warning: Unknown achievement gain key '{gain_key}'")
        elif gain_type == "level":
            print(f"Warning: Ignoring level gain '{gain_key}' (levels unsupported in Fabrax)")

    for item, lambda_str in consumption.items():
        count = eval_lambda(lambda_str, n=1)
        if count > 0:
            effects_negative.add(HasPredicate(item, count))

    return SkillOperator(
        name=skill_data["skill_name"],
        preconditions=preconditions,
        effects_positive=effects_positive,
        effects_negative=effects_negative,
    )


def eval_lambda(lambda_str: str, n: int) -> int:
    try:
        if isinstance(lambda_str, (int, float)):
            return int(lambda_str * n)

        if isinstance(lambda_str, str):
            if lambda_str.startswith("lambda"):
                lambda_func = eval(lambda_str)
                return int(lambda_func(n))
            if lambda_str == "n":
                return n
            return int(eval(lambda_str.replace("n", str(n))))

        return int(lambda_str)
    except Exception as exc:
        print(f"Error evaluating lambda '{lambda_str}': {exc}")
        return 0


def compute_reachable_states(
    initial_state: SymbolicState,
    skills: List[SkillOperator],
    max_iterations: int = 100,
) -> Set[SymbolicState]:
    reachable = {initial_state}

    for _ in range(max_iterations):
        new_states = set()

        for state in reachable:
            for skill in skills:
                if skill.can_apply(state):
                    next_state = skill.apply(state)
                    if next_state not in reachable:
                        new_states.add(next_state)

        if not new_states:
            break

        reachable.update(new_states)

    return reachable


def verify_skill_proposal(
    skill_operator: SkillOperator, existing_operators: List[SkillOperator]
) -> Tuple[bool, bool]:
    existing_effects = set()
    for operator in existing_operators:
        existing_effects.update(operator.effects_positive)

    feasibility = all(effect in existing_effects for effect in skill_operator.preconditions)

    novelty = any(effect not in existing_effects for effect in skill_operator.effects_positive)

    return novelty, feasibility


def get_frontier_summary(reachable_states: Set[SymbolicState]) -> str:
    if not reachable_states:
        return "## Current Exploration Frontier\n\nNo reachable states found."

    max_inventory: Dict[str, int] = {}
    all_achievements: Set[int] = set()

    for state in reachable_states:
        for item, count in state.inventory.items():
            max_inventory[item] = max(max_inventory.get(item, 0), count)

        all_achievements.update(state.achievements)

    achievement_names = {ach.value: ach.name for ach in Achievement}

    summary = "## Current Exploration Frontier\n\n"

    summary += "**Maximum Reachable Inventory Levels:**\n"
    if max_inventory:
        for item, count in sorted(max_inventory.items()):
            summary += f"- {item}: {count}\n"
    else:
        summary += "- None yet\n"

    summary += "\n**Achievements Unlocked:**\n"
    if all_achievements:
        for achievement_id in sorted(all_achievements):
            name = achievement_names.get(achievement_id, f"ACH_{achievement_id}")
            summary += f"- {achievement_id}: {name}\n"
    else:
        summary += "- None yet\n"

    return summary


__all__ = [
    "SymbolicState",
    "SymbolicPredicate",
    "HasPredicate",
    "AchievedPredicate",
    "SkillOperator",
    "abstract_state",
    "convert_skill_to_operator",
    "compute_reachable_states",
    "verify_skill_proposal",
    "get_frontier_summary",
    "eval_lambda",
    "TIERED_INVENTORY_ITEMS",
]
