"""
Symbolic State Representation for SCALAR
========================================

This module provides a symbolic representation of Craftax game states focused on
verifiable state components: inventory, achievements, and levels. This enables
explicit frontier tracking and pre-policy verification as described in the SCALAR paper.
"""

from typing import Dict, Set, List, Tuple, Optional
from dataclasses import dataclass, field

from craftax.craftax.constants import Achievement, MONSTERS_KILLED_TO_CLEAR_LEVEL


# Inventory entries that represent tier indices rather than counts
TIERED_INVENTORY_ITEMS = {"pickaxe", "sword"}


ACHIEVEMENT_NAME_TO_ID = {achievement.name.lower(): achievement.value for achievement in Achievement}


def _achievement_key_to_id(key: str) -> Optional[int]:
    """Extract achievement id from keys like 'achievement:PLACE_FURNACE'."""
    if not isinstance(key, str):
        return None

    if ":" in key:
        _, suffix = key.split(":", 1)
    else:
        suffix = key

    return ACHIEVEMENT_NAME_TO_ID.get(suffix.lower())


def _split_level_key(key: str) -> Tuple[str, str]:
    """Return (prefix, stat) for keys like 'level:player_strength'."""
    if not isinstance(key, str):
        return "", ""
    if ":" in key:
        prefix, suffix = key.split(":", 1)
        return prefix, suffix
    return key, ""


@dataclass
class SymbolicState:
    """
    Symbolic representation of Craftax game state.

    Focuses on verifiable state components that can be reliably tracked:
    - Inventory levels (item counts)
    - Achievement completion status
    - Player progression levels
    """
    inventory: Dict[str, int] = field(default_factory=dict)
    achievements: Set[int] = field(default_factory=set)  # Achievement enum values
    levels: Dict[str, int] = field(default_factory=dict)  # level_type -> level_value
    # Auxiliary, read-only for planning: per-floor monster kill counts (len=num_levels, typically 9)
    monsters_killed_by_level: Tuple[int, ...] = field(default_factory=tuple)

    def __hash__(self):
        """Make SymbolicState hashable for use in sets"""
        return hash((
            tuple(sorted(self.inventory.items())),
            tuple(sorted(self.achievements)),
            tuple(sorted(self.levels.items())),
            self.monsters_killed_by_level,
        ))

    def __eq__(self, other):
        """Equality comparison for SymbolicState"""
        if not isinstance(other, SymbolicState):
            return False
        return (self.inventory == other.inventory and
                self.achievements == other.achievements and
                self.levels == other.levels and
                self.monsters_killed_by_level == other.monsters_killed_by_level)

    def copy(self):
        """Create a deep copy of this state"""
        return SymbolicState(
            inventory=self.inventory.copy(),
            achievements=self.achievements.copy(),
            levels=self.levels.copy(),
            monsters_killed_by_level=tuple(self.monsters_killed_by_level),
        )


class SymbolicPredicate:
    """Base class for symbolic predicates over states"""

    def satisfied(self, state: SymbolicState) -> bool:
        """Check if this predicate is satisfied in the given state"""
        raise NotImplementedError

    def apply_positive(self, state: SymbolicState):
        """Apply this predicate as a positive effect"""
        raise NotImplementedError

    def apply_negative(self, state: SymbolicState):
        """Apply this predicate as a negative effect"""
        raise NotImplementedError


class HasPredicate(SymbolicPredicate):
    """Predicate: has(item, count) - inventory contains at least 'count' of 'item'"""

    def __init__(self, item: str, count: int):
        self.item = item
        self.count = count

    def satisfied(self, state: SymbolicState) -> bool:
        return state.inventory.get(self.item, 0) >= self.count

    def apply_positive(self, state: SymbolicState):
        """Add items to inventory"""
        current = state.inventory.get(self.item, 0)
        if self.item in TIERED_INVENTORY_ITEMS:
            state.inventory[self.item] = max(current, self.count)
        else:
            state.inventory[self.item] = current + self.count

    def apply_negative(self, state: SymbolicState):
        """Remove items from inventory"""
        current = state.inventory.get(self.item, 0)
        state.inventory[self.item] = max(0, current - self.count)
        if state.inventory[self.item] == 0:
            del state.inventory[self.item]

    def __str__(self):
        return f"has({self.item}, {self.count})"

    def __hash__(self):
        return hash((self.item, self.count))

    def __eq__(self, other):
        return isinstance(other, HasPredicate) and self.item == other.item and self.count == other.count


class AchievedPredicate(SymbolicPredicate):
    """Predicate: achieved(achievement) - achievement is completed"""

    def __init__(self, achievement_id: int):
        self.achievement_id = achievement_id

    def satisfied(self, state: SymbolicState) -> bool:
        return self.achievement_id in state.achievements

    def apply_positive(self, state: SymbolicState):
        """Mark achievement as completed"""
        state.achievements.add(self.achievement_id)

    def apply_negative(self, state: SymbolicState):
        """Remove achievement (rarely used)"""
        state.achievements.discard(self.achievement_id)

    def __str__(self):
        return f"achieved({self.achievement_id})"

    def __hash__(self):
        return hash(self.achievement_id)

    def __eq__(self, other):
        return isinstance(other, AchievedPredicate) and self.achievement_id == other.achievement_id


class LevelPredicate(SymbolicPredicate):
    """Predicate: level(stat, value) - player stat is at least 'value'"""

    def __init__(self, stat: str, value: int):
        self.stat = stat
        self.value = value

    def satisfied(self, state: SymbolicState) -> bool:
        return state.levels.get(self.stat, 0) >= self.value

    def apply_positive(self, state: SymbolicState):
        """Increase level"""
        current = state.levels.get(self.stat, 0)
        state.levels[self.stat] = max(current, self.value)

    def apply_negative(self, state: SymbolicState):
        """Decrease level (rarely used)"""
        current = state.levels.get(self.stat, 0)
        state.levels[self.stat] = max(0, current - self.value)
        if state.levels[self.stat] == 0:
            del state.levels[self.stat]

    def __str__(self):
        return f"level({self.stat}, {self.value})"

    def __hash__(self):
        return hash((self.stat, self.value))

    def __eq__(self, other):
        return isinstance(other, LevelPredicate) and self.stat == other.stat and self.value == other.value


@dataclass
class SkillOperator:
    """
    SCALAR-style skill operator with symbolic preconditions and effects.

    Args:
        name: Human-readable skill name
        preconditions: Set of predicates that must be satisfied to apply skill
        effects_positive: Set of predicates gained when skill is applied
        effects_negative: Set of predicates lost when skill is applied
    """
    name: str
    preconditions: Set[SymbolicPredicate] = field(default_factory=set)
    effects_positive: Set[SymbolicPredicate] = field(default_factory=set)
    effects_negative: Set[SymbolicPredicate] = field(default_factory=set)

    def can_apply(self, state: SymbolicState) -> bool:
        """Check if this skill can be applied in the given state"""
        return all(pred.satisfied(state) for pred in self.preconditions)

    def apply(self, state: SymbolicState) -> SymbolicState:
        """Apply this skill to create a new state"""
        new_state = state.copy()

        # Apply negative effects (consumption) first
        for effect in self.effects_negative:
            effect.apply_negative(new_state)

        # Apply positive effects (gains)
        for effect in self.effects_positive:
            effect.apply_positive(new_state)

        return new_state


def abstract_state(env_state) -> SymbolicState:
    """
    Convert raw Craftax environment state to symbolic representation.

    Args:
        env_state: Raw Craftax environment state

    Returns:
        SymbolicState: Symbolic representation focusing on verifiable components
    """
    symbolic = SymbolicState()

    # Extract inventory - handle both Craftax Classic and full Craftax
    inventory_fields = [
        'wood', 'stone', 'coal', 'iron', 'diamond', 'sapling',
        'wood_pickaxe', 'stone_pickaxe', 'iron_pickaxe',
        'wood_sword', 'stone_sword', 'iron_sword'
    ]

    # Add full Craftax fields if they exist
    full_craftax_fields = [
        'pickaxe', 'sword', 'bow', 'arrows', 'torches', 'ruby', 'sapphire', 'books'
    ]

    all_fields = inventory_fields + full_craftax_fields

    for field_name in all_fields:
        if hasattr(env_state.inventory, field_name):
            count = getattr(env_state.inventory, field_name)
            # Handle array fields (armour, potions) by taking sum or max
            if hasattr(count, 'sum'):  # numpy array
                count = int(count.sum())
            elif isinstance(count, (int, float)):
                count = int(count)
            else:
                continue  # Skip unsupported types

            if count > 0:
                symbolic.inventory[field_name] = count

    # Extract achievements
    if hasattr(env_state, 'achievements'):
        for i, achieved in enumerate(env_state.achievements):
            if achieved:
                symbolic.achievements.add(i)

    # Extract levels (if available in environment)
    if hasattr(env_state, 'player_level'):
        symbolic.levels['player_level'] = int(env_state.player_level)

    # Extract player stats if available (for full Craftax)
    stat_fields = ['player_dexterity', 'player_strength', 'player_intelligence']
    for stat_field in stat_fields:
        if hasattr(env_state, stat_field):
            symbolic.levels[stat_field] = int(getattr(env_state, stat_field))

    # Extract per-floor monsters killed (auxiliary context only)
    try:
        if hasattr(env_state, 'monsters_killed'):
            # Convert to simple tuple of ints for hashability
            mk = tuple(int(x) for x in list(env_state.monsters_killed))
            symbolic.monsters_killed_by_level = mk
        else:
            # Default to 9 floors of zeros if not available
            symbolic.monsters_killed_by_level = tuple([0] * 9)
    except Exception:
        # Be robust to unexpected shapes
        symbolic.monsters_killed_by_level = tuple([0] * 9)

    return symbolic


def _parse_gain_entries(gain: Dict) -> List[Tuple[str, str, str, Optional[str]]]:
    """Normalize gain entries to (key, type, expression, description)."""
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


def _predicate_from_key(key: str, amount: int) -> Optional['SymbolicPredicate']:
    """Create predicate for requirements based on key prefixes."""
    if key.startswith("achievement:"):
        achievement_id = _achievement_key_to_id(key)
        if achievement_id is None:
            print(f"Warning: Unknown achievement key '{key}' in requirements")
            return None
        return AchievedPredicate(achievement_id)

    if key.startswith("level:"):
        _, stat = _split_level_key(key)
        if not stat:
            print(f"Warning: Invalid level key '{key}' in requirements")
            return None
        return LevelPredicate(stat, amount)

    if amount <= 0:
        return None
    return HasPredicate(key, amount)


def convert_skill_to_operator(skill_data: Dict, processed_skills: Dict = None,
                             inline_ephemeral: bool = True) -> SkillOperator:
    """
    Convert current skill format to symbolic operator.

    This handles the dependency resolver's ephemeral inlining approach:
    - If inline_ephemeral=True, uses processed_skills with ephemeral requirements inlined
    - If inline_ephemeral=False, uses raw skills and represents ephemeral requirements as preconditions

    Args:
        skill_data: Skill data in current format with skill_with_consumption
        processed_skills: Optional processed skills with ephemeral requirements inlined
        inline_ephemeral: Whether to inline ephemeral requirements

    Returns:
        SkillOperator: Symbolic representation of the skill
    """
    # Use processed skills if available and inlining is enabled
    if inline_ephemeral and processed_skills and skill_data['skill_name'] in processed_skills:
        skill_spec = processed_skills[skill_data['skill_name']]['skill_with_consumption']
    else:
        skill_spec = skill_data['skill_with_consumption']

    # Extract components
    requirements = skill_spec.get('requirements', {}) or {}
    consumption = skill_spec.get('consumption', {}) or {}
    gain = skill_spec.get('gain', {}) or {}
    ephemeral = skill_spec.get('ephemeral', False)

    gain_entries = _parse_gain_entries(gain)
    if not gain_entries and isinstance(gain, dict) and gain:
        # Fallback: treat any non-empty gain dict with missing structured metadata as inventory
        for key, value in gain.items():
            gain_entries.append((key, "inventory", str(value), None))

    if not ephemeral:
        ephemeral = any(entry_type == "ephemeral" for _, entry_type, _, _ in gain_entries)

    # Convert requirements to symbolic preconditions
    preconditions: Set[SymbolicPredicate] = set()
    for item, lambda_str in requirements.items():
        count = eval_lambda(lambda_str, n=1)  # Evaluate for n=1
        predicate = _predicate_from_key(item, count)
        if predicate:
            preconditions.add(predicate)

    effects_positive: Set[SymbolicPredicate] = set()
    effects_negative: Set[SymbolicPredicate] = set()

    # Comprehensive achievement mapping for both ephemeral and non-ephemeral skills
    achievement_mapping = {
        'collect_wood': 0,          # COLLECT_WOOD = 0
        'place_table': 1,           # PLACE_TABLE = 1
        'place_crafting_table': 1,  # Alternative naming
        'eat_cow': 2,               # EAT_COW = 2
        'collect_sapling': 3,       # COLLECT_SAPLING = 3
        'collect_drink': 4,         # COLLECT_DRINK = 4
        'make_wood_pickaxe': 5,     # MAKE_WOOD_PICKAXE = 5
        'craft_wood_pickaxe': 5,    # Alternative naming
        'make_wood_sword': 6,       # MAKE_WOOD_SWORD = 6
        'craft_wood_sword': 6,      # Alternative naming
        'place_plant': 7,           # PLACE_PLANT = 7
        'defeat_zombie': 8,         # DEFEAT_ZOMBIE = 8
        'collect_stone': 9,         # COLLECT_STONE = 9
        'place_stone': 10,          # PLACE_STONE = 10
        'eat_plant': 11,            # EAT_PLANT = 11
        'defeat_skeleton': 12,      # DEFEAT_SKELETON = 12
        'make_stone_pickaxe': 13,   # MAKE_STONE_PICKAXE = 13
        'craft_stone_pickaxe': 13,  # Alternative naming
        'make_stone_sword': 14,     # MAKE_STONE_SWORD = 14
        'craft_stone_sword': 14,    # Alternative naming
        'wake_up': 15,              # WAKE_UP = 15
        'place_furnace': 16,        # PLACE_FURNACE = 16
        'collect_coal': 17,         # COLLECT_COAL = 17
        'collect_iron': 18,         # COLLECT_IRON = 18
        'collect_diamond': 19,      # COLLECT_DIAMOND = 19
        'make_iron_pickaxe': 20,    # MAKE_IRON_PICKAXE = 20
        'craft_iron_pickaxe': 20,   # Alternative naming
        'make_iron_sword': 21,      # MAKE_IRON_SWORD = 21
        'craft_iron_sword': 21,     # Alternative naming
        'make_arrow': 22,           # MAKE_ARROW = 22
        'craft_arrow': 22,          # Alternative naming
        'make_torch': 23,           # MAKE_TORCH = 23
        'craft_torch': 23,          # Alternative naming
        'place_torch': 24,          # PLACE_TORCH = 24
    }

    # Try to map skill name to achievement (works for legacy skills without explicit gain typing)
    skill_name_lower = skill_data['skill_name'].lower().replace(' ', '_')
    if skill_name_lower in achievement_mapping:
        achievement_id = achievement_mapping[skill_name_lower]
        achievement_predicate = AchievedPredicate(achievement_id)
        if achievement_predicate not in effects_positive:
            effects_positive.add(achievement_predicate)

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
            _, stat = _split_level_key(gain_key)
            if stat:
                level_value = eval_lambda(expression, n=1)
                if level_value > 0:
                    effects_positive.add(LevelPredicate(stat, level_value))
            else:
                print(f"Warning: Invalid level gain key '{gain_key}'")
        elif gain_type == "ephemeral":
            # Ephemeral capabilities are tracked through dependent requirements or achievements
            continue

    for item, lambda_str in consumption.items():
        count = eval_lambda(lambda_str, n=1)
        if count > 0:
            effects_negative.add(HasPredicate(item, count))

    return SkillOperator(
        name=skill_data['skill_name'],
        preconditions=preconditions,
        effects_positive=effects_positive,
        effects_negative=effects_negative
    )


def eval_lambda(lambda_str: str, n: int) -> int:
    """
    Safely evaluate lambda string with given n value.

    Args:
        lambda_str: Lambda function as string (e.g., "lambda n: 2*n + 1")
        n: Value to substitute for n

    Returns:
        int: Evaluated result
    """
    try:
        if isinstance(lambda_str, (int, float)):
            return int(lambda_str * n)

        if isinstance(lambda_str, str):
            if lambda_str.startswith("lambda"):
                lambda_func = eval(lambda_str)
                return int(lambda_func(n))
            else:
                # Handle simple expressions like "2" or "n"
                if lambda_str == "n":
                    return n
                else:
                    return int(eval(lambda_str.replace("n", str(n))))

        return int(lambda_str)
    except Exception as e:
        print(f"Error evaluating lambda '{lambda_str}': {e}")
        return 0


def compute_reachable_states(initial_state: SymbolicState,
                           skills: List[SkillOperator],
                           max_iterations: int = 100) -> Set[SymbolicState]:
    """
    Compute all reachable states through skill composition (frontier computation).

    Args:
        initial_state: Starting state
        skills: List of available skills
        max_iterations: Maximum iterations to prevent infinite loops

    Returns:
        Set[SymbolicState]: All states reachable from initial state
    """
    reachable = {initial_state}

    for _ in range(max_iterations):
        new_states = set()

        for state in reachable:
            for skill in skills:
                if skill.can_apply(state):
                    new_state = skill.apply(state)
                    if new_state not in reachable:
                        new_states.add(new_state)

        if not new_states:
            break  # Fixed point reached

        reachable.update(new_states)

    return reachable


def verify_skill_proposal(skill_operator: SkillOperator,
                         existing_operators: List[SkillOperator]) -> tuple[bool, bool]:
    """
    Pre-policy verification: check novelty and feasibility efficiently using skill signatures.

    Args:
        skill_operator: Proposed skill
        existing_operators: List of existing skill operators

    Returns:
        tuple[bool, bool]: (novelty, feasibility)
            - novelty: True if skill effects are not achievable by existing skills
            - feasibility: True if skill preconditions can be satisfied by existing skill effects
    """
    # Collect all positive effects (gains) from existing skills
    existing_effects = set()
    for op in existing_operators:
        existing_effects.update(op.effects_positive)

    # Feasibility check: can existing skills satisfy the proposed skill's preconditions?
    feasibility = True
    for precondition in skill_operator.preconditions:
        if precondition not in existing_effects:
            feasibility = False
            break

    # Novelty check: does this skill produce effects that existing skills cannot?
    novelty = False
    for effect in skill_operator.effects_positive:
        if effect not in existing_effects:
            novelty = True
            break

    return novelty, feasibility


def get_frontier_summary(reachable_states: Set[SymbolicState]) -> str:
    """
    Generate human-readable summary of exploration frontier for LLM prompts.

    Args:
        reachable_states: Set of all reachable states

    Returns:
        str: Formatted summary for use in prompts
    """
    if not reachable_states:
        return "## Current Exploration Frontier\n\nNo reachable states found."

    # Aggregate capabilities across all reachable states
    max_inventory = {}
    all_achievements = set()
    max_levels = {}
    monsters_killed_by_level = None

    for state in reachable_states:
        # Track maximum inventory levels
        for item, count in state.inventory.items():
            max_inventory[item] = max(max_inventory.get(item, 0), count)

        # Collect all achievements
        all_achievements.update(state.achievements)

        # Track maximum levels
        for stat, level in state.levels.items():
            max_levels[stat] = max(max_levels.get(stat, 0), level)
        # Record monsters killed vector (identical across reachable states in current abstraction)
        if state.monsters_killed_by_level:
            monsters_killed_by_level = state.monsters_killed_by_level

    # Achievement name mapping for better readability
    achievement_names = {
        0: "COLLECT_WOOD",
        1: "PLACE_TABLE",
        2: "EAT_COW",
        3: "COLLECT_SAPLING",
        4: "COLLECT_DRINK",
        5: "MAKE_WOOD_PICKAXE",
        6: "MAKE_WOOD_SWORD",
        7: "PLACE_PLANT",
        8: "DEFEAT_ZOMBIE",
        9: "COLLECT_STONE",
        10: "PLACE_STONE",
        11: "EAT_PLANT",
        12: "DEFEAT_SKELETON",
        13: "MAKE_STONE_PICKAXE",
        14: "MAKE_STONE_SWORD",
        15: "WAKE_UP",
        16: "PLACE_FURNACE",
        17: "COLLECT_COAL",
        18: "COLLECT_IRON",
        19: "COLLECT_DIAMOND",
        20: "MAKE_IRON_PICKAXE",
        21: "MAKE_IRON_SWORD",
        22: "MAKE_ARROW",
        23: "MAKE_TORCH",
        24: "PLACE_TORCH",
    }

    # Format summary
    summary = "## Current Exploration Frontier\n\n"

    summary += "**Maximum Reachable Inventory Levels:**\n"
    if max_inventory:
        for item, count in sorted(max_inventory.items()):
            summary += f"- {item}: {count}\n"
    else:
        summary += "- No items in inventory\n"

    summary += "\n**Completed Achievements (Verified Capabilities):**\n"
    if all_achievements:
        # Separate ephemeral vs inventory achievements
        ephemeral_achievements = {1, 2, 7, 8, 10, 11, 12, 15, 16, 24}  # PLACE_*, DEFEAT_*, EAT_*, WAKE_UP
        inventory_achievements = all_achievements - ephemeral_achievements

        if inventory_achievements:
            summary += "  *Inventory/Crafting Achievements:*\n"
            for achievement_id in sorted(inventory_achievements):
                name = achievement_names.get(achievement_id, f"Achievement {achievement_id}")
                summary += f"  - {name}\n"

        if ephemeral_achievements.intersection(all_achievements):
            summary += "  *Ephemeral Action Achievements (completed but effects may not persist):*\n"
            for achievement_id in sorted(ephemeral_achievements.intersection(all_achievements)):
                name = achievement_names.get(achievement_id, f"Achievement {achievement_id}")
                summary += f"  - {name}\n"
    else:
        summary += "- No achievements completed\n"

    if max_levels:
        summary += "\n**Maximum Levels:**\n"
        for stat, level in sorted(max_levels.items()):
            summary += f"- {stat}: {level}\n"

    # Dungeon progress (auxiliary context)
    if monsters_killed_by_level is not None and len(monsters_killed_by_level) > 0:
        summary += "\n**Dungeon Progress (per floor):**\n"
        summary += f"- monsters_killed_by_level: {list(monsters_killed_by_level)}\n"
        cleared = [i for i, c in enumerate(monsters_killed_by_level) if c >= MONSTERS_KILLED_TO_CLEAR_LEVEL]
        if cleared:
            summary += f"- floors_cleared (≥{MONSTERS_KILLED_TO_CLEAR_LEVEL} kills): {cleared}\n"

    summary += f"\n**Frontier Statistics:**\n"
    summary += f"- Total reachable states: {len(reachable_states)}\n"
    summary += f"- Unique items accessible: {len(max_inventory)}\n"
    summary += f"- Achievements unlocked: {len(all_achievements)}\n"

    summary += "\n**Skill Proposal Guidelines:**\n"
    summary += "Your proposed skill MUST:\n"
    summary += "1. Have preconditions achievable using current inventory/achievement levels\n"
    summary += "2. Have effects that extend beyond currently reachable capabilities\n"
    summary += "3. Focus on pushing the exploration frontier to new states\n"
    summary += "4. For ephemeral skills: Effects are verified through achievements, not persistent world changes\n"

    return summary


def convert_skills_to_operators(skills: Dict, max_inventory_capacity: int = 99) -> List[SkillOperator]:
    """
    Convert a dictionary of skills to a list of SkillOperator objects.

    Args:
        skills: Dictionary mapping skill_name -> skill_data
        max_inventory_capacity: Maximum inventory capacity for the environment

    Returns:
        List[SkillOperator]: List of skill operators
    """
    operators = []
    for skill_name, skill_data in skills.items():
        try:
            # Use the skill_data directly - it should contain skill_with_consumption
            operator = convert_skill_to_operator(skill_data)
            operators.append(operator)
        except Exception as e:
            print(f"Warning: Could not convert skill '{skill_name}' to operator: {e}")
            continue
    return operators


def verify_skill(skill_name: str, skill_data: Dict, existing_skills: Dict, max_inventory_capacity: int = 99) -> tuple[bool, bool]:
    """
    Verify that a proposed skill is both novel and feasible.

    Args:
        skill_name: Name of the proposed skill
        skill_data: Skill data dictionary
        existing_skills: Dictionary of existing skills
        max_inventory_capacity: Maximum inventory capacity

    Returns:
        tuple[bool, bool]: (is_novel, is_feasible)
    """
    try:
        # Convert existing skills to operators
        existing_operators = convert_skills_to_operators(existing_skills, max_inventory_capacity)

        # Convert proposed skill to operator
        # Create skill data in the expected format
        skill_dict = {"skill_name": skill_name, "skill_with_consumption": skill_data}
        proposed_operator = convert_skill_to_operator(skill_dict)

        # Use efficient verification function (no state enumeration)
        return verify_skill_proposal(proposed_operator, existing_operators)

    except Exception as e:
        print(f"Error in skill verification: {e}")
        return True, True  # Default to allowing the skill


def compute_frontier_summary_fast(skills: Dict, max_inventory_capacity: int = 99) -> str:
    """
    Fast frontier summary - analyzes what's reachable without enumerating all states.

    Args:
        skills: Dictionary of skill_name -> skill_data
        max_inventory_capacity: Maximum inventory capacity for the environment

    Returns:
        str: Formatted frontier summary for prompts
    """
    if not skills:
        return """## Current Exploration Frontier

**Starting State**: No skills learned yet
- Available resources: None (starting from empty inventory)
- Achievable goals: Only basic actions like movement and simple interactions
- Next frontier: Learn basic resource collection skills (wood, stone, etc.)

**Skill Proposal Guidelines:**
Your next skill should focus on basic resource collection or simple crafting tasks that don't require any preconditions."""

    # Compute topological levels of skills
    reachable_items = set()  # Items that can be obtained
    reachable_achievements = set()  # Achievements that can be completed
    skill_levels = {}  # skill_name -> level (0 = no reqs, 1+ = depends on level N-1)
    skills_by_level = {}  # level -> list of skills

    # Tiered items and their tier names
    TIERED_ITEMS = {
        "pickaxe": {1: "wood", 2: "stone", 3: "iron", 4: "diamond"},
        "sword": {1: "wood", 2: "stone", 3: "iron", 4: "diamond"},
        "armour": {1: "leather", 2: "iron", 3: "diamond"},
        "bow": {1: "basic"},
    }

    def format_requirement(req_item, amount):
        """Format a requirement with tier information if applicable."""
        if req_item in TIERED_ITEMS:
            tier_map = TIERED_ITEMS[req_item]
            if amount in tier_map:
                tier_name = tier_map[amount]
                return f"{req_item} level {amount} ({tier_name})"
            else:
                return f"{req_item} level {amount}"
        elif amount > 0:
            return f"{req_item}={amount}"
        else:
            return req_item

    # Helper function to get what a skill produces
    def get_skill_outputs(skill_data):
        outputs = {"items": set(), "achievements": set()}
        skill_info = skill_data.get("skill_with_consumption", {})
        gain = skill_info.get("gain", {})

        for item_name, gain_expr in gain.items():
            if isinstance(gain_expr, dict):
                gain_type = gain_expr.get("type", "inventory")
                if gain_type == "inventory":
                    outputs["items"].add(item_name)
                elif gain_type == "achievement":
                    outputs["achievements"].add(item_name)
            else:
                outputs["items"].add(item_name)
        return outputs

    # Compute levels iteratively
    remaining_skills = set(skills.keys())
    current_level = 0
    max_iterations = 100  # Safety limit

    while remaining_skills and current_level < max_iterations:
        skills_at_this_level = []

        for skill_name in list(remaining_skills):
            skill_info = skills[skill_name].get("skill_with_consumption", {})
            requirements = skill_info.get("requirements", {})

            # Check if all requirements are satisfied by previously reachable resources
            if current_level == 0:
                # Level 0: no requirements
                can_execute = not requirements
            else:
                # Level N: all requirements must be in reachable set
                can_execute = all(
                    req in reachable_items or req in reachable_achievements
                    for req in requirements.keys()
                )

            if can_execute:
                skills_at_this_level.append(skill_name)
                skill_levels[skill_name] = current_level
                remaining_skills.remove(skill_name)

                # Add outputs to reachable set
                outputs = get_skill_outputs(skills[skill_name])
                reachable_items.update(outputs["items"])
                reachable_achievements.update(outputs["achievements"])

        if skills_at_this_level:
            skills_by_level[current_level] = skills_at_this_level
            current_level += 1
        else:
            # No progress made, remaining skills have unsatisfiable dependencies
            break

    # Any remaining skills have circular or unsatisfiable dependencies
    if remaining_skills:
        skills_by_level["unreachable"] = list(remaining_skills)

    # Format summary
    summary = "## Current Exploration Frontier\n\n"
    summary += f"**Skills Learned**: {len(skills)} skills across {len(skills_by_level)} levels\n\n"

    # Show skills by topological level
    summary += "**Skill Dependency Levels** (topological ordering):\n\n"

    for level in sorted([k for k in skills_by_level.keys() if isinstance(k, int)]):
        level_skills = skills_by_level[level]
        summary += f"**Level {level}** ({len(level_skills)} skills"

        if level == 0:
            summary += ", no requirements):\n"
        else:
            summary += "):\n"

        # Show all skills at this level
        for skill in level_skills:
            skill_info = skills[skill].get("skill_with_consumption", {})
            requirements = skill_info.get("requirements", {})

            if requirements:
                # Format requirements with tier information
                req_strs = []
                for req_item, req_lambda in requirements.items():
                    # Try to evaluate the requirement to get the tier/amount
                    try:
                        # Evaluate lambda at n=1 to get required amount
                        if isinstance(req_lambda, str) and req_lambda.startswith("lambda"):
                            req_func = eval(req_lambda)
                            amount = req_func(1)
                            req_strs.append(format_requirement(req_item, amount))
                        else:
                            req_strs.append(req_item)
                    except:
                        req_strs.append(req_item)

                summary += f"  - {skill} (needs: {', '.join(req_strs)})\n"
            else:
                summary += f"  - {skill}\n"

        summary += "\n"

    # Show unreachable skills if any
    if "unreachable" in skills_by_level:
        unreachable = skills_by_level["unreachable"]
        summary += f"**Unreachable Skills** ({len(unreachable)} - circular/unsatisfiable dependencies):\n"
        for skill in unreachable:
            skill_info = skills[skill].get("skill_with_consumption", {})
            requirements = skill_info.get("requirements", {})

            # Format requirements with tier information
            req_strs = []
            for req_item, req_lambda in requirements.items():
                try:
                    if isinstance(req_lambda, str) and req_lambda.startswith("lambda"):
                        req_func = eval(req_lambda)
                        amount = req_func(1)
                        req_strs.append(format_requirement(req_item, amount))
                    else:
                        req_strs.append(req_item)
                except:
                    req_strs.append(req_item)

            summary += f"  - {skill} (needs: {', '.join(req_strs)})\n"
        summary += "\n"

    # Summary of reachable resources
    summary += f"**Total Reachable Resources**:\n"
    summary += f"  - Items: {len(reachable_items)}"
    if reachable_items:
        items_list = sorted(list(reachable_items))
        summary += f" ({', '.join(items_list)})\n"
    else:
        summary += "\n"

    if reachable_achievements:
        summary += f"  - Achievements: {len(reachable_achievements)}"
        achievements_list = sorted(list(reachable_achievements))
        summary += f" ({', '.join(achievements_list)})\n"

    summary += "\n**Skill Proposal Guidelines:**\n"
    summary += "Your proposed skill should either:\n"
    if reachable_items:
        summary += f"1. Utilize reachable resources: {', '.join(sorted(list(reachable_items)))}\n"
    summary += "2. Open up new resource types or achievements not yet accessible\n"
    summary += "3. Extend the highest level skills to achieve more complex goals\n"

    return summary


def compute_frontier_summary(skills: Dict, max_inventory_capacity: int = 99) -> str:
    """
    Compute and format frontier summary from skills dictionary.

    Uses fast approximation by default. Set use_exact=True for full state enumeration.

    Args:
        skills: Dictionary of skill_name -> skill_data
        max_inventory_capacity: Maximum inventory capacity for the environment

    Returns:
        str: Formatted frontier summary for prompts
    """
    # Use fast version by default
    return compute_frontier_summary_fast(skills, max_inventory_capacity)


def compute_frontier_summary_exact(skills: Dict, max_inventory_capacity: int = 99) -> str:
    """
    SLOW: Exact frontier summary via full state enumeration.
    Only use for debugging or when you need exact state information.

    Args:
        skills: Dictionary of skill_name -> skill_data
        max_inventory_capacity: Maximum inventory capacity for the environment

    Returns:
        str: Formatted frontier summary for prompts
    """
    if not skills:
        return """## Current Exploration Frontier

**Starting State**: No skills learned yet
- Available resources: None (starting from empty inventory)
- Achievable goals: Only basic actions like movement and simple interactions
- Next frontier: Learn basic resource collection skills (wood, stone, etc.)

**Skill Proposal Guidelines:**
Your next skill should focus on basic resource collection or simple crafting tasks that don't require any preconditions."""

    try:
        # Convert skills to operators (using functions already defined in this file)
        skill_operators = convert_skills_to_operators(skills, max_inventory_capacity)

        # Compute reachable states from empty starting state
        initial_state = SymbolicState()
        reachable_states = compute_reachable_states(initial_state, skill_operators)

        # Use existing function to format the summary
        return get_frontier_summary(reachable_states)

    except Exception as e:
        return f"""## Current Exploration Frontier

**Error computing frontier**: {str(e)}

**Fallback Summary**:
- Skills learned: {list(skills.keys())}
- Unable to compute detailed frontier analysis
- Propose skills that extend current capabilities

**Skill Proposal Guidelines:**
Your proposed skill should achieve something not yet accomplished by existing skills."""


# Example usage and testing
if __name__ == "__main__":
    # Example: Create initial state
    initial = SymbolicState()

    # Example: Create skill chain with ephemeral dependency
    collect_wood = SkillOperator(
        name="Collect Wood",
        preconditions=set(),  # No preconditions
        effects_positive={HasPredicate('wood', 1)},
        effects_negative=set()
    )

    # Ephemeral skill: Place Crafting Table
    place_crafting_table = SkillOperator(
        name="Place Crafting Table",
        preconditions={HasPredicate('wood', 4)},
        effects_positive={AchievedPredicate(1)},  # PLACE_TABLE achievement
        effects_negative={HasPredicate('wood', 4)}
        # Note: This produces "crafting_table" capability but we track via achievement
        # The dependency resolver will inline this requirement into dependent skills
    )

    # Example skill data for Craft Wood Pickaxe (raw form)
    craft_pickaxe_skill_data = {
        'skill_name': 'Craft Wood Pickaxe',
        'skill_with_consumption': {
            'requirements': {
                'wood': 'lambda n: 3*n + 0',        # Direct wood need
                'crafting_table': 'lambda n: 0*n + 1'  # Ephemeral precondition
            },
            'consumption': {
                'wood': 'lambda n: 3*n + 0'
            },
            'gain': {
                'wood_pickaxe': 'lambda n: n'
            },
            'ephemeral': False
        }
    }

    # Example processed skills (after dependency resolver inlining)
    craft_pickaxe_processed_data = {
        'skill_name': 'Craft Wood Pickaxe',
        'skill_with_consumption': {
            'requirements': {
                'wood': 'lambda n: 7*n + 0'  # 3 (pickaxe) + 4 (table) = 7 total
            },
            'consumption': {
                'wood': 'lambda n: 7*n + 0'  # All wood consumed
            },
            'gain': {
                'wood_pickaxe': 'lambda n: n'
            },
            'ephemeral': False
        }
    }

    # Convert to operators using the convert function
    craft_wood_pickaxe_raw = convert_skill_to_operator(craft_pickaxe_skill_data, inline_ephemeral=False)
    craft_wood_pickaxe_processed = convert_skill_to_operator(craft_pickaxe_processed_data, inline_ephemeral=True)

    # Test frontier computation with raw skills
    print("=== Testing with Raw Skills (before dependency resolution) ===")
    raw_skills = [collect_wood, place_crafting_table, craft_wood_pickaxe_raw]
    raw_reachable = compute_reachable_states(initial, raw_skills)

    print(f"Found {len(raw_reachable)} reachable states:")
    for i, state in enumerate(raw_reachable):
        print(f"State {i}: {state}")

    print("\n" + get_frontier_summary(raw_reachable))

    # Test frontier computation with processed skills (what the dependency resolver produces)
    print("\n=== Testing with Processed Skills (after dependency resolution) ===")
    processed_skills = [collect_wood, place_crafting_table, craft_wood_pickaxe_processed]
    processed_reachable = compute_reachable_states(initial, processed_skills)

    print(f"Found {len(processed_reachable)} reachable states:")
    for i, state in enumerate(processed_reachable):
        print(f"State {i}: {state}")

    # Test verification
    new_skill = SkillOperator(
        name="Craft Stone Pickaxe",
        preconditions={HasPredicate('wood_pickaxe', 1), HasPredicate('stone', 3)},
        effects_positive={HasPredicate('stone_pickaxe', 1)},
        effects_negative={HasPredicate('stone', 3)}
    )

    novelty, feasibility = verify_skill_proposal(new_skill, processed_reachable)
    print(f"\nNew skill verification - Novelty: {novelty}, Feasibility: {feasibility}")

    print("\n=== Demonstrating convert_skill_to_operator with different processing ===")
    print("Same skill data, different processing:")
    print("\nRaw form (ephemeral preconditions preserved):")
    for pred in craft_wood_pickaxe_raw.preconditions:
        print(f"  - {pred}")

    print("\nProcessed form (ephemeral preconditions inlined by dependency resolver):")
    for pred in craft_wood_pickaxe_processed.preconditions:
        print(f"  - {pred}")

    print("\nThis demonstrates the convert_skill_to_operator function working with:")
    print("- Raw skill data: includes ephemeral 'crafting_table' precondition")
    print("- Processed skill data: ephemeral requirements replaced with underlying resources")
    print("- Same skill, different representations depending on dependency resolution stage")
