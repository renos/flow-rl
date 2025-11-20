"""
Skill Dependency Resolver - Node-based Graph Implementation

This module handles skill dependencies by building a graph structure where each Node
represents a skill and its children represent the skills needed to fulfill its requirements.
The graph is then pruned to remove unnecessary executions of skills that produce
non-consumable tools/infrastructure.

This version supports optional symbolic state tracking for richer preconditions
(achievements, levels) beyond just inventory.
"""

import importlib
from math import ceil
from typing import Optional, Dict, Union

# Symbolic state bindings are configured per-environment.
SymbolicState = None
SkillOperator = None
convert_skill_to_operator = None
SYMBOLIC_STATE_AVAILABLE = False
TIERED_INVENTORY_ITEMS = set()  # Environment-specific tiered items


def _load_default_symbolic_module() -> None:
    """Attempt to load the default Craftax symbolic module."""

    candidates = [
        "flowrl.llm.craftax.symbolic_state",
    ]

    for module_path in candidates:
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            continue
        else:
            configure_symbolic_state_module(module)
            break


def configure_symbolic_state_module(module: Union[str, object]) -> None:
    """Configure symbolic state helpers from the provided module or module path."""

    global SymbolicState, SkillOperator, convert_skill_to_operator, SYMBOLIC_STATE_AVAILABLE, TIERED_INVENTORY_ITEMS

    if isinstance(module, str):
        module = importlib.import_module(module)

    SymbolicState = getattr(module, "SymbolicState", None)
    SkillOperator = getattr(module, "SkillOperator", None)
    convert_skill_to_operator = getattr(module, "convert_skill_to_operator", None)

    # Load environment-specific tiered inventory items (Craftax has pickaxe/sword, Fabrax doesn't)
    TIERED_INVENTORY_ITEMS = getattr(module, "TIERED_INVENTORY_ITEMS", set())
    if not isinstance(TIERED_INVENTORY_ITEMS, set):
        TIERED_INVENTORY_ITEMS = set(TIERED_INVENTORY_ITEMS) if TIERED_INVENTORY_ITEMS else set()

    SYMBOLIC_STATE_AVAILABLE = all(
        attr is not None for attr in (SymbolicState, SkillOperator, convert_skill_to_operator)
    )

    if not SYMBOLIC_STATE_AVAILABLE:
        SymbolicState = None
        SkillOperator = None
        convert_skill_to_operator = None


# Load default bindings at import time for backward compatibility
_load_default_symbolic_module()


class Node:
    def __init__(self, skill_name, amount_needed=1, skill_data=None):
        """
        A node in the skill dependency graph.

        Args:
            skill_name: Name of the skill this node represents
            amount_needed: How many times this skill needs to be executed
            skill_data: The skill data dictionary for this skill
        """
        self.skill_name = skill_name
        self.amount_needed = amount_needed
        self.skill_data = skill_data
        self.children = []  # List of child nodes (dependencies)
        self.level = 0  # Depth level in the graph (0 = target skill)
        self.is_pruned = False  # Whether this node has been pruned
        self.required_key: Optional[str] = None  # Requirement key this node was added to satisfy
        self.goal_key: Optional[str] = None  # Concrete gain key expected from this node
        self.goal_kind: Optional[str] = None  # inventory, achievement, level, etc.
        self.goal_amount: Optional[int] = None  # Quantity of the requirement that motivated this node
        self.goal_entries: Optional[list] = None  # Detailed requirement entries satisfied by this node

    def add_child(self, child_node):
        """Add a child dependency node."""
        self.children.append(child_node)
        child_node.level = self.level + 1

    def __repr__(self):
        return f"Node({self.skill_name}, amount={self.amount_needed}, level={self.level}, children={len(self.children)})"


class SkillDependencyResolver:
    def __init__(self, skills, max_inventory_capacity=99, initial_state=None):
        """
        Initialize the resolver with a dictionary of skills.

        Args:
            skills: Dict mapping skill_name -> skill_data
            max_inventory_capacity: Maximum number of items that can be held of each type
            initial_state: Optional SymbolicState for tracking achievements/levels
        """
        self.skills = skills
        self.max_inventory_capacity = max_inventory_capacity
        self.initial_state = initial_state.copy() if initial_state else None
        self.symbolic_operators = {}

    def resolve_dependencies(self, target_skill_name, n=1, initial_state=None, inline_ephemeral=False):
        """
        Build a dependency graph and prune it to remove unnecessary tool productions.

        Args:
            target_skill_name: Name of the skill to build dependencies for
            n: Number of times to apply this skill
            initial_state: Optional SymbolicState to override self.initial_state
            inline_ephemeral: Whether to inline ephemeral skills (for code gen) or keep them (for training order)

        Returns:
            List of skills in execution order
        """
        if target_skill_name not in self.skills:
            raise ValueError(f"Skill '{target_skill_name}' not found in skills")

        print(f"Building dependency graph for skill '{target_skill_name}' (n={n})")

        # Use provided initial_state or fall back to instance initial_state
        if initial_state is not None:
            self.initial_state = initial_state.copy()

        # Preprocess skills to inline ephemeral requirements (only if requested)
        if inline_ephemeral:
            processed_skills = self._inline_ephemeral_requirements()
        else:
            processed_skills = self.skills

        # Build symbolic operators if symbolic state is available
        if SYMBOLIC_STATE_AVAILABLE:
            self.symbolic_operators = self._build_symbolic_operators(processed_skills, inline_ephemeral=inline_ephemeral)

        # Build the complete dependency graph using processed skills
        root_node = self._build_graph(target_skill_name, n, processed_skills=processed_skills)

        # Prune unnecessary tool/infrastructure productions from the graph
        self._prune_graph_nodes(root_node, processed_skills)

        # Convert pruned graph to execution order (level-order traversal for just-in-time resource collection)
        execution_order = self._graph_to_execution_order_levelwise(root_node)

        # Optimize execution order by splitting producers to enable consumers earlier (Just-In-Time)
        # Run multiple passes to ensure convergence
        for _ in range(3):
            execution_order = self._optimize_execution_order_simulation(execution_order, processed_skills)

        # Old approach (commented out for safety):
        # execution_order = self._graph_to_execution_order(root_node)

        # Apply inventory capacity constraints
        execution_order = self._apply_inventory_constraints(execution_order, processed_skills)

        # Combine consecutive identical skills
        execution_order = self._combine_consecutive_skills(execution_order)

        # Convert execution counts to target inventory amounts
        execution_order = self._convert_to_target_amounts(execution_order, processed_skills)

        return execution_order

    def _build_graph(self, skill_name, amount_needed, visited=None, level=0, processed_skills=None, allowed_levels=None):
        """
        Recursively build the dependency graph.

        Args:
            skill_name: Name of the skill to build graph for
            amount_needed: How many times this skill is needed
            visited: Set to track cycles (for this branch only)
            level: Current depth level
            processed_skills: Skills with ephemeral requirements inlined
            allowed_levels: Set of levels where this skill can execute (constraint from parent)

        Returns:
            Node representing this skill and its dependencies
        """
        if visited is None:
            visited = set()

        if processed_skills is None:
            processed_skills = self.skills

        if level > 20:
            print(f"Warning: Maximum recursion depth reached for {skill_name}")
            return Node(skill_name, amount_needed, processed_skills.get(skill_name))

        # Create cycle detection key
        cycle_key = f"{skill_name}_{amount_needed}"
        if cycle_key in visited:
            print(f"Cycle detected: {skill_name} (amount={amount_needed}), creating leaf node")
            return Node(skill_name, amount_needed, processed_skills.get(skill_name))

        visited.add(cycle_key)

        print(f"{'  ' * level}Building graph for '{skill_name}' (amount={amount_needed})")

        # Create the node for this skill
        node = Node(skill_name, amount_needed, processed_skills.get(skill_name))
        node.level = level

        # If skill doesn't exist, return leaf node
        if skill_name not in processed_skills:
            print(f"{'  ' * level}Warning: Skill '{skill_name}' not found - creating leaf node")
            visited.remove(cycle_key)
            return node

        skill_data = processed_skills[skill_name]
        skill_with_consumption = skill_data["skill_with_consumption"]
        requirements = skill_with_consumption.get("requirements", {})

        # Extract level requirements for this skill
        current_skill_levels = None
        for req_item, req_formula in requirements.items():
            if req_item.startswith("level:"):
                if isinstance(req_formula, (list, tuple)):
                    current_skill_levels = set(req_formula)
                elif isinstance(req_formula, str) and req_formula.startswith("["):
                    try:
                        current_skill_levels = set(eval(req_formula))
                    except:
                        pass
                break

        # Apply parent's constraint (remove levels > parent's max allowed level)
        if allowed_levels is not None and current_skill_levels is not None:
            max_allowed = max(allowed_levels)
            original_levels = current_skill_levels.copy()
            current_skill_levels = {lv for lv in current_skill_levels if lv <= max_allowed}
            if current_skill_levels != original_levels:
                print(f"{'  ' * level}Constrained {skill_name} levels from {sorted(original_levels)} to {sorted(current_skill_levels)} (parent max level {max_allowed})")
            if not current_skill_levels:
                print(f"{'  ' * level}Warning: No valid levels for {skill_name} after constraint (parent max {max_allowed}, original {sorted(original_levels)})")
        elif allowed_levels is not None:
            # If this skill has no level requirement, it inherits parent's constraint
            current_skill_levels = allowed_levels

        # This skill's children must have levels <= the max level of this skill
        if current_skill_levels:
            max_level_here = max(current_skill_levels)
            child_allowed_levels = set(range(max_level_here + 1))  # {0, 1, ..., max_level_here}
        else:
            child_allowed_levels = allowed_levels

        provider_requirements = {}

        # For each requirement, find the providing skill and group by provider
        for req_item, req_formula in requirements.items():
            # Handle level requirements with OR conditions (e.g., [2, 5])
            if isinstance(req_formula, (list, tuple)):
                # Apply level constraint from parent (remove levels > max allowed)
                constrained_levels = req_formula
                if child_allowed_levels is not None and req_item.startswith("level:"):
                    max_allowed = max(child_allowed_levels)
                    original_levels = list(req_formula)
                    constrained_levels = [lv for lv in req_formula if lv <= max_allowed]
                    if constrained_levels != original_levels:
                        print(f"{'  ' * level}Constrained requirement {req_item} from {original_levels} to {constrained_levels} (max allowed level: {max_allowed})")
                    if not constrained_levels:
                        print(f"{'  ' * level}Warning: No valid levels for {req_item} after constraint (max allowed {max_allowed}, requirement is {original_levels})")
                        continue

                # Pick the minimum level as heuristic
                quantity_needed = min(constrained_levels)
                print(f"{'  ' * level}Need level {quantity_needed} (from options {constrained_levels}) of '{req_item}'")
            else:
                quantity_needed = self._evaluate_lambda(req_formula, amount_needed)
                if quantity_needed <= 0:
                    continue
                print(f"{'  ' * level}Need {quantity_needed} of '{req_item}'")

            providing_skill = None
            providing_gain_key = req_item
            requirement_kind = "inventory"
            required_tier = None

            if req_item.startswith("achievement:"):
                requirement_kind = "achievement"
                achievement_result = self._find_skill_providing_achievement(req_item, processed_skills)
                if achievement_result is not None:
                    providing_skill, providing_gain_key = achievement_result
            elif req_item.startswith("level:"):
                requirement_kind = "level"
                # Level 0 is the initial state - no skill needs to provide it
                if quantity_needed == 0:
                    print(f"{'  ' * level}Level 0 is initial state - requirement satisfied")
                    continue
                # Find skill that provides this exact level value
                level_result = self._find_skill_providing_level_value(req_item, quantity_needed, processed_skills)
                if level_result is not None:
                    providing_skill, providing_gain_key = level_result
            elif req_item.startswith("stat:"):
                requirement_kind = "stat"
                # Find skill that provides this stat
                stat_result = self._find_skill_providing_stat(req_item, quantity_needed, processed_skills)
                if stat_result is not None:
                    providing_skill, providing_gain_key = stat_result
            elif req_item.startswith("ephemeral:"):
                requirement_kind = "ephemeral"
                # Find skill that provides this ephemeral resource
                ephemeral_result = self._find_skill_providing_ephemeral(req_item, processed_skills)
                if ephemeral_result is not None:
                    providing_skill, providing_gain_key = ephemeral_result
            else:
                required_tier = quantity_needed if req_item in TIERED_INVENTORY_ITEMS else None
                providing_skill = self._find_skill_providing_item(
                    req_item,
                    processed_skills,
                    required_tier=required_tier,
                )

            if providing_skill is None:
                print(f"{'  ' * level}Warning: No skill provides '{req_item}' - treating as basic requirement")
                continue

            provider_entry = provider_requirements.setdefault(
                providing_skill,
                {"entries": []},
            )
            provider_entry["entries"].append(
                {
                    "req_item": req_item,
                    "quantity_needed": quantity_needed,
                    "requirement_kind": requirement_kind,
                    "gain_key": providing_gain_key,
                    "required_tier": required_tier,
                }
            )

        # Build subgraphs for each providing skill once, satisfying all grouped requirements
        for providing_skill, info in provider_requirements.items():
            entries = info.get("entries", [])
            if not entries:
                continue

            providing_skill_data = processed_skills.get(providing_skill)
            if not providing_skill_data:
                print(f"{'  ' * level}Warning: Skill '{providing_skill}' data missing during grouping")
                continue

            providing_gain = providing_skill_data["skill_with_consumption"].get("gain", {})

            times_needed = 0
            valid_requirement = False

            for entry in entries:
                requirement_kind = entry["requirement_kind"]
                gain_key = entry["gain_key"]
                quantity_needed = entry["quantity_needed"]
                required_tier = entry.get("required_tier")

                if requirement_kind == "achievement":
                    if gain_key and gain_key not in providing_gain:
                        print(
                            f"{'  ' * level}Warning: Skill '{providing_skill}' does not expose achievement gain '{gain_key}'"
                        )
                        continue

                    requirement_runs = 1
                    valid_requirement = True
                elif requirement_kind == "level":
                    requirement_runs = 1
                    valid_requirement = True
                elif requirement_kind == "ephemeral":
                    # Ephemeral resources are produced once per skill execution
                    requirement_runs = 1
                    valid_requirement = True
                else:
                    gain_entry = providing_gain.get(gain_key)
                    if gain_entry is None:
                        print(
                            f"{'  ' * level}Warning: Skill '{providing_skill}' does not expose gain '{gain_key}'"
                        )
                        continue

                    if isinstance(gain_entry, dict):
                        gain_expression = gain_entry.get("expression", "lambda n: n")
                    else:
                        gain_expression = gain_entry

                    gain_per_execution = self._evaluate_lambda_or_expr(gain_expression, 1)

                    if required_tier is not None:
                        requirement_runs = 1
                        valid_requirement = True
                    else:
                        if gain_per_execution == 0:
                            print(
                                f"{'  ' * level}Warning: '{providing_skill}' has zero gain for '{gain_key}', skipping"
                            )
                            continue

                        requirement_runs = max(
                            1, int(ceil(quantity_needed / gain_per_execution))
                        )
                        valid_requirement = True

                times_needed = max(times_needed, requirement_runs)

            if not valid_requirement:
                continue

            times_needed = max(1, times_needed)

            print(
                f"{'  ' * level}Need to run '{providing_skill}' {times_needed} times (covers {len(entries)} requirement(s))"
            )

            child_node = self._build_graph(
                providing_skill,
                times_needed,
                visited.copy(),
                level + 1,
                processed_skills,
                child_allowed_levels,
            )

            node.add_child(child_node)

            child_node.goal_entries = [entry.copy() for entry in entries]
            child_node.required_key = entries[0]["req_item"] if entries else None
            child_node.goal_key = entries[0]["gain_key"] if entries else None

            if all(e["requirement_kind"] == "achievement" for e in entries):
                child_node.goal_kind = "achievement"
                child_node.goal_amount = (
                    int(ceil(entries[0]["quantity_needed"])) if entries else None
                )
            elif all(e["requirement_kind"] == "inventory" for e in entries):
                child_node.goal_kind = "inventory"
                child_node.goal_amount = (
                    int(ceil(entries[0]["quantity_needed"])) if entries else None
                )
            elif all(e["requirement_kind"] == "level" for e in entries):
                child_node.goal_kind = "level"
                try:
                    child_node.goal_amount = int(ceil(entries[0]["quantity_needed"])) if entries else None
                except Exception:
                    child_node.goal_amount = None
            else:
                child_node.goal_kind = "mixed"
                child_node.goal_amount = None

        visited.remove(cycle_key)
        return node

    def _build_symbolic_operators(self, processed_skills: Dict, inline_ephemeral: bool = False) -> Dict:
        """Build symbolic operators from processed skills for richer state tracking."""
        if not SYMBOLIC_STATE_AVAILABLE:
            return {}

        operators = {}
        for skill_name, skill_data in self.skills.items():
            try:
                # Ensure skill_name is in skill_data for the converter
                if "skill_name" not in skill_data:
                    skill_data["skill_name"] = skill_name
                
                operators[skill_name] = convert_skill_to_operator(
                    skill_data,
                    processed_skills=processed_skills,
                    inline_ephemeral=inline_ephemeral,
                )
            except Exception as exc:
                print(f"Warning: failed to convert skill '{skill_name}' to operator: {exc}")
        return operators

    # OLD POST-ORDER APPROACH (commented out - kept for safety)
    # def _graph_to_execution_order(self, root_node):
    #     """
    #     Convert the graph to an execution order using post-order traversal.
    #     Dependencies are executed before the skills that need them.
    #     """
    #     print("\nConverting graph to execution order...")
    #
    #     execution_order = []
    #     visited_nodes = set()
    #     self._post_order_traversal(root_node, execution_order, visited_nodes)
    #
    #     print("Raw execution order from graph:")
    #     for i, (skill_name, amount) in enumerate(execution_order):
    #         print(f"  {i+1}. {skill_name} (amount={amount})")
    #
    #     return execution_order

    def _graph_to_execution_order_levelwise(self, root_node):
        """
        Convert the graph to an execution order using level-order traversal.
        This ensures resources are collected just-in-time, closer to when they're needed.
        """
        print("\nConverting graph to execution order (level-wise)...")

        # Group nodes by their level (distance from root)
        levels = {}
        self._collect_nodes_by_level(root_node, levels)

        execution_order = []
        skill_dependencies = {}  # skill_name -> list of dependency skill_names

        # Process levels from deepest to shallowest (highest level number to 0)
        max_level = max(levels.keys()) if levels else 0
        for level in range(max_level, -1, -1):
            print(f"Processing level {level}:")
            level_nodes = levels[level]

            # Add all nodes at this level to execution order and track dependencies
            for node in level_nodes:
                execution_order.append((node.skill_name, node.amount_needed))
                print(f"  Added: {node.skill_name} (amount={node.amount_needed})")

                # Track dependencies (children) for this skill
                dependencies = [child.skill_name for child in node.children]
                skill_dependencies[node.skill_name] = dependencies
                if dependencies:
                    print(f"    Dependencies: {dependencies}")

        print("Raw execution order from graph (level-wise):")
        for i, (skill_name, amount) in enumerate(execution_order):
            print(f"  {i+1}. {skill_name} (amount={amount})")

        # Optimize execution order by moving nodes as early as possible
        execution_order = self._optimize_execution_order(execution_order, skill_dependencies)

        return execution_order

    def _graph_to_execution_order_levelwise_nodes(self, root_node):
        """
        Convert the graph to an execution order using level-order traversal, returning Node objects.
        This is used for pruning simulation to ensure consistent traversal order.
        """
        print("Converting graph to execution order for pruning (level-wise nodes)...")

        # Group nodes by their level (distance from root)
        levels = {}
        self._collect_nodes_by_level(root_node, levels)

        execution_order_nodes = []

        # Process levels from deepest to shallowest (highest level number to 0)
        max_level = max(levels.keys()) if levels else 0
        for level in range(max_level, -1, -1):
            print(f"Processing level {level} for pruning:")
            level_nodes = levels[level]

            # Add all nodes at this level to execution order
            for node in level_nodes:
                execution_order_nodes.append(node)
                print(f"  Added: {node.skill_name} (amount={node.amount_needed})")

        return execution_order_nodes

    def _optimize_execution_order(self, execution_order, skill_dependencies):
        """
        Move skills with floor requirements to just before the floor transition.
        This ensures all floor 0 skills happen before descending to floor 1, etc.
        """
        print("\nReordering by floor requirements...")

        order = list(execution_order)
        current_floor = 0
        floor_transition_positions = {}  # floor -> position of transition skill

        # First pass: identify floor transitions
        for i, (skill_name, amount) in enumerate(order):
            is_transition, target_floor = self._is_floor_transition_skill(skill_name)
            if is_transition:
                floor_transition_positions[current_floor] = i
                current_floor = target_floor

        # Second pass: move skills to their required floor
        i = 0
        current_floor = 0
        max_iterations = len(order) * 10  # Safeguard against infinite loops
        iteration_count = 0

        while i < len(order):
            iteration_count += 1
            if iteration_count > max_iterations:
                print(f"  WARNING: Floor reordering exceeded max iterations ({max_iterations}), stopping")
                break

            skill_name, amount = order[i]

            # Check if this is a floor transition
            is_transition, target_floor = self._is_floor_transition_skill(skill_name)
            if is_transition:
                current_floor = target_floor
                i += 1
                continue

            # Check floor requirement
            required_floor = self._get_skill_floor_requirement(skill_name)

            if required_floor is not None:
                # Handle OR conditions - pick first matching floor
                if isinstance(required_floor, (list, tuple)):
                    if current_floor in required_floor:
                        # Current floor is valid, stay here
                        i += 1
                        continue
                    else:
                        # Need to move to one of the required floors (pick first)
                        required_floor = required_floor[0]

                # Check if we're on the wrong floor
                if required_floor != current_floor:
                    # Move this skill to just before we left the required floor
                    if required_floor in floor_transition_positions:
                        target_pos = floor_transition_positions[required_floor]

                        # Collect this skill and all its dependencies that need to move
                        skills_to_move = self._collect_skill_and_dependencies(
                            skill_name, i, order, skill_dependencies, required_floor
                        )

                        if not skills_to_move:
                            i += 1
                            continue

                        print(f"  Moving '{skill_name}' and its dependencies from position {i} to {target_pos} (requires floor {required_floor}, currently on {current_floor})")
                        print(f"    Skills to move: {[s for s, _ in skills_to_move]}")

                        # Store the actual tuples before removing them
                        tuples_to_move = []
                        for skill_to_move, old_pos in skills_to_move:
                            tuples_to_move.append(order[old_pos])

                        # Remove all skills that need to move (in reverse order to maintain indices)
                        # Also adjust target_pos for each removal before it
                        for skill_to_move, old_pos in reversed(skills_to_move):
                            order.pop(old_pos)
                            # If we removed something before target_pos, adjust it
                            if old_pos < target_pos:
                                target_pos -= 1

                        # Insert all skills at target position (in dependency order)
                        for tuple_to_insert in tuples_to_move:
                            order.insert(target_pos, tuple_to_insert)
                            target_pos += 1

                        # After moving skills, restart from beginning to recalculate floors correctly
                        # Rebuild floor transition positions
                        floor_transition_positions = {}
                        temp_floor = 0
                        for idx, (s_name, _) in enumerate(order):
                            is_trans, t_floor = self._is_floor_transition_skill(s_name)
                            if is_trans:
                                floor_transition_positions[temp_floor] = idx
                                temp_floor = t_floor

                        # Restart from beginning
                        i = 0
                        current_floor = 0
                        continue

            i += 1

        print("\nReordered execution order:")
        for i, (skill_name, amount) in enumerate(order):
            print(f"  {i+1}. {skill_name} (amount={amount})")

        return order

    def _build_dependency_map(self, root_node):
        """Build a map of skill dependencies from the graph."""
        deps = {}
        def traverse(node):
            if node.skill_name not in deps:
                deps[node.skill_name] = [child.skill_name for child in node.children]
                for child in node.children:
                    traverse(child)
        traverse(root_node)
        return deps

    def _get_skill_consumption(self, skill_name, amount=1, skills_map=None):
        """Get resource consumption for a skill."""
        skills = skills_map if skills_map is not None else self.skills
        if skill_name not in skills:
            return {}
        skill_data = skills[skill_name]["skill_with_consumption"]
        consumption_exprs = skill_data.get("consumption", {})
        consumption = {}
        for req, expr in consumption_exprs.items():
            # Only consider inventory items
            if ":" not in req:
                val = self._evaluate_lambda_or_expr(expr, amount)
                if val > 0:
                    consumption[req] = val
        return consumption
    
    def _get_skill_requirements(self, skill_name, amount=1, skills_map=None):
        """Get all requirements for a skill (inventory, levels, achievements)."""
        skills = skills_map if skills_map is not None else self.skills
        if skill_name not in skills:
            return {}
        skill_data = skills[skill_name]["skill_with_consumption"]
        requirements_exprs = skill_data.get("requirements", {})
        requirements = {}
        for req, expr in requirements_exprs.items():
            if isinstance(expr, (list, tuple)):
                # For OR conditions (like levels), we can't easily represent as a single number
                # We'll pass the list through
                requirements[req] = expr
            else:
                val = self._evaluate_lambda_or_expr(expr, amount)
                if val > 0 or req.startswith("level:"): # Keep level 0 requirements
                    requirements[req] = val
        return requirements

    def _get_skill_production(self, skill_name, amount=1, skills_map=None):
        """Get all production for a skill (inventory, levels, achievements)."""
        skills = skills_map if skills_map is not None else self.skills
        if skill_name not in skills:
            return {}
        skill_data = skills[skill_name]["skill_with_consumption"]
        gain = skill_data.get("gain", {})
        production = {}
        for item, expr in gain.items():
            if isinstance(expr, dict):
                expr = expr.get("expression", "lambda n: n")
            val = self._evaluate_lambda_or_expr(expr, amount)
            if val > 0 or item.startswith("level:"):
                production[item] = val
        return production

    def _optimize_execution_order_simulation(self, execution_order, processed_skills=None):
        """
        Optimize execution order using a "Priority Pull" simulation.
        
        Algorithm:
        1. Maintain a list of 'pending_tasks' (initially the input order).
        2. Maintain a 'current_state' (inventory, levels, achievements).
        3. Iteratively select the next task to execute:
           a. Look for a high-priority "Consumer" task in pending_tasks that can be made executable 
              by pulling resources from preceding "Producer" tasks.
           b. If found, "harvest" necessary resources from predecessors (splitting them), 
              execute the Consumer, and update state.
           c. If no Consumer can be pulled, execute the first task in pending_tasks.
        
        This ensures tools and state changes happen as early as possible (Just-In-Time),
        while respecting dependencies and floor constraints.
        """
        print("\nOptimizing execution order via Priority Pull simulation...")
        
        pending_tasks = list(execution_order)
        scheduled_ops = []
        
        # Initial state
        current_inventory = {}
        current_state = {} # levels, achievements
        
        while pending_tasks:
            # 1. Try to find a Consumer task to pull forward
            candidate_idx = -1
            harvest_plan = None # (producer_idx, amount_to_take) list
            
            # We only look ahead a certain distance to avoid complexity? 
            # For now, scan all.
            for i, (skill_name, amount) in enumerate(pending_tasks):
                # Skip the very first task (it's the default choice anyway)
                if i == 0:
                    continue
                
                # Is this a "Consumer"? 
                # Heuristic: Consumers have requirements OR change state (level/achievement) OR are tools.
                # Producers mainly just give inventory.
                # We want to pull Consumers.
                if self._is_producer_skill(skill_name, amount, processed_skills):
                    continue
                
                # Check if we can execute this task NOW, potentially by harvesting predecessors
                can_exec, plan = self._can_execute_with_harvest(
                    skill_name, amount, 
                    pending_tasks[:i], # Predecessors
                    current_inventory, 
                    current_state,
                    processed_skills
                )
                
                if can_exec:
                    candidate_idx = i
                    harvest_plan = plan
                    print(f"  Pulling Consumer '{skill_name}' from pos {i}")
                    break
            
            # 2. Execute the chosen task
            if candidate_idx != -1:
                # We found a candidate to pull!
                
                # First, execute the harvest plan (split/move producers)
                # Harvest plan is a map: index_in_pending -> amount_to_take
                # We process from last to first to avoid index shifting issues?
                # No, we are moving them to scheduled_ops.
                
                # We need to be careful. 'pending_tasks[:i]' are the predecessors.
                # We iterate through them and take what's needed.
                
                # Reconstruct predecessors with splits
                new_pending_prefix = []
                
                # Apply harvest
                for pred_idx in range(candidate_idx):
                    pred_skill, pred_amount = pending_tasks[pred_idx]
                    
                    if pred_idx in harvest_plan:
                        take_amount = harvest_plan[pred_idx]
                        if take_amount > 0:
                            # Schedule the taken amount
                            self._execute_task(pred_skill, take_amount, current_inventory, current_state, processed_skills)
                            scheduled_ops.append((pred_skill, take_amount))
                            print(f"    Harvested '{pred_skill}' ({take_amount})")
                            
                            # Keep remainder in pending
                            remainder = pred_amount - take_amount
                            if remainder > 0:
                                new_pending_prefix.append((pred_skill, remainder))
                        else:
                            # Take nothing, keep all
                            new_pending_prefix.append((pred_skill, pred_amount))
                    else:
                        # Not harvested, keep in pending
                        new_pending_prefix.append((pred_skill, pred_amount))
                
                # Execute the candidate
                cand_skill, cand_amount = pending_tasks[candidate_idx]
                self._execute_task(cand_skill, cand_amount, current_inventory, current_state, processed_skills)
                scheduled_ops.append((cand_skill, cand_amount))
                
                # Update pending tasks
                # New pending = new_prefix + pending[candidate_idx+1:]
                pending_tasks = new_pending_prefix + pending_tasks[candidate_idx+1:]
                
            else:
                # No candidate found, execute the first task
                skill_name, amount = pending_tasks.pop(0)
                self._execute_task(skill_name, amount, current_inventory, current_state, processed_skills)
                scheduled_ops.append((skill_name, amount))
        
        print("Optimized order:")
        for i, (s, a) in enumerate(scheduled_ops):
            print(f"  {i+1}. {s} ({a})")
            
        return scheduled_ops

    def _is_producer_skill(self, skill_name, amount, processed_skills):
        """Check if a skill is primarily a resource producer (low priority)."""
        prod = self._get_skill_production(skill_name, amount, processed_skills)
        cons = self._get_skill_consumption(skill_name, amount, processed_skills)
        reqs = self._get_skill_requirements(skill_name, amount, processed_skills)
        
        # If it changes level/achievement, it's a Consumer (state change)
        for k in prod:
            if k.startswith("level:") or k.startswith("achievement:"):
                return False
        
        # If it has requirements (tools/infrastructure), it's likely a Consumer
        # Exception: "Collect Wood" might require "Axe" (if we modeled it), but usually basic collection has no reqs.
        # If it consumes things, it's a Consumer (crafting).
        if cons:
            return False
            
        # If it only produces inventory and has no complex requirements, it's a Producer
        return True

    def _execute_task(self, skill_name, amount, inventory, state, processed_skills):
        """Update state by 'executing' a task."""
        prod = self._get_skill_production(skill_name, amount, processed_skills)
        cons = self._get_skill_consumption(skill_name, amount, processed_skills)
        
        # Apply consumption
        for item, qty in cons.items():
            inventory[item] = max(0, inventory.get(item, 0) - qty)
            
        # Apply production
        for item, qty in prod.items():
            if item.startswith("level:"):
                state[item] = qty
            elif item.startswith("achievement:"):
                state[item] = max(state.get(item, 0), qty)
            else:
                inventory[item] = inventory.get(item, 0) + qty

    def _can_execute_with_harvest(self, skill_name, amount, predecessors, current_inventory, current_state, processed_skills):
        """
        Check if a task can be executed by harvesting resources from predecessors.
        Returns (bool, harvest_plan) where harvest_plan is {pred_index: amount_to_take}.
        """
        reqs = self._get_skill_requirements(skill_name, amount, processed_skills)
        cons = self._get_skill_consumption(skill_name, amount, processed_skills)
        
        # 1. Check State Requirements (Level/Achievement)
        # These MUST be met by current_state (we assume predecessors don't change state, or if they do, we can't skip them easily)
        # Actually, if a predecessor changes state (e.g. Descend), and we need that state, we can't pull past it.
        # If we need a state that is NOT met, we can't execute.
        for req, val in reqs.items():
            if req.startswith("level:"):
                curr = current_state.get(req, 0)
                if isinstance(val, (list, tuple)):
                    if curr not in val: return False, None
                elif curr != val: return False, None
            elif req.startswith("achievement:"):
                if current_state.get(req, 0) < val: return False, None
        
        # 2. Check Floor Constraints for Predecessors
        # We cannot pull 'skill_name' past a predecessor 'P' if:
        # a. 'P' requires a specific floor, AND 'skill_name' changes the floor (e.g. Descend).
        #    (If we pull Descend before P, P will fail).
        # b. 'skill_name' requires a specific floor, AND 'P' changes the floor.
        #    (If we pull skill before P, skill might fail if it needed P's floor change).
        
        my_floor_req = self._get_skill_floor_requirement(skill_name)
        am_i_transition, my_target_floor = self._is_floor_transition_skill(skill_name)
        
        for i, (pred_name, _) in enumerate(predecessors):
            pred_floor_req = self._get_skill_floor_requirement(pred_name)
            is_pred_trans, pred_target_floor = self._is_floor_transition_skill(pred_name)
            
            # Case A: I change floor, Pred needs floor.
            if am_i_transition and pred_floor_req is not None:
                # If I execute before Pred, I change the floor.
                # Pred needs 'pred_floor_req'.
                # If I change to a floor != pred_floor_req, Pred is broken.
                # Usually Descend goes 0->1. Pred needs 0.
                # If I pull Descend, I'm on 1. Pred needs 0. Fail.
                # So if I am transition, and Pred has ANY floor req, I can't jump it?
                # Unless I transition TO the floor Pred needs? (Unlikely for Descend).
                # Safe heuristic: Don't jump floor-dependent tasks if I change floor.
                return False, None
                
            # Case B: I need floor, Pred changes floor.
            # If I jump Pred, I execute BEFORE Pred changes floor.
            # So I use the CURRENT floor.
            # If my req != current floor, I can't execute yet.
            # But this is covered by "Check State Requirements" above? 
            # Floor is usually tracked in 'level:player_level' or similar?
            # If 'level:player_level' is in current_state, we checked it.
            # But `_get_skill_floor_requirement` might check a different key or logic.
            # Let's explicitly check my floor req against current state.
            if my_floor_req is not None:
                # Assuming current_state tracks floor? 
                # If not, we rely on the fact that we are in a valid sequence.
                # If I need floor X, and I'm currently at floor Y (in current_state).
                # If X != Y, I can't execute.
                # We need to know current floor from current_state.
                # Let's assume 'level:dungeon_level' or similar is in current_state.
                # Or we trust `_get_skill_requirements` included the floor level req.
                pass

        # 3. Calculate Inventory Needs
        # Need = (Consumption + Requirements) - Current_Inventory
        needed = {}
        for item, qty in cons.items():
            needed[item] = needed.get(item, 0) + qty
        for item, qty in reqs.items():
            if not item.startswith("level:") and not item.startswith("achievement:"):
                needed[item] = max(needed.get(item, 0), qty)
        
        missing = {}
        for item, qty in needed.items():
            have = current_inventory.get(item, 0)
            if have < qty:
                missing[item] = qty - have
        
        if not missing:
            return True, {} # No harvest needed
            
        # 4. Try to Harvest from Predecessors
        harvest_plan = {}
        
        for item, amount_needed in missing.items():
            amount_found = 0
            
            # Scan predecessors to find this item
            for i, (pred_name, pred_amount) in enumerate(predecessors):
                prod = self._get_skill_production(pred_name, pred_amount, processed_skills)
                if item in prod:
                    # How much does it produce per unit?
                    # Assuming linear: total_prod / pred_amount
                    total_prod = prod[item]
                    if total_prod <= 0: continue
                    
                    per_unit = total_prod / pred_amount
                    
                    # How much do we need from this specific producer?
                    remaining_need = amount_needed - amount_found
                    
                    # How many units of pred do we need to take?
                    units_to_take = int(ceil(remaining_need / per_unit))
                    units_to_take = min(units_to_take, pred_amount)
                    
                    # Record harvest
                    # If we already taking from this pred (for another item), take max
                    current_take = harvest_plan.get(i, 0)
                    harvest_plan[i] = max(current_take, units_to_take)
                    
                    amount_found += units_to_take * per_unit
                    
                    if amount_found >= amount_needed:
                        break
            
            if amount_found < amount_needed:
                return False, None # Cannot satisfy requirements
                
        return True, harvest_plan

    def _collect_skill_and_dependencies(self, skill_name, skill_pos, order, skill_dependencies, required_floor):
        """
        Collect a skill and all its dependencies that need to be moved together.
        Returns list of (skill_name, position) tuples in dependency order (dependencies first).
        """
        skills_to_move = []
        visited = set()

        def collect_recursive(s_name, s_pos):
            if s_name in visited:
                return
            visited.add(s_name)

            # Get dependencies for this skill
            deps = skill_dependencies.get(s_name, [])

            # Recursively collect dependencies
            for dep in deps:
                # Find position of this dependency in order
                # Search backwards from current position to find the closest dependency instance
                dep_pos = None
                for idx in range(s_pos - 1, -1, -1):
                    name, _ = order[idx]
                    if name == dep:
                        dep_pos = idx
                        break

                if dep_pos is not None and dep_pos > s_pos:
                    # This dependency is after the current skill, which shouldn't happen
                    # but we'll skip moving it
                    continue
                elif dep_pos is not None:
                    collect_recursive(dep, dep_pos)

            # Add this skill to the list
            skills_to_move.append((s_name, s_pos))

        collect_recursive(skill_name, skill_pos)
        return skills_to_move

    def _find_earliest_position(self, skill_name, amount, current_pos, order, skill_dependencies):
        """
        Find the earliest position where a skill can be placed without violating constraints.

        Constraints checked:
        - All dependencies must be executed before this position
        - All resource requirements must be met at this position
        - Floor requirement must be satisfied at this position
        """
        if skill_name not in self.skills:
            return current_pos

        skill_data = self.skills[skill_name]["skill_with_consumption"]
        requirements = skill_data.get("requirements", {})
        consumption = skill_data.get("consumption", {})

        # Get dependencies for this skill
        dependencies = skill_dependencies.get(skill_name, [])

        # Simulate execution from start to find earliest valid position
        inventory = {}
        achievements = {}
        current_floor = 0
        executed_skills = set()

        earliest_valid = current_pos

        for pos in range(current_pos + 1):  # Check positions 0 to current_pos
            # Check if we can place our skill at this position (BEFORE simulating it)
            can_place = True

            # Check 1: All dependencies must be executed
            for dep in dependencies:
                if dep not in executed_skills:
                    can_place = False
                    break

            if not can_place:
                # Can't place here, simulate this position and try next
                if pos < len(order) and order[pos][0] != skill_name:
                    self._simulate_skill_execution(order[pos], pos, executed_skills, current_floor, inventory, achievements)
                    # Update floor if this was a transition
                    is_transition, target_floor = self._is_floor_transition_skill(order[pos][0])
                    if is_transition:
                        current_floor = target_floor
                continue

            # Check 2: All requirements must be satisfied
            for req_item, req_expr in requirements.items():
                # Handle floor requirements
                if req_item.startswith("level:"):
                    if isinstance(req_expr, (list, tuple)):
                        # OR condition - check if current floor is in the list
                        if current_floor not in req_expr:
                            can_place = False
                            break
                    else:
                        required_floor = self._evaluate_lambda_or_expr(req_expr, amount)
                        if current_floor != required_floor:
                            can_place = False
                            break
                    continue

                # Skip stat requirements - they're game state, not resources
                if req_item.startswith("stat:"):
                    continue

                # Handle achievement requirements
                if req_item.startswith("achievement:"):
                    required_amount = self._evaluate_lambda_or_expr(req_expr, amount)
                    if achievements.get(req_item, 0) < required_amount:
                        can_place = False
                        break
                    continue

                # Handle inventory requirements (tools/infrastructure)
                required_amount = self._evaluate_lambda_or_expr(req_expr, amount)
                if inventory.get(req_item, 0) < required_amount:
                    can_place = False
                    break

            if not can_place:
                # Can't place here, simulate this position and try next
                if pos < len(order) and order[pos][0] != skill_name:
                    self._simulate_skill_execution(order[pos], pos, executed_skills, current_floor, inventory, achievements)
                    # Update floor if this was a transition
                    is_transition, target_floor = self._is_floor_transition_skill(order[pos][0])
                    if is_transition:
                        current_floor = target_floor
                continue

            # Check 3: All consumption resources must be available
            for cons_item, cons_expr in consumption.items():
                cons_amount = self._evaluate_lambda_or_expr(cons_expr, amount)
                if inventory.get(cons_item, 0) < cons_amount:
                    can_place = False
                    break

            if can_place:
                # Found earliest valid position!
                earliest_valid = pos
                break
            else:
                # Can't place here, simulate this position and try next
                if pos < len(order) and order[pos][0] != skill_name:
                    self._simulate_skill_execution(order[pos], pos, executed_skills, current_floor, inventory, achievements)
                    # Update floor if this was a transition
                    is_transition, target_floor = self._is_floor_transition_skill(order[pos][0])
                    if is_transition:
                        current_floor = target_floor

        return earliest_valid

    def _simulate_skill_execution(self, skill_entry, pos, executed_skills, current_floor, inventory, achievements):
        """Helper to simulate executing a skill and update state."""
        pos_skill, pos_amount = skill_entry

        if pos_skill not in self.skills:
            return

        pos_data = self.skills[pos_skill]["skill_with_consumption"]

        # Track execution
        executed_skills.add(pos_skill)

        # Update inventory/achievements
        pos_gain = pos_data.get("gain", {})
        pos_consumption = pos_data.get("consumption", {})

        # Apply consumption
        for item, cons_expr in pos_consumption.items():
            cons_amount = self._evaluate_lambda_or_expr(cons_expr, pos_amount)
            inventory[item] = inventory.get(item, 0) - cons_amount

        # Apply gains
        for item, gain_expr in pos_gain.items():
            if isinstance(gain_expr, dict):
                gain_type = gain_expr.get("type", "inventory")
                gain_expr = gain_expr.get("expression", "lambda n: n")
            else:
                gain_type = "inventory"

            gain_amount = self._evaluate_lambda_or_expr(gain_expr, pos_amount)

            if isinstance(item, str) and item.startswith("achievement:"):
                achievements[item] = achievements.get(item, 0) + gain_amount
            elif gain_type == "achievement":
                achievements[item] = achievements.get(item, 0) + gain_amount
            else:
                # For tiered items (pickaxe, sword, etc.), track max level
                if item in TIERED_INVENTORY_ITEMS:
                    inventory[item] = max(inventory.get(item, 0), gain_amount)
                else:
                    inventory[item] = inventory.get(item, 0) + gain_amount

    def _fix_floor_violations(self, execution_order, skill_dependencies):
        """
        Fix floor requirement violations by moving skills backwards.
        If a skill requires floor X but we're on floor Y, move it (and all its dependencies)
        to just before the floor transition that took us away from floor X.
        """
        print("\nFixing floor requirement violations...")

        # Keep fixing until no violations remain
        max_iterations = len(execution_order) * 2  # Prevent infinite loops
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            current_floor = 0
            floor_transition_positions = {0: -1}  # floor -> last position before leaving that floor
            violation_found = False

            # Scan for violations
            for i, (skill_name, amount) in enumerate(execution_order):
                # Check if this is a floor transition
                is_transition, target_floor = self._is_floor_transition_skill(skill_name)

                if is_transition:
                    # Record position before we leave current floor
                    floor_transition_positions[current_floor] = i
                    current_floor = target_floor
                    print(f"  Position {i}: Transition to floor {target_floor}")
                    continue

                # Check floor requirement
                required_floor = self._get_skill_floor_requirement(skill_name)

                if required_floor is not None:
                    if isinstance(required_floor, (list, tuple)):
                        # OR condition - check if current floor is in the list
                        if current_floor not in required_floor:
                            # Pick the first required floor as target
                            required_floor = required_floor[0]
                            needs_move = True
                        else:
                            needs_move = False
                    else:
                        needs_move = (current_floor != required_floor)

                    if needs_move:
                        # Found a violation! Move this skill and its dependencies backwards
                        print(f"  Violation at position {i}: '{skill_name}' requires floor {required_floor}, but we're on floor {current_floor}")

                        # Find where to move it (just before we left the required floor)
                        if required_floor in floor_transition_positions:
                            target_pos = floor_transition_positions[required_floor]
                            # Insert just before the transition (or at start if -1)
                            insert_pos = target_pos if target_pos >= 0 else 0
                        else:
                            # Haven't been to that floor yet, move to start
                            insert_pos = 0

                        # Collect all skills that need to move (this skill + all its dependencies)
                        skills_to_move = self._collect_all_dependencies(skill_name, execution_order, skill_dependencies)
                        print(f"    Moving {len(skills_to_move)} skills: {[s for s, _ in skills_to_move]}")

                        # Remove all skills to move (in reverse order to maintain indices)
                        skills_to_move_with_indices = []
                        for move_skill, move_amount in skills_to_move:
                            for idx, (s, a) in enumerate(execution_order):
                                if s == move_skill and a == move_amount:
                                    skills_to_move_with_indices.append((idx, move_skill, move_amount))
                                    break

                        # Sort by index descending so we can remove without shifting
                        skills_to_move_with_indices.sort(reverse=True, key=lambda x: x[0])
                        removed_skills = []
                        for idx, move_skill, move_amount in skills_to_move_with_indices:
                            execution_order.pop(idx)
                            removed_skills.append((move_skill, move_amount))
                            print(f"      Removed '{move_skill}' from position {idx}")

                        # Insert all at target position (in reverse order to maintain their relative order)
                        removed_skills.reverse()
                        for move_skill, move_amount in removed_skills:
                            execution_order.insert(insert_pos, (move_skill, move_amount))
                            print(f"      Inserted '{move_skill}' at position {insert_pos}")

                        violation_found = True
                        break  # Start over after moving

            if not violation_found:
                print("  No violations found!")
                break

        if iteration >= max_iterations:
            print(f"  Warning: Reached max iterations ({max_iterations}), may still have violations")

        print("Fixed execution order:")
        for i, (skill_name, amount) in enumerate(execution_order):
            print(f"  {i+1}. {skill_name} (amount={amount})")

        return execution_order

    def _collect_all_dependencies(self, skill_name, execution_order, skill_dependencies):
        """
        Recursively collect a skill and all its dependencies that appear in execution_order.
        Returns list of (skill_name, amount) tuples in the order they appear in execution_order.
        """
        collected = []
        visited = set()

        def collect_recursive(name):
            if name in visited:
                return
            visited.add(name)

            # First collect dependencies recursively
            if name in skill_dependencies:
                for dep in skill_dependencies[name]:
                    collect_recursive(dep)

            # Then add this skill if it's in execution order
            for skill, amount in execution_order:
                if skill == name:
                    collected.append((skill, amount))
                    break

        collect_recursive(skill_name)
        return collected

    def _reorder_by_floor_requirements(self, execution_order):
        """
        Reorder execution to respect floor requirements.
        Skills with floor requirements must be executed in floor order.
        Preserves the original dependency ordering within each floor.
        """
        print("\nReordering execution by floor requirements...")

        # Group skills by their floor requirement, keeping track of original index
        skills_by_floor = {}  # floor -> [(original_index, skill_name, amount)]
        floor_transitions = []  # [(original_index, skill_name, amount, target_floor)]
        no_floor_requirement = []  # [(original_index, skill_name, amount)]

        for idx, (skill_name, amount) in enumerate(execution_order):
            floor_req = self._get_skill_floor_requirement(skill_name)

            # Check if this is a floor transition skill (changes player_level)
            is_transition, target_floor = self._is_floor_transition_skill(skill_name)

            if is_transition:
                floor_transitions.append((idx, skill_name, amount, target_floor))
            elif floor_req is not None:
                if isinstance(floor_req, (list, tuple)):
                    # OR condition - use the first floor
                    floor_req = floor_req[0] if floor_req else 0
                skills_by_floor.setdefault(floor_req, []).append((idx, skill_name, amount))
            else:
                no_floor_requirement.append((idx, skill_name, amount))

        # Build reordered execution
        reordered = []

        # Get all floors mentioned
        all_floors = sorted(set(skills_by_floor.keys()) | {t[3] for t in floor_transitions})

        for floor in all_floors:
            # Add skills for this floor in their original order
            if floor in skills_by_floor:
                # Sort by original index to preserve dependency order
                floor_skills = sorted(skills_by_floor[floor], key=lambda x: x[0])
                print(f"Adding {len(floor_skills)} skills for floor {floor} (preserving original order)")
                for _, skill_name, amount in floor_skills:
                    reordered.append((skill_name, amount))
                    print(f"  - {skill_name} (amount={amount})")

            # Add floor transition if needed to reach next floor
            if floor < max(all_floors, default=floor):
                next_floor = min(f for f in all_floors if f > floor)
                # Find transition skill to next_floor (use the first one by original index)
                transitions_to_next = [(idx, name, amt) for idx, name, amt, tgt in floor_transitions if tgt == next_floor]
                if transitions_to_next:
                    transitions_to_next.sort(key=lambda x: x[0])
                    _, transition_name, transition_amount = transitions_to_next[0]
                    reordered.append((transition_name, transition_amount))
                    print(f"Adding floor transition: {transition_name} -> floor {next_floor}")

        # Add skills without floor requirements at the end, in original order
        if no_floor_requirement:
            no_floor_requirement.sort(key=lambda x: x[0])
            print(f"Adding {len(no_floor_requirement)} skills without floor requirements")
            for _, skill_name, amount in no_floor_requirement:
                reordered.append((skill_name, amount))

        print("Reordered execution order:")
        for i, (skill_name, amount) in enumerate(reordered):
            print(f"  {i+1}. {skill_name} (amount={amount})")

        return reordered

    def _get_skill_floor_requirement(self, skill_name):
        """Get the floor requirement for a skill (level:player_level)."""
        if skill_name not in self.skills:
            return None

        requirements = self.skills[skill_name]["skill_with_consumption"].get("requirements", {})
        floor_req = requirements.get("level:player_level")

        if floor_req is None:
            return None

        # Handle list/tuple (OR conditions)
        if isinstance(floor_req, (list, tuple)):
            return floor_req

        # Handle lambda string
        if isinstance(floor_req, str):
            try:
                return self._evaluate_lambda_or_expr(floor_req, 1)
            except:
                return None

        return floor_req

    def _is_floor_transition_skill(self, skill_name):
        """Check if a skill transitions between floors. Returns (is_transition, target_floor)."""
        if skill_name not in self.skills:
            return False, None

        gain = self.skills[skill_name]["skill_with_consumption"].get("gain", {})

        if "level:player_level" not in gain:
            return False, None

        gain_entry = gain["level:player_level"]

        # Extract the target floor
        if isinstance(gain_entry, dict):
            expr = gain_entry.get("expression", "1")
        else:
            expr = gain_entry

        try:
            target_floor = self._evaluate_lambda_or_expr(expr, 1)
            return True, int(target_floor)
        except:
            return False, None

    def _collect_nodes_by_level(self, root_node, levels, visited_nodes=None):
        """
        Collect all nodes grouped by their level (distance from root).
        Uses BFS-style traversal to avoid revisiting nodes.
        """
        if visited_nodes is None:
            visited_nodes = set()

        node_id = id(root_node)
        if node_id in visited_nodes:
            return
        visited_nodes.add(node_id)

        level = root_node.level
        if level not in levels:
            levels[level] = []
        levels[level].append(root_node)

        # Recursively collect children
        for child in root_node.children:
            self._collect_nodes_by_level(child, levels, visited_nodes)

    def _post_order_traversal(self, node, execution_order, visited_nodes):
        """
        Post-order traversal: visit children first, then the node itself.
        """
        node_id = id(node)
        if node_id in visited_nodes:
            return
        visited_nodes.add(node_id)

        # First, process all children (dependencies)
        for child in node.children:
            self._post_order_traversal(child, execution_order, visited_nodes)

        # Then add this node to execution order
        execution_order.append((node.skill_name, node.amount_needed))

    def _prune_graph_nodes(self, root_node, processed_skills):
        """
        Prune unnecessary nodes from the graph along with their entire subtrees.

        Strategy:
        1. Collect all nodes that produce tools/infrastructure
        2. Simulate execution to see which tools are redundant
        3. Mark redundant nodes for pruning (including their children)
        4. Remove pruned nodes from the graph
        """
        print("\nPruning unnecessary tool/infrastructure productions from graph...")

        # First, collect all nodes in the SAME execution order we'll actually use (level-order)
        temp_execution_order_nodes = self._graph_to_execution_order_levelwise_nodes(root_node)

        # Simulate execution to identify redundant tool productions
        redundant_nodes = self._identify_redundant_nodes(temp_execution_order_nodes, processed_skills)

        # Mark redundant nodes and their subtrees for pruning
        for node in redundant_nodes:
            self._mark_subtree_for_pruning(node)
            print(f"Pruning node and its subtree: {node.skill_name} (amount={node.amount_needed})")

        # Remove pruned nodes from the graph
        self._remove_pruned_nodes(root_node)

        print("Graph pruning complete")

    def _collect_execution_order(self, node, execution_order, visited_nodes):
        """Collect nodes in post-order (execution order) for simulation."""
        node_id = id(node)
        if node_id in visited_nodes:
            return
        visited_nodes.add(node_id)

        # First process children
        for child in node.children:
            self._collect_execution_order(child, execution_order, visited_nodes)

        # Then add this node
        execution_order.append(node)

    def _identify_redundant_nodes(self, execution_order_nodes, processed_skills):
        """
        Simulate execution and identify nodes that produce redundant tools.
        Returns list of nodes that should be pruned.

        NOTE: The target skill (level 0) is never pruned, as it's the goal.
        """
        inventory = {}
        achievements_state = {}
        level_state = {}
        stat_state = {}
        redundant_nodes = []

        for node in execution_order_nodes:
            skill_name = node.skill_name
            amount = node.amount_needed

            # NEVER prune the target skill (root node at level 0)
            if node.level == 0:
                gains = self._parse_lambda_gain(skill_name, amount, processed_skills)
                achievement_gains = self._parse_achievement_gain(skill_name, amount, processed_skills)
                level_gains = self._parse_level_gain(skill_name, amount, processed_skills)
                stat_gains = self._parse_stat_gain(skill_name, amount, processed_skills)
                consumption = self._parse_lambda_consumption(skill_name, amount, processed_skills)

                for item, amount_consumed in consumption.items():
                    inventory[item] = inventory.get(item, 0) - amount_consumed
                for item, amount_gained in gains.items():
                    if item in TIERED_INVENTORY_ITEMS:
                        inventory[item] = max(inventory.get(item, 0), amount_gained)
                    else:
                        inventory[item] = inventory.get(item, 0) + amount_gained

                self._update_achievement_state(achievements_state, achievement_gains)
                for key, val in level_gains.items():
                    try:
                        lv = int(val)
                    except Exception:
                        lv = 0
                    level_state[key] = max(level_state.get(key, 0), lv)
                for key, val in stat_gains.items():
                    try:
                        add = int(val)
                    except Exception:
                        add = 0
                    stat_state[key] = stat_state.get(key, 0) + add

                print(f"Keeping target skill {skill_name} (amount={amount}), inventory: {inventory}")
                continue

            if skill_name not in processed_skills:
                continue

            gains = self._parse_lambda_gain(skill_name, amount, processed_skills)
            achievement_gains = self._parse_achievement_gain(skill_name, amount, processed_skills)
            level_gains = self._parse_level_gain(skill_name, amount, processed_skills)
            stat_gains = self._parse_stat_gain(skill_name, amount, processed_skills)
            ephemeral_gains = self._parse_ephemeral_gain(skill_name, amount, processed_skills)
            consumption = self._parse_lambda_consumption(skill_name, amount, processed_skills)

            # Never prune skills that produce ephemeral resources (they need to be trained)
            if ephemeral_gains:
                print(f"Node {skill_name} produces ephemeral resources {list(ephemeral_gains.keys())}, never pruning")
                # Still update state and continue
                for item, amount_consumed in consumption.items():
                    inventory[item] = inventory.get(item, 0) - amount_consumed
                for item, amount_gained in gains.items():
                    if item in TIERED_INVENTORY_ITEMS:
                        inventory[item] = max(inventory.get(item, 0), amount_gained)
                    else:
                        inventory[item] = inventory.get(item, 0) + amount_gained
                self._update_achievement_state(achievements_state, achievement_gains)
                for key, val in level_gains.items():
                    try:
                        lv = int(val)
                    except Exception:
                        lv = 0
                    level_state[key] = max(level_state.get(key, 0), lv)
                for key, val in stat_gains.items():
                    try:
                        add = int(val)
                    except Exception:
                        add = 0
                    stat_state[key] = stat_state.get(key, 0) + add
                continue

            should_prune = False

            # First check if this node provides stats or levels that are required later
            # This must happen BEFORE we check goal_entries to prevent early pruning
            provides_required_stat = False
            provides_required_level = False

            if stat_gains:
                for stat_key, stat_value in stat_gains.items():
                    # Only keep if we don't already have this stat value
                    current_stat = stat_state.get(stat_key, 0)
                    if current_stat < stat_value:
                        if self._is_stat_required_later_in_nodes(
                            execution_order_nodes, node, stat_key, processed_skills
                        ):
                            print(f"Node {skill_name} provides required stat {stat_key}, must keep it")
                            provides_required_stat = True
                            break
                    else:
                        print(f"Node {skill_name} provides stat {stat_key}={stat_value} but we already have {current_stat}")

            if level_gains:
                for level_key, level_value in level_gains.items():
                    # Only keep if we don't already have this level
                    current_level = level_state.get(level_key, 0)
                    try:
                        target_level = int(level_value)
                    except:
                        target_level = 0

                    if current_level < target_level:
                        if self._is_stat_required_later_in_nodes(
                            execution_order_nodes, node, level_key, processed_skills
                        ):
                            print(f"Node {skill_name} provides required level {level_key}={target_level}, must keep it")
                            provides_required_level = True
                            break
                    else:
                        print(f"Node {skill_name} provides level {level_key}={target_level} but we already have level {current_level}")

            goal_entries = getattr(node, "goal_entries", None)
            achievement_entries = []
            inventory_entries = []

            if goal_entries:
                if self._are_goal_entries_satisfied(
                    goal_entries, inventory, achievements_state, level_state, stat_state
                ):
                    # Only prune if it doesn't provide required stats or levels
                    if not provides_required_stat and not provides_required_level:
                        print(
                            f"Node {skill_name} already satisfies all grouped requirements"
                        )
                        redundant_nodes.append(node)
                        continue
                    else:
                        print(
                            f"Node {skill_name} satisfies grouped requirements but provides required stats/levels, keeping it"
                        )

                for entry in goal_entries:
                    kind = entry.get("requirement_kind")
                    if kind == "achievement":
                        achievement_entries.append(entry)
                    elif kind == "inventory":
                        inventory_entries.append(entry)

            if goal_entries and achievement_entries and not inventory_entries:
                all_satisfied = True
                for achievement_entry in achievement_entries:
                    required_key = achievement_entry.get("req_item")
                    produced_key = achievement_entry.get("gain_key")
                    goal_amount_raw = achievement_entry.get("quantity_needed") or 1
                    try:
                        goal_amount = max(1, int(ceil(goal_amount_raw)))
                    except Exception:
                        goal_amount = 1

                    candidate_keys = set()
                    for key in (required_key, produced_key):
                        if isinstance(key, str) and key:
                            candidate_keys.add(key)
                            normalized = self._normalize_achievement_key(key)
                            if normalized:
                                candidate_keys.add(normalized)

                    current_amount = 0
                    for key in candidate_keys:
                        current_amount = max(current_amount, achievements_state.get(key, 0))

                    if current_amount < goal_amount:
                        all_satisfied = False
                        break

                if all_satisfied:
                    key_names = [entry.get("req_item") for entry in achievement_entries]
                    print(
                        f"Node {skill_name} already satisfied achievement requirement(s) {key_names}"
                    )
                    should_prune = True
            elif getattr(node, "goal_kind", None) == "achievement":
                required_key = getattr(node, "required_key", None)
                produced_key = getattr(node, "goal_key", None)
                goal_amount = getattr(node, "goal_amount", None) or 1

                candidate_keys = []
                if isinstance(required_key, str):
                    candidate_keys.extend(
                        [required_key, self._normalize_achievement_key(required_key)]
                    )
                if isinstance(produced_key, str):
                    candidate_keys.extend(
                        [produced_key, self._normalize_achievement_key(produced_key)]
                    )

                current_amount = 0
                for key in candidate_keys:
                    if key:
                        current_amount = max(current_amount, achievements_state.get(key, 0))

                if current_amount >= goal_amount:
                    print(
                        f"Node {skill_name} already satisfied achievement requirement '{required_key}'"
                    )
                    should_prune = True

            if not should_prune:
                for item, amount_gained in gains.items():
                    item_ever_used = self._is_item_consumed_later_in_nodes(
                        execution_order_nodes, node, item, processed_skills
                    )

                    if item in TIERED_INVENTORY_ITEMS:
                        current_level = inventory.get(item, 0)
                        target_level = amount_gained
                        if not item_ever_used and current_level >= target_level:
                            print(
                                f"Node {skill_name} produces redundant {item}: have level {current_level}, target level {target_level}, never used"
                            )
                            should_prune = True
                            break
                    else:
                        if not item_ever_used and inventory.get(item, 0) > 0:
                            print(
                                f"Node {skill_name} produces redundant {item}: have {inventory.get(item, 0)}, never used"
                            )
                            should_prune = True
                            break

            # Override pruning if this node provides required stats or levels
            if provides_required_stat or provides_required_level:
                should_prune = False

            if should_prune:
                redundant_nodes.append(node)
                continue

            for item, amount_consumed in consumption.items():
                inventory[item] = inventory.get(item, 0) - amount_consumed
            for item, amount_gained in gains.items():
                if item in TIERED_INVENTORY_ITEMS:
                    inventory[item] = max(inventory.get(item, 0), amount_gained)
                else:
                    inventory[item] = inventory.get(item, 0) + amount_gained

            self._update_achievement_state(achievements_state, achievement_gains)
            for key, val in level_gains.items():
                try:
                    lv = int(val)
                except Exception:
                    lv = 0
                level_state[key] = max(level_state.get(key, 0), lv)
            for key, val in stat_gains.items():
                try:
                    add = int(val)
                except Exception:
                    add = 0
                stat_state[key] = stat_state.get(key, 0) + add

            print(f"Keeping node {skill_name} (amount={amount}), inventory: {inventory}")

        return redundant_nodes

    def _are_goal_entries_satisfied(self, goal_entries, inventory, achievements_state, level_state=None, stat_state=None):
        """Check whether all grouped requirements for a node are already satisfied."""
        if level_state is None:
            level_state = {}
        if stat_state is None:
            stat_state = {}

        for entry in goal_entries:
            kind = entry.get("requirement_kind")
            req_item = entry.get("req_item")
            quantity = entry.get("quantity_needed") or 0

            if kind == "achievement":
                candidate_keys = set()
                if isinstance(req_item, str) and req_item:
                    candidate_keys.add(req_item)
                    normalized = self._normalize_achievement_key(req_item)
                    if normalized:
                        candidate_keys.add(normalized)

                current_amount = 0
                for key in candidate_keys:
                    current_amount = max(current_amount, achievements_state.get(key, 0))

                try:
                    needed = max(1, int(ceil(quantity)))
                except Exception:
                    needed = 1

                if current_amount < needed:
                    return False

            elif kind == "inventory":
                if req_item is None:
                    return False

                try:
                    needed = max(0, int(ceil(quantity)))
                except Exception:
                    needed = 0

                required_tier = entry.get("required_tier")

                if required_tier is not None:
                    current_level = inventory.get(req_item, 0)
                    if current_level < required_tier:
                        return False
                else:
                    current_amount = inventory.get(req_item, 0)
                    if current_amount < needed:
                        return False
            elif kind == "level":
                if not isinstance(req_item, str):
                    return False
                try:
                    needed = max(0, int(ceil(quantity)))
                except Exception:
                    needed = 0
                current = level_state.get(req_item, 0)
                if current < needed:
                    return False
            elif kind == "stat_kills":
                if not isinstance(req_item, str):
                    return False
                try:
                    needed = max(0, int(ceil(quantity)))
                except Exception:
                    needed = 0
                current = stat_state.get(req_item, 0)
                if current < needed:
                    return False

        return True

    def _is_item_consumed_later_in_nodes(self, execution_order_nodes, current_node, item, processed_skills):
        """Check if an item is consumed OR required by any node that appears later in execution order."""
        found_current = False
        for node in execution_order_nodes:
            if node == current_node:
                found_current = True
                continue

            if found_current and node.skill_name in processed_skills:
                # Check consumption (items that are used up)
                consumption = self._parse_lambda_consumption(node.skill_name, node.amount_needed, processed_skills)
                if item in consumption and consumption[item] > 0:
                    return True

                # Check requirements (tools/infrastructure needed but not consumed)
                skill_data = processed_skills[node.skill_name]
                requirements = skill_data["skill_with_consumption"].get("requirements", {})
                if item in requirements:
                    return True

        return False

    def _is_stat_required_later_in_nodes(self, execution_order_nodes, current_node, stat_key, processed_skills):
        """Check if a stat is required by any node that appears later in execution order."""
        found_current = False
        for node in execution_order_nodes:
            if node == current_node:
                found_current = True
                continue

            if found_current:
                # Check goal_entries for this stat
                goal_entries = getattr(node, "goal_entries", None)
                if goal_entries:
                    for entry in goal_entries:
                        if entry.get("req_item") == stat_key or entry.get("gain_key") == stat_key:
                            return True

                # Also check skill requirements
                if node.skill_name in processed_skills:
                    skill_data = processed_skills[node.skill_name]
                    requirements = skill_data["skill_with_consumption"].get("requirements", {})
                    if stat_key in requirements:
                        return True

        return False

    def _mark_subtree_for_pruning(self, node):
        """Mark a node and all its children for pruning."""
        node.is_pruned = True
        for child in node.children:
            self._mark_subtree_for_pruning(child)

    def _remove_pruned_nodes(self, node):
        """Remove nodes marked for pruning from the graph."""
        # Remove pruned children
        node.children = [child for child in node.children if not getattr(child, 'is_pruned', False)]

        # Recursively clean up remaining children
        for child in node.children:
            self._remove_pruned_nodes(child)


    def _parse_skill_gains_by_type(self, skill_name, n=1, processed_skills=None):
        """Return gains grouped by type (inventory, achievement, level, etc.)."""
        skills_to_use = processed_skills if processed_skills is not None else self.skills

        results = {
            "inventory": {},
            "achievement": {},
            "level": {},
            "stat": {},
            "ephemeral": {},
        }

        if skill_name not in skills_to_use:
            return results

        skill_data = skills_to_use[skill_name]
        skill_with_consumption = skill_data["skill_with_consumption"]
        gain = skill_with_consumption.get("gain", {})

        for item_name, gain_entry in gain.items():
            try:
                gain_type = None
                gain_expr = gain_entry

                if isinstance(gain_entry, dict):
                    entry_type = gain_entry.get("type", "inventory")
                    if isinstance(entry_type, str):
                        gain_type = entry_type.lower()
                    gain_expr = gain_entry.get("expression", "lambda n: n")

                if not gain_type and isinstance(item_name, str):
                    if item_name.startswith("achievement:"):
                        gain_type = "achievement"
                    elif item_name.startswith("level:"):
                        gain_type = "level"
                    elif item_name.startswith("stat:"):
                        gain_type = "stat"
                    elif item_name.startswith("ephemeral:"):
                        gain_type = "ephemeral"
                    else:
                        gain_type = "inventory"

                if not gain_type:
                    gain_type = "inventory"

                quantity_gained = self._evaluate_lambda_or_expr(gain_expr, n)

                # Store under recognized type bucket (fallback to inventory)
                target_bucket = results.setdefault(gain_type, {})
                target_bucket[item_name] = quantity_gained
            except Exception as e:
                print(
                    f"Error parsing gain '{gain_entry}' for item '{item_name}' in skill '{skill_name}': {e}"
                )

        return results

    def _parse_lambda_gain(self, skill_name, n=1, processed_skills=None):
        """Parse lambda functions in skill gain and return actual quantities gained.

        Supports both simple format (item: lambda) and structured format (item: {type, expression, description}).
        Only returns inventory-type gains for capacity tracking.
        """
        return self._parse_skill_gains_by_type(skill_name, n, processed_skills).get("inventory", {})

    def _parse_achievement_gain(self, skill_name, n=1, processed_skills=None):
        """Parse gains and return achievement-type quantities."""
        return self._parse_skill_gains_by_type(skill_name, n, processed_skills).get(
            "achievement", {}
        )

    def _parse_level_gain(self, skill_name, n=1, processed_skills=None):
        """Parse gains and return level-type quantities."""
        return self._parse_skill_gains_by_type(skill_name, n, processed_skills).get(
            "level", {}
        )

    def _parse_stat_gain(self, skill_name, n=1, processed_skills=None):
        """Parse gains and return stat-type quantities (e.g., stat_kills).

        Stats are environment-specific (e.g., Craftax has monster kills, Fabrax doesn't).
        Returns empty dict if stat gains are not defined for this environment.
        """
        return self._parse_skill_gains_by_type(skill_name, n, processed_skills).get(
            "stat", {}
        )
    
    def _parse_ephemeral_gain(self, skill_name, n=1, processed_skills=None):
        """Parse gains and return ephemeral-type quantities."""
        return self._parse_skill_gains_by_type(skill_name, n, processed_skills).get(
            "ephemeral", {}
        )

    def _update_achievement_state(self, achievement_state: Dict[str, float], gains: Dict[str, float]):
        """Merge newly acquired achievements into the running state."""
        for key, amount in gains.items():
            if amount is None:
                continue

            normalized = self._normalize_achievement_key(key)

            if key:
                achievement_state[key] = achievement_state.get(key, 0) + amount
            if normalized:
                achievement_state[normalized] = achievement_state.get(normalized, 0) + amount

    def _parse_lambda_consumption(self, skill_name, n=1, processed_skills=None):
        """Parse lambda functions in skill consumption and return actual quantities consumed."""
        skills_to_use = processed_skills if processed_skills is not None else self.skills

        if skill_name not in skills_to_use:
            return {}

        skill_data = skills_to_use[skill_name]
        skill_with_consumption = skill_data["skill_with_consumption"]
        consumption = skill_with_consumption.get("consumption", {})

        parsed_consumption = {}
        for item_name, lambda_str in consumption.items():
            try:
                quantity_consumed = self._evaluate_lambda(lambda_str, n)
                parsed_consumption[item_name] = quantity_consumed
            except Exception as e:
                print(f"Error parsing consumption lambda '{lambda_str}' for item '{item_name}' in skill '{skill_name}': {e}")
                parsed_consumption[item_name] = 0

        return parsed_consumption

    def _evaluate_lambda(self, lambda_str, n):
        """Evaluate a lambda function string with the given n value."""
        try:
            # Handle list/tuple inputs (e.g. for OR conditions in levels)
            if isinstance(lambda_str, (list, tuple)):
                if not lambda_str:
                    return 0
                try:
                    return int(min(lambda_str))
                except:
                    return 0

            if isinstance(lambda_str, (int, float)):
                return int(lambda_str * n)

            lambda_func = eval(lambda_str)
            return lambda_func(n)
        except Exception as e:
            print(f"Error evaluating lambda '{lambda_str}': {e}")
            return 0

    def _evaluate_lambda_or_expr(self, expr, n):
        """Evaluate a lambda function string or expression with the given n value."""
        try:
            # Handle list/tuple inputs (e.g. for OR conditions in levels)
            if isinstance(expr, (list, tuple)):
                if not expr:
                    return 0
                try:
                    return int(min(expr))
                except:
                    return 0

            if isinstance(expr, str):
                if expr == "n":
                    return n
                elif expr.startswith("lambda"):
                    lambda_func = eval(expr)
                    return lambda_func(n)
                else:
                    return eval(expr.replace("n", str(n)))
            else:
                return expr * n
        except Exception as e:
            print(f"Error evaluating expression '{expr}': {e}")
            return 0

    def _find_skill_providing_item(self, item_name, skills_dict=None, required_tier=None):
        """Find a skill that provides (gains) the specified item.

        For tiered items (pickaxe, sword), finds the skill that provides at least the
        required tier, preferring the skill with the closest tier to avoid waste.

        Args:
            item_name: Name of the item to find
            skills_dict: Optional dict of skills to search (defaults to self.skills)
            required_tier: For tiered items, the minimum tier needed (e.g., 1 for wood pickaxe)

        Returns:
            Skill name that provides the item, or None if not found
        """
        skills_to_search = skills_dict if skills_dict is not None else self.skills
        is_tiered = item_name in TIERED_INVENTORY_ITEMS

        # For tiered items, collect all candidates and choose the best one
        if is_tiered and required_tier is not None:
            candidates = []  # List of (skill_name, tier)

            for skill_name, skill_data in skills_to_search.items():
                skill_with_consumption = skill_data["skill_with_consumption"]
                gain = skill_with_consumption.get("gain", {})

                if item_name in gain:
                    gain_entry = gain[item_name]

                    # Check if it's an inventory gain (for structured format)
                    if isinstance(gain_entry, dict):
                        gain_type = gain_entry.get("type", "inventory")
                        if isinstance(gain_type, str):
                            gain_type = gain_type.lower()

                        if gain_type != "inventory":
                            continue

                        gain_expr = gain_entry.get("expression", "lambda n: n")
                    else:
                        gain_expr = gain_entry

                    # Evaluate the tier this skill provides
                    tier = self._evaluate_lambda_or_expr(gain_expr, 1)

                    # Only consider skills that meet the required tier
                    if tier >= required_tier:
                        candidates.append((skill_name, tier))

            if not candidates:
                return None

            # Sort by tier and return the skill with the lowest sufficient tier
            candidates.sort(key=lambda x: x[1])
            return candidates[0][0]

        # For non-tiered items, return the first skill that provides it
        for skill_name, skill_data in skills_to_search.items():
            skill_with_consumption = skill_data["skill_with_consumption"]
            gain = skill_with_consumption.get("gain", {})

            if item_name in gain:
                gain_entry = gain[item_name]

                # If structured format, check if it's an inventory gain
                if isinstance(gain_entry, dict):
                    gain_type = gain_entry.get("type", "inventory")
                    if isinstance(gain_type, str):
                        gain_type = gain_type.lower()

                    # Only return if it's an inventory gain
                    if gain_type == "inventory":
                        return skill_name
                else:
                    # Simple format - always return
                    return skill_name

        return None

    def _find_skill_providing_achievement(self, achievement_name, skills_dict=None):
        """Find a skill that produces the specified achievement gain.

        Returns:
            Tuple of (skill_name, gain_key) if found, otherwise None.
        """

        skills_to_search = skills_dict if skills_dict is not None else self.skills

        normalized_key = achievement_name
        if not normalized_key.startswith("achievement:"):
            normalized_key = f"achievement:{normalized_key}"

        for skill_name, skill_data in skills_to_search.items():
            skill_with_consumption = skill_data.get("skill_with_consumption", {})
            gain = skill_with_consumption.get("gain", {})

            if not isinstance(gain, dict):
                continue

            # Direct match on normalized key.
            if normalized_key in gain:
                gain_entry = gain[normalized_key]

                if not isinstance(gain_entry, dict):
                    return skill_name, normalized_key

                entry_type = gain_entry.get("type", "")
                if isinstance(entry_type, str) and entry_type.lower() == "achievement":
                    return skill_name, normalized_key

            # Inspect all gain entries for explicit achievement types even if the key omits the prefix.
            for gain_key, gain_value in gain.items():
                if not isinstance(gain_value, dict):
                    continue

                entry_type = gain_value.get("type", "")
                if not isinstance(entry_type, str) or entry_type.lower() != "achievement":
                    continue

                candidate_keys = {gain_key}
                if not gain_key.startswith("achievement:"):
                    candidate_keys.add(f"achievement:{gain_key}")

                if normalized_key in candidate_keys:
                    return skill_name, gain_key

        return None

    def _find_skill_providing_ephemeral(self, ephemeral_name, skills_dict=None):
        """Find a skill that produces the specified ephemeral resource gain.

        Returns:
            Tuple of (skill_name, gain_key) if found, otherwise None.
        """
        skills_to_search = skills_dict if skills_dict is not None else self.skills

        normalized_key = ephemeral_name
        if not normalized_key.startswith("ephemeral:"):
            normalized_key = f"ephemeral:{normalized_key}"

        for skill_name, skill_data in skills_to_search.items():
            skill_with_consumption = skill_data.get("skill_with_consumption", {})
            gain = skill_with_consumption.get("gain", {})

            if not isinstance(gain, dict):
                continue

            # Direct match on normalized key
            if normalized_key in gain:
                gain_entry = gain[normalized_key]

                if not isinstance(gain_entry, dict):
                    return skill_name, normalized_key

                entry_type = gain_entry.get("type", "")
                if isinstance(entry_type, str) and entry_type.lower() == "ephemeral":
                    return skill_name, normalized_key

            # Inspect all gain entries for explicit ephemeral types even if the key omits the prefix
            for gain_key, gain_value in gain.items():
                if not isinstance(gain_value, dict):
                    continue

                entry_type = gain_value.get("type", "")
                if not isinstance(entry_type, str) or entry_type.lower() != "ephemeral":
                    continue

                candidate_keys = {gain_key}
                if not gain_key.startswith("ephemeral:"):
                    candidate_keys.add(f"ephemeral:{gain_key}")

                if normalized_key in candidate_keys:
                    return skill_name, gain_key

        return None

    def _find_skill_providing_stat(self, stat_key, required_value, skills_dict=None):
        """Find a skill that provides the specified stat with sufficient value.

        Args:
            stat_key: The stat key (e.g., 'stat:monsters_killed:1')
            required_value: The stat value needed (e.g., 8)
            skills_dict: Optional dict of skills to search

        Returns:
            Tuple of (skill_name, gain_key) if found, otherwise None
        """
        skills_to_search = skills_dict if skills_dict is not None else self.skills

        for skill_name, skill_data in skills_to_search.items():
            skill_with_consumption = skill_data.get("skill_with_consumption", {})
            gain = skill_with_consumption.get("gain", {})
            if not isinstance(gain, dict):
                continue

            if stat_key in gain:
                gain_entry = gain[stat_key]

                # Extract the expression
                if isinstance(gain_entry, dict):
                    gtype = gain_entry.get("type", "").lower() if isinstance(gain_entry.get("type", ""), str) else ""
                    if gtype != "stat":
                        continue
                    gain_expr = gain_entry.get("expression", "lambda n: n")
                else:
                    gain_expr = gain_entry

                # Evaluate the value this skill provides
                try:
                    provided_value = self._evaluate_lambda_or_expr(gain_expr, required_value)
                    if provided_value >= required_value:
                        return skill_name, stat_key
                except:
                    continue

        return None

    def _find_skill_providing_level(self, level_key, skills_dict=None):
        """Find a skill that provides the specified level key (e.g., 'level:player_level')."""
        skills_to_search = skills_dict if skills_dict is not None else self.skills

        for skill_name, skill_data in skills_to_search.items():
            skill_with_consumption = skill_data.get("skill_with_consumption", {})
            gain = skill_with_consumption.get("gain", {})
            if not isinstance(gain, dict):
                continue

            if level_key in gain:
                gain_entry = gain[level_key]
                if isinstance(gain_entry, dict):
                    gtype = gain_entry.get("type", "").lower() if isinstance(gain_entry.get("type", ""), str) else ""
                    if gtype == "level":
                        return skill_name, level_key
                else:
                    # Unstructured but matching key; assume level
                    return skill_name, level_key

            # Also scan typed entries
            for gain_key, gain_entry in gain.items():
                if isinstance(gain_entry, dict):
                    gtype = gain_entry.get("type", "").lower() if isinstance(gain_entry.get("type", ""), str) else ""
                    if gtype != "level":
                        continue
                    candidates = {gain_key}
                    if not gain_key.startswith("level:"):
                        candidates.add(f"level:{gain_key}")
                    if level_key in candidates:
                        return skill_name, gain_key

        return None

    def _find_skill_providing_level_value(self, level_key, required_value, skills_dict=None):
        """Find a skill that provides the specified level key with the exact value needed.

        Args:
            level_key: The level key (e.g., 'level:player_level')
            required_value: The exact level value needed (e.g., 2)
            skills_dict: Optional dict of skills to search

        Returns:
            Tuple of (skill_name, gain_key) if found, otherwise None
        """
        skills_to_search = skills_dict if skills_dict is not None else self.skills

        for skill_name, skill_data in skills_to_search.items():
            skill_with_consumption = skill_data.get("skill_with_consumption", {})
            gain = skill_with_consumption.get("gain", {})
            if not isinstance(gain, dict):
                continue

            if level_key in gain:
                gain_entry = gain[level_key]

                # Extract the expression
                if isinstance(gain_entry, dict):
                    gtype = gain_entry.get("type", "").lower() if isinstance(gain_entry.get("type", ""), str) else ""
                    if gtype != "level":
                        continue
                    gain_expr = gain_entry.get("expression", "1")
                else:
                    gain_expr = gain_entry

                # Evaluate the value this skill provides
                try:
                    provided_value = self._evaluate_lambda_or_expr(gain_expr, 1)
                    if provided_value == required_value:
                        return skill_name, level_key
                except:
                    continue

        return None

    def _normalize_achievement_key(self, key: Optional[str]) -> Optional[str]:
        """Ensure achievement keys carry the canonical 'achievement:' prefix."""
        if not key or not isinstance(key, str):
            return key
        return key if key.startswith("achievement:") else f"achievement:{key}"

    def _inline_ephemeral_requirements(self):
        """
        Preprocess skills to inline ephemeral requirements.

        For each skill that requires output from an ephemeral skill, replace that requirement
        with the ephemeral skill's requirements, properly combining lambda functions.

        Returns:
            Dict of processed skills with ephemeral requirements inlined
        """
        import copy
        processed_skills = copy.deepcopy(self.skills)

        print("Inlining ephemeral requirements...")

        # Find all ephemeral skills
        ephemeral_skills = {}
        for skill_name, skill_data in self.skills.items():
            skill_with_consumption = skill_data.get("skill_with_consumption", {})
            if skill_with_consumption.get("ephemeral", False):
                ephemeral_skills[skill_name] = skill_data
                print(f"Found ephemeral skill: {skill_name}")

        # For each non-ephemeral skill, check if it requires output from ephemeral skills
        for skill_name, skill_data in processed_skills.items():
            skill_with_consumption = skill_data["skill_with_consumption"]
            if skill_with_consumption.get("ephemeral", False):
                continue  # Skip ephemeral skills themselves

            requirements = skill_with_consumption.get("requirements", {})
            consumption = skill_with_consumption.get("consumption", {})
            new_requirements = {}
            new_consumption = {}

            # First, copy original consumption as baseline
            for cons_item, cons_lambda in consumption.items():
                new_consumption[cons_item] = cons_lambda

            # Process requirements (infrastructure/tools)
            for req_item, req_lambda in requirements.items():
                # Check if this is a floor OR condition (either already a list, or evals to one)
                req_lambda_is_list = isinstance(req_lambda, (list, tuple))
                if not req_lambda_is_list and isinstance(req_lambda, str):
                    # Try to eval and check if it's a list
                    try:
                        eval_result = eval(req_lambda)
                        if isinstance(eval_result, (list, tuple)):
                            req_lambda_is_list = True
                            req_lambda = eval_result  # Use the evaluated list
                    except:
                        pass

                # Skip inlining for level requirements with OR conditions (they're just floor checks)
                if req_item.startswith("level:") and req_lambda_is_list:
                    # Level requirements with OR conditions (e.g., [0, 2] for floor 0 OR 2)
                    # Just pass through as-is, no composition needed
                    new_requirements[req_item] = req_lambda
                    continue

                # Do not inline any level requirements (e.g., level:player_level)
                # Levels are location/state constraints, not resources produced by ephemeral skills.
                if req_item.startswith("level:"):
                    new_requirements[req_item] = req_lambda
                    continue

                # Check if this requirement comes from an ephemeral skill
                # IMPORTANT: Must check both KEY and VALUE match, not just the key!
                # E.g., don't inline "Descend to Floor 3" (provides level:player_level=3)
                # when requirement is level:player_level=1
                providing_ephemeral_skill = None
                req_value = self._evaluate_lambda_or_expr(req_lambda, 1)

                for ephemeral_name, ephemeral_data in ephemeral_skills.items():
                    ephemeral_gain = ephemeral_data["skill_with_consumption"].get("gain", {})
                    if req_item in ephemeral_gain:
                        # Check if the VALUE also matches
                        ephemeral_gain_entry = ephemeral_gain[req_item]
                        if isinstance(ephemeral_gain_entry, dict):
                            ephemeral_gain_expr = ephemeral_gain_entry.get("expression", "lambda n: n")
                        else:
                            ephemeral_gain_expr = ephemeral_gain_entry

                        ephemeral_value = self._evaluate_lambda_or_expr(ephemeral_gain_expr, 1)

                        # Only inline if the values match
                        if ephemeral_value == req_value:
                            providing_ephemeral_skill = ephemeral_name
                            print(f"  Found matching ephemeral skill '{ephemeral_name}' that provides '{req_item}'={ephemeral_value}")
                            break
                        else:
                            print(f"  Skipping ephemeral skill '{ephemeral_name}' - provides '{req_item}'={ephemeral_value}, but need {req_value}")

                if providing_ephemeral_skill:
                    print(f"Inlining ephemeral requirement '{req_item}' from '{providing_ephemeral_skill}' in skill '{skill_name}'")

                    eph_swc = ephemeral_skills[providing_ephemeral_skill]["skill_with_consumption"]
                    ephemeral_requirements = eph_swc.get("requirements", {})
                    ephemeral_consumption = eph_swc.get("consumption", {})

                    # If req_lambda is a floor OR condition (list), we can't compose - just copy ephemeral reqs directly
                    if req_lambda_is_list:
                        print(f"  Outer requirement is floor OR condition {req_lambda}, copying ephemeral reqs directly (no composition)")
                        for eph_req_item, eph_req_lambda in ephemeral_requirements.items():
                            # SKIP floor requirements - they're location constraints, not resources
                            if eph_req_item.startswith("level:player_level"):
                                print(f"    Skipping floor requirement '{eph_req_item}' (location constraint)")
                                continue
                            # Copy as-is
                            if eph_req_item not in new_requirements:
                                new_requirements[eph_req_item] = eph_req_lambda

                        for eph_cons_item, eph_cons_lambda in ephemeral_consumption.items():
                            # Copy as-is
                            if eph_cons_item not in new_consumption:
                                new_consumption[eph_cons_item] = eph_cons_lambda
                    else:
                        # Normal lambda composition for lambda requirements
                        # Inline ephemeral requirements
                        for eph_req_item, eph_req_lambda in ephemeral_requirements.items():
                            # SKIP floor requirements - they're location constraints, not resources
                            if eph_req_item.startswith("level:player_level"):
                                print(f"  Skipping floor requirement '{eph_req_item}' (location constraint, not a resource)")
                                continue

                            composed_lambda = self._compose_lambda_requirements(eph_req_lambda, req_lambda)
                            if eph_req_item in new_requirements:
                                new_requirements[eph_req_item] = self._combine_lambda_requirements(
                                    new_requirements[eph_req_item], composed_lambda
                                )
                            else:
                                new_requirements[eph_req_item] = composed_lambda

                        # Inline ephemeral consumption
                        for eph_cons_item, eph_cons_lambda in ephemeral_consumption.items():
                            composed_lambda = self._compose_lambda_requirements(eph_cons_lambda, req_lambda)
                            if eph_cons_item in new_consumption:
                                new_consumption[eph_cons_item] = self._combine_lambda_requirements(
                                    new_consumption[eph_cons_item], composed_lambda
                                )
                            else:
                                new_consumption[eph_cons_item] = composed_lambda
                else:
                    # Keep non-ephemeral requirements, combining if already exists
                    if req_item in new_requirements:
                        # Don't try to combine level OR conditions (lists) - just keep the existing one
                        if not req_lambda_is_list:
                            new_requirements[req_item] = self._combine_lambda_requirements(
                                new_requirements[req_item], req_lambda
                            )
                        # If it's a list, keep the existing requirement (no combination needed for OR conditions)
                    else:
                        new_requirements[req_item] = req_lambda

            # Update the skill's requirements and consumption
            processed_skills[skill_name]["skill_with_consumption"]["requirements"] = new_requirements
            processed_skills[skill_name]["skill_with_consumption"]["consumption"] = new_consumption
            print(f"Updated requirements for '{skill_name}': {new_requirements}")
            print(f"Updated consumption for '{skill_name}': {new_consumption}")

        return processed_skills

    def _compose_lambda_requirements(self, ephemeral_lambda, need_lambda):
        """
        Compose two lambda requirements by substituting need_lambda into ephemeral_lambda.

        If ephemeral requires "lambda n: c*n + d" and we need "lambda n: a*n + b" of it,
        the result is "lambda n: (c*n + d) * (a*n + b)"

        For simple cases like need="lambda n: 0*n + 1" and ephemeral="lambda n: 4*n",
        this becomes "lambda n: 4*1" = "lambda n: 4"
        """
        # FAIL LOUDLY if either is a list - can't compose floor OR conditions
        assert not isinstance(need_lambda, (list, tuple)), (
            f"Cannot compose lambda with floor OR condition (need_lambda={need_lambda})"
        )
        assert not isinstance(ephemeral_lambda, (list, tuple)), (
            f"Cannot compose lambda with floor OR condition (ephemeral_lambda={ephemeral_lambda})"
        )

        try:
            # Extract coefficients from both lambdas
            need_func = eval(need_lambda)
            eph_func = eval(ephemeral_lambda)

            # Extract a, b from need_lambda: f(n) = a*n + b
            need_b = need_func(0)  # Constant term
            need_a = need_func(1) - need_func(0)  # Coefficient of n

            # Extract c, d from ephemeral_lambda: g(n) = c*n + d
            eph_d = eph_func(0)  # Constant term
            eph_c = eph_func(1) - eph_func(0)  # Coefficient of n

            # For ephemeral_lambda(need_lambda(n)):
            # If need_lambda = a*n + b and ephemeral_lambda = c*n + d
            # Then result = c*(a*n + b) + d = c*a*n + c*b + d
            if need_a == 0:
                # Simple case: need constant amount b, so ephemeral_lambda(b) = c*b + d
                result_coeff = 0  # No scaling with n since we need constant amount
                result_const = eph_c * need_b + eph_d  # Constant: c*b + d

                if result_coeff == 0:
                    return f"lambda n: 0*n + {result_const}"
                elif result_const == 0:
                    return f"lambda n: {result_coeff}*n + 0"
                else:
                    return f"lambda n: {result_coeff}*n + {result_const}"
            else:
                # Complex case: would result in n^2 terms, fallback to evaluation at n=1
                result_amount = eph_func(need_func(1))
                return f"lambda n: 0*n + {result_amount}"

        except Exception as e:
            print(f"Error composing lambdas '{ephemeral_lambda}' and '{need_lambda}': {e}")
            return ephemeral_lambda

    def _combine_lambda_requirements(self, existing_lambda, new_lambda):
        """
        Combine two lambda requirements when a skill has multiple sources for the same resource.

        This combines lambda expressions of the form "lambda n: a*n + b" by adding their coefficients.
        """
        # Handle level OR conditions (lists) - these can't be combined, just keep the existing one
        if isinstance(existing_lambda, (list, tuple)) or isinstance(new_lambda, (list, tuple)):
            print(f"Warning: Cannot combine level OR conditions {existing_lambda} and {new_lambda}, keeping existing")
            return existing_lambda

        try:
            # Parse both lambda expressions to extract coefficients
            existing_func = eval(existing_lambda)
            new_func = eval(new_lambda)

            # Evaluate at n=0 and n=1 to extract a and b coefficients
            # For lambda n: a*n + b:
            # f(0) = b
            # f(1) = a + b
            # So: a = f(1) - f(0), b = f(0)

            existing_b = existing_func(0)  # Constant term
            existing_a = existing_func(1) - existing_func(0)  # Coefficient of n

            new_b = new_func(0)  # Constant term
            new_a = new_func(1) - new_func(0)  # Coefficient of n

            # Combine: (a1*n + b1) + (a2*n + b2) = (a1+a2)*n + (b1+b2)
            combined_a = existing_a + new_a
            combined_b = existing_b + new_b

            # Generate the combined lambda
            if combined_a == 0:
                return f"lambda n: 0*n + {combined_b}"
            elif combined_b == 0:
                return f"lambda n: {combined_a}*n + 0"
            else:
                return f"lambda n: {combined_a}*n + {combined_b}"

        except Exception as e:
            print(f"Error combining lambdas '{existing_lambda}' and '{new_lambda}': {e}")
            # Fallback: just keep the existing lambda
            return existing_lambda

    def _apply_inventory_constraints(self, execution_order, processed_skills):
        """
        Apply inventory capacity constraints by modifying the execution order.
        Split collection skills and insert deferred collection nodes where capacity allows.

        Args:
            execution_order: List of (skill_name, count) tuples
            processed_skills: Skills with ephemeral requirements inlined

        Returns:
            Modified execution order that respects inventory constraints
        """
        print(f"\nApplying inventory constraints (max capacity: {self.max_inventory_capacity})")

        # Track current inventory levels and deferred amounts
        inventory = {}
        deferred_collections = {}  # item_name -> amount_to_collect_later
        modified_execution = []

        for skill_name, count in execution_order:
            # Calculate what this skill would do to inventory
            gain = self._parse_lambda_gain(skill_name, count, processed_skills)
            consumption = self._parse_lambda_consumption(skill_name, count, processed_skills)

            # Check if we can execute this skill (consumption requirements met)
            can_execute = True
            for item, amount in consumption.items():
                if inventory.get(item, 0) < amount:
                    can_execute = False
                    break

            if not can_execute:
                print(f"Cannot execute {skill_name} yet - insufficient materials")
                modified_execution.append((skill_name, count))
                continue

            # Check if this skill's gains would exceed capacity
            needs_splitting = False
            for item, amount in gain.items():
                current_amount = inventory.get(item, 0)
                projected = (
                    max(current_amount, amount)
                    if item in TIERED_INVENTORY_ITEMS
                    else current_amount + amount
                )
                if projected > self.max_inventory_capacity:
                    needs_splitting = True
                    break

            if needs_splitting:
                # Split collection skills that would exceed capacity
                if len(gain) == 1:  # Simple collection skill with one output
                    item = list(gain.keys())[0]
                    amount = gain[item]
                    current_amount = inventory.get(item, 0)
                    max_can_collect = self.max_inventory_capacity - current_amount

                    if max_can_collect > 0:
                        # Collect what we can now
                        modified_execution.append((skill_name, max_can_collect))
                        inventory[item] = self.max_inventory_capacity

                        # Defer the rest
                        remaining = amount - max_can_collect
                        deferred_collections[item] = deferred_collections.get(item, 0) + remaining
                        print(f"Split {skill_name}: collect {max_can_collect} now, defer {remaining} '{item}'")
                    else:
                        # Can't collect any now, defer all
                        deferred_collections[item] = deferred_collections.get(item, 0) + amount
                        print(f"Defer all {amount} '{item}' from {skill_name}")
                else:
                    # Complex skill with multiple outputs - execute as-is for now
                    modified_execution.append((skill_name, count))
                    for item, amount in consumption.items():
                        inventory[item] = max(0, inventory.get(item, 0) - amount)
                    for item, amount in gain.items():
                        if item in TIERED_INVENTORY_ITEMS:
                            inventory[item] = max(inventory.get(item, 0), amount)
                        else:
                            inventory[item] = inventory.get(item, 0) + amount
            else:
                # Normal execution - no capacity issues
                modified_execution.append((skill_name, count))

                # Update inventory
                for item, amount in consumption.items():
                    inventory[item] = max(0, inventory.get(item, 0) - amount)
                for item, amount in gain.items():
                    if item in TIERED_INVENTORY_ITEMS:
                        inventory[item] = max(inventory.get(item, 0), amount)
                    else:
                        inventory[item] = inventory.get(item, 0) + amount

                print(f"Executed {skill_name} (n={count}), inventory: {inventory}")

            # After each skill, try to place deferred collections
            for item, deferred_amount in list(deferred_collections.items()):
                if deferred_amount > 0:
                    current_amount = inventory.get(item, 0)
                    can_collect = min(deferred_amount, self.max_inventory_capacity - current_amount)

                    if can_collect > 0:
                        # Find the collection skill for this item
                        # For tiered items, pass the tier needed
                        required_tier = can_collect if item in TIERED_INVENTORY_ITEMS else None
                        collection_skill = self._find_skill_providing_item(item, processed_skills, required_tier=required_tier)
                        if collection_skill:
                            modified_execution.append((collection_skill, can_collect))
                            if item in TIERED_INVENTORY_ITEMS:
                                inventory[item] = max(current_amount, can_collect)
                            else:
                                inventory[item] = current_amount + can_collect
                            deferred_collections[item] -= can_collect
                            print(f"Placed deferred collection: {collection_skill} (n={can_collect}) for '{item}'")

                            if deferred_collections[item] == 0:
                                del deferred_collections[item]

        # Handle any remaining deferred collections at the end
        for item, deferred_amount in deferred_collections.items():
            if deferred_amount > 0:
                # For tiered items, pass the tier needed
                required_tier = deferred_amount if item in TIERED_INVENTORY_ITEMS else None
                collection_skill = self._find_skill_providing_item(item, processed_skills, required_tier=required_tier)
                if collection_skill:
                    modified_execution.append((collection_skill, deferred_amount))
                    print(f"Added final deferred collection: {collection_skill} (n={deferred_amount}) for '{item}'")

        print(f"\nFinal execution order with inventory constraints:")
        for i, (skill_name, count) in enumerate(modified_execution):
            print(f"  {i+1}. {skill_name} (n={count})")

        return modified_execution

    def _combine_consecutive_skills(self, execution_order):
        """
        Combine consecutive identical skills into single entries.

        For example:
        [('Collect Wood', 4), ('Collect Wood', 4), ('Collect Wood', 1)]
        becomes:
        [('Collect Wood', 9)]

        Args:
            execution_order: List of (skill_name, count) tuples

        Returns:
            List with consecutive identical skills combined
        """
        if not execution_order:
            return execution_order

        print("\nCombining consecutive identical skills...")

        combined_order = []
        current_skill = execution_order[0][0]
        current_count = execution_order[0][1]

        for skill_name, count in execution_order[1:]:
            if skill_name == current_skill:
                # Same skill, add to current count
                current_count += count
                print(f"  Combining {skill_name}: {current_count - count} + {count} = {current_count}")
            else:
                # Different skill, save current and start new
                combined_order.append((current_skill, current_count))
                current_skill = skill_name
                current_count = count

        # Add the last skill
        combined_order.append((current_skill, current_count))

        print("Combined execution order:")
        for i, (skill_name, count) in enumerate(combined_order):
            print(f"  {i+1}. {skill_name} (n={count})")

        return combined_order

    def _convert_to_target_amounts(self, execution_order, processed_skills):
        """
        Convert execution counts to target inventory amounts.
        For collection skills, n should mean target inventory level, not execution count.

        Args:
            execution_order: List of (skill_name, count) tuples where count is execution count
            processed_skills: Skills with ephemeral requirements inlined

        Returns:
            List of (skill_name, target_amount) tuples where target_amount is desired inventory level
        """
        print(f"\nConverting execution counts to target inventory amounts...")

        # Track current inventory levels
        inventory = {}
        target_execution_order = []

        for skill_name, execution_count in execution_order:
            skills_to_use = processed_skills if processed_skills is not None else self.skills

            if skill_name not in skills_to_use:
                target_execution_order.append((skill_name, execution_count))
                continue

            # Calculate what this skill does to inventory
            gain = self._parse_lambda_gain(skill_name, execution_count, processed_skills)
            consumption = self._parse_lambda_consumption(skill_name, execution_count, processed_skills)

            # For skills that gain items, convert to target amount
            if gain:
                # Find the main item this skill produces
                main_item = max(gain.keys(), key=lambda x: gain[x]) if gain else None

                if main_item and gain[main_item] > 0:
                    # This is a collection/production skill
                    current_amount = inventory.get(main_item, 0)
                    if main_item in TIERED_INVENTORY_ITEMS:
                        target_amount = max(current_amount, gain[main_item])
                        projected_gain = target_amount - current_amount
                    else:
                        target_amount = current_amount + gain[main_item]
                        projected_gain = gain[main_item]

                    print(f"Skill '{skill_name}': execution_count={execution_count} -> target_amount={target_amount} for '{main_item}'")
                    print(f"  Current {main_item}: {current_amount}, will gain: {projected_gain}, target: {target_amount}")

                    target_execution_order.append((skill_name, target_amount))
                else:
                    # Not a collection skill, keep execution count
                    target_execution_order.append((skill_name, execution_count))
            else:
                # No gain, keep execution count
                target_execution_order.append((skill_name, execution_count))

            # Update inventory after this skill
            for item, amount in gain.items():
                if item in TIERED_INVENTORY_ITEMS:
                    inventory[item] = max(inventory.get(item, 0), amount)
                else:
                    inventory[item] = inventory.get(item, 0) + amount
            for item, amount in consumption.items():
                inventory[item] = max(0, inventory.get(item, 0) - amount)

            print(f"  Inventory after '{skill_name}': {inventory}")

        print(f"\nFinal execution order with target amounts:")
        for i, (skill_name, target_amount) in enumerate(target_execution_order):
            print(f"  {i+1}. {skill_name} (target={target_amount})")

        return target_execution_order


# Create alias for compatibility with symbolic state imports
SymbolicSkillDependencyResolver = SkillDependencyResolver

__all__ = [
    "SkillDependencyResolver",
    "SymbolicSkillDependencyResolver",
    "Node",
    "TIERED_INVENTORY_ITEMS",
    "configure_symbolic_state_module",
]
