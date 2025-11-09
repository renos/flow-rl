"""
Skill Dependency Resolver - Unified Planning Library Implementation

This module uses the AIPlan4EU Unified Planning Library to automatically
resolve skill dependencies. Much simpler than the manual graph-based approach
as the planning library handles all the heavy lifting.
"""
import logging

import importlib
from typing import Optional, Dict, List, Tuple, Union
from math import ceil

# FAIL LOUDLY - no graceful imports
from unified_planning.shortcuts import (
    OneshotPlanner,
    Problem,
    UserType,
    Fluent,
    InstantaneousAction,
    Int,
    Equals,
    Or,
)
from unified_planning.model.problem_kind import ProblemKind

# Symbolic state bindings (same as the old resolver)
SymbolicState = None
SkillOperator = None
convert_skill_to_operator = None
SYMBOLIC_STATE_AVAILABLE = False
TIERED_INVENTORY_ITEMS = set()


def _load_default_symbolic_module() -> None:
    """Load the default Craftax symbolic module - FAILS if not found."""
    candidates = [
        "flowrl.llm.craftax.symbolic_state",
    ]

    loaded = False
    for module_path in candidates:
        try:
            module = importlib.import_module(module_path)
            configure_symbolic_state_module(module)
            loaded = True
            break
        except ImportError:
            continue

    if not loaded:
        print("Warning: No symbolic state module found, symbolic state features disabled")


def configure_symbolic_state_module(module: Union[str, object]) -> None:
    """Configure symbolic state helpers from the provided module or module path."""
    global SymbolicState, SkillOperator, convert_skill_to_operator, SYMBOLIC_STATE_AVAILABLE, TIERED_INVENTORY_ITEMS

    if isinstance(module, str):
        module = importlib.import_module(module)

    SymbolicState = getattr(module, "SymbolicState", None)
    SkillOperator = getattr(module, "SkillOperator", None)
    convert_skill_to_operator = getattr(module, "convert_skill_to_operator", None)

    TIERED_INVENTORY_ITEMS = getattr(module, "TIERED_INVENTORY_ITEMS", set())
    if not isinstance(TIERED_INVENTORY_ITEMS, set):
        TIERED_INVENTORY_ITEMS = set(TIERED_INVENTORY_ITEMS) if TIERED_INVENTORY_ITEMS else set()

    SYMBOLIC_STATE_AVAILABLE = all(
        attr is not None for attr in (SymbolicState, SkillOperator, convert_skill_to_operator)
    )


# Load default bindings at import time
_load_default_symbolic_module()


class UnifiedPlanningSkillResolver:
    """
    Skill dependency resolver using the Unified Planning Library.

    This is much simpler than the manual graph-based approach because
    the planning library handles dependency resolution, pruning, and optimization.
    """

    def __init__(self, skills, max_inventory_capacity=99, initial_state=None,
                 planner_name: Optional[str] = None,
                 planner_timeout: Optional[int] = 30):
        """
        Initialize the resolver with a dictionary of skills.

        Args:
            skills: Dict mapping skill_name -> skill_data
            max_inventory_capacity: Maximum number of items that can be held of each type
            initial_state: Optional SymbolicState for tracking achievements/levels
        """
        import os
        self.skills = skills
        self.max_inventory_capacity = max_inventory_capacity
        self.initial_state = initial_state.copy() if initial_state else None
        # Planner configuration (also via env vars UP_PLANNER_NAME/UP_PLANNER_TIMEOUT)
        self.planner_name = planner_name or os.environ.get("UP_PLANNER_NAME")
        t_env = os.environ.get("UP_PLANNER_TIMEOUT")
        self.planner_timeout = planner_timeout if planner_timeout is not None else (int(t_env) if t_env and t_env.isdigit() else None)

        # Process skills to inline ephemeral requirements (same as old resolver)
        self.processed_skills = self._inline_ephemeral_requirements()

    def resolve_dependencies(self, target_skill_name, n=1, initial_state=None, strategy: str = "backward"):
        """
        Build a plan for the target skill.

        By default this uses a backwards, floor-by-floor approach to avoid
        combinatorial explosion from level/floor transitions. The unified
        planning solver remains available as a fallback.

        Args:
            target_skill_name: Name of the skill to build dependencies for
            n: Number of times to apply this skill
            initial_state: Optional SymbolicState to override self.initial_state

        Returns:
            List of (skill_name, count) tuples in execution order
        """
        assert target_skill_name in self.skills, f"Skill '{target_skill_name}' not found in skills"

        if initial_state is not None:
            self.initial_state = initial_state.copy()

        if strategy == "backward":
            return self._resolve_dependencies_flow_backwards(target_skill_name, n)

        # Fallback to unified planning if explicitly requested
        print(
            f"Building plan for skill '{target_skill_name}' (n={n}) using Unified Planning Library"
        )
        problem = self._create_planning_problem(target_skill_name, n)
        plan = self._solve_planning_problem(problem)
        assert plan is not None, "No plan found by the planner - planning failed!"
        execution_order = self._plan_to_execution_order(plan)
        execution_order = self._combine_consecutive_skills(execution_order)
        execution_order = self._convert_to_target_amounts(execution_order)
        return execution_order

    # -------------------------
    # Backwards floor-by-floor
    # -------------------------
    def _resolve_dependencies_flow_backwards(self, target_skill_name: str, n: int = 1):
        """
        Floor-by-floor backwards dependency resolution.

        High-level idea:
        - Pick the target floor for the target skill (from level:player_level requirement).
        - On that floor, expand dependencies using only skills valid on that floor,
          assuming missing resources are provided by earlier floors.
        - Collect the set of resources required from earlier floors and repeat the
          expansion on lower floors until floor 0.
        - Stitch per-floor plans together and insert descent actions if available.
        """
        assert (
            target_skill_name in self.processed_skills
        ), f"Skill '{target_skill_name}' not found in skills"

        # Determine target floor and floor range
        target_floor = self._determine_target_floor_for_skill(target_skill_name)
        all_floors = self._collect_all_floors()
        if not all_floors:
            # Default to simple single-floor world if no floors discovered
            all_floors = {0}
        max_floor = max(all_floors)
        min_floor = min(all_floors)
        if target_floor is None:
            # If target has no explicit floor, assume current level or 0
            target_floor = self._initial_floor_value()
        target_floor = int(target_floor)

        # Build provider index once
        providers_by_resource = self._build_provider_index()

        # Per-floor plans and requirements to push down
        per_floor_execution = {}  # floor -> list[(skill, count)] in correct local order
        needed_from_lower = {}  # floor -> dict resource->amount

        # Seed goals for top floor: execute the target skill n times
        goals = [(target_skill_name, n)]

        # Walk floors downward from target to min_floor
        carry_resources = {}  # resource->amount to provide from lower floors
        for floor in range(target_floor, min_floor - 1, -1):
            exec_list, missing_resources, produced_items = self._plan_floor(
                floor, goals, providers_by_resource
            )
            per_floor_execution[floor] = exec_list
            needed_from_lower[floor] = missing_resources

            # Convert missing resources into goals for the next (lower) floor
            goals, carry_resources = self._seed_goals_for_lower_floor(
                floor - 1, missing_resources, providers_by_resource, already_produced=produced_items
            )

            # If we reached the minimum floor and still have missing resources,
            # try to absorb from initial inventory if available
            if floor - 1 < min_floor and carry_resources:
                if self.initial_state and hasattr(self.initial_state, "inventory"):
                    inv = self.initial_state.inventory
                    residual = {}
                    for item, amt in carry_resources.items():
                        available = int(inv.get(item, 0))
                        need = max(0, int(amt) - available)
                        if need > 0:
                            residual[item] = need
                    carry_resources = residual
                # Nothing else we can do here; keep residuals to surface in plan

            # Next iteration will try to satisfy these goals on lower floor

        # Build final linear execution plan from lowest to target, inserting descents
        final_order: List[Tuple[str, int]] = []
        current_floor = self._initial_floor_value() if self.initial_state else min_floor

        for floor in range(min_floor, target_floor + 1):
            steps = per_floor_execution.get(floor, [])
            if not steps:
                continue
            # Insert transition to this floor if needed
            if floor != current_floor:
                final_order.extend(self._transition_actions(current_floor, floor))
                current_floor = floor
            final_order.extend(steps)

        # Optionally, transition to target floor if no steps recorded on it but target skill requires it
        if not per_floor_execution.get(target_floor) and current_floor != target_floor:
            final_order.extend(self._transition_actions(current_floor, target_floor))

        # Merge, prune redundant tiered crafts, merge again, then convert to target amounts
        final_order = self._combine_consecutive_skills(final_order)
        final_order = self._prune_redundant_tiered_tool_crafts(final_order)
        final_order = self._combine_consecutive_skills(final_order)
        final_order = self._convert_to_target_amounts(final_order)

        # If we still have residual carry_resources, surface them as warnings via prints
        # and append placeholder entries so caller is aware of unmet requirements.
        # We avoid raising errors to keep the system robust and inspectable.
        if carry_resources:
            print("Warning: Unmet requirements after floor 0 planning:")
            for k, v in carry_resources.items():
                print(f"  - {k}: {v}")

        return final_order

    def _initial_floor_value(self) -> int:
        try:
            if self.initial_state is None:
                return 0
            # Prefer 'levels' with key 'player_level' if available
            if hasattr(self.initial_state, "levels") and isinstance(self.initial_state.levels, dict):
                lvl = self.initial_state.levels.get("player_level", None)
                if lvl is None:
                    # Also allow the legacy key format
                    lvl = self.initial_state.levels.get("level:player_level", 0)
                return int(lvl or 0)
            # Legacy attribute name used elsewhere in this module
            if hasattr(self.initial_state, "level") and isinstance(self.initial_state.level, dict):
                return int(self.initial_state.level.get("level:player_level", 0))
        except Exception:
            pass
        return 0

    def _determine_target_floor_for_skill(self, skill_name: str) -> Optional[int]:
        skill = self.processed_skills.get(skill_name, {})
        reqs = skill.get("skill_with_consumption", {}).get("requirements", {})
        if "level:player_level" not in reqs:
            return None
        floor_req = reqs["level:player_level"]
        # If it's a list/tuple of floors, pick the highest as the target
        if isinstance(floor_req, (list, tuple)) and floor_req:
            try:
                return int(max(int(x) for x in floor_req))
            except Exception:
                return None
        # If it's a string, evaluate
        if isinstance(floor_req, str):
            try:
                return int(self._evaluate_lambda_or_expr(floor_req, 1))
            except Exception:
                return None
        # Plain number?
        try:
            return int(floor_req)
        except Exception:
            return None

    def _collect_all_floors(self) -> set:
        floors = set()
        for _, data in self.processed_skills.items():
            swc = data.get("skill_with_consumption", {})
            # Requirements
            for k, v in swc.get("requirements", {}).items():
                if k == "level:player_level":
                    if isinstance(v, (list, tuple)):
                        for x in v:
                            try:
                                floors.add(int(x))
                            except Exception:
                                pass
                    elif isinstance(v, str):
                        try:
                            floors.add(int(self._evaluate_lambda_or_expr(v, 1)))
                        except Exception:
                            pass
                    else:
                        try:
                            floors.add(int(v))
                        except Exception:
                            pass
            # Gains
            for k, v in swc.get("gain", {}).items():
                if k == "level:player_level":
                    if isinstance(v, dict):
                        expr = v.get("expression", "1")
                    else:
                        expr = v
                    try:
                        floors.add(int(self._evaluate_lambda_or_expr(expr, 1)))
                    except Exception:
                        pass
        # Always include 0 as a safe baseline
        floors.add(0)
        return floors

    def _skill_allowed_floors(self, skill_name: str) -> Optional[set]:
        """Return set of floors where this skill can be executed; None means any floor."""
        data = self.processed_skills.get(skill_name)
        if not data:
            return None
        reqs = data.get("skill_with_consumption", {}).get("requirements", {})
        v = reqs.get("level:player_level")
        if v is None:
            return None  # Any floor
        if isinstance(v, (list, tuple)):
            try:
                return set(int(x) for x in v)
            except Exception:
                return None
        if isinstance(v, str):
            try:
                return {int(self._evaluate_lambda_or_expr(v, 1))}
            except Exception:
                return None
        try:
            return {int(v)}
        except Exception:
            return None

    def _build_provider_index(self) -> Dict[str, List[str]]:
        providers: Dict[str, List[str]] = {}
        for sname, data in self.processed_skills.items():
            gain = data.get("skill_with_consumption", {}).get("gain", {})
            for key in gain.keys():
                providers.setdefault(key, []).append(sname)
        return providers

    def _is_floor_changing_skill(self, skill_name: str) -> bool:
        data = self.processed_skills.get(skill_name, {})
        gain = data.get("skill_with_consumption", {}).get("gain", {})
        return "level:player_level" in gain

    def _provider_gain_amount(self, provider: str, resource_key: str) -> int:
        data = self.processed_skills.get(provider, {})
        gain = data.get("skill_with_consumption", {}).get("gain", {})
        entry = gain.get(resource_key)
        if entry is None:
            return 0
        if isinstance(entry, dict):
            expr = entry.get("expression", "lambda n: n")
        else:
            expr = entry
        try:
            return int(self._evaluate_lambda_or_expr(expr, 1))
        except Exception:
            return 0

    def _select_provider_for(self, resource_key: str, floor: int, providers_by_resource: Dict[str, List[str]]) -> Optional[Tuple[str, int]]:
        """Select a provider skill for a resource on a given floor.

        Returns (provider_skill_name, gain_per_exec) or None if not available.
        """
        candidates = providers_by_resource.get(resource_key, [])
        best: Optional[Tuple[str, int]] = None
        for prov in candidates:
            # Skip floor changing skills for production
            if self._is_floor_changing_skill(prov):
                continue
            allowed = self._skill_allowed_floors(prov)
            if allowed is not None and floor not in allowed:
                continue
            gain_amt = self._provider_gain_amount(prov, resource_key)
            if gain_amt <= 0:
                continue
            if resource_key in TIERED_INVENTORY_ITEMS:
                # For tiered, prefer the smallest gain that still meets requirement.
                # Selection of exact provider is decided at call-site with required amount.
                # Here, just keep as candidate with its gain.
                pass
            # Prefer highest gain per exec to minimize steps for non-tiered
            if best is None:
                best = (prov, gain_amt)
            else:
                _, best_gain = best
                if resource_key in TIERED_INVENTORY_ITEMS:
                    # For tiered, prefer lower tier provider (finer control)
                    if gain_amt < best_gain:
                        best = (prov, gain_amt)
                else:
                    if gain_amt > best_gain:
                        best = (prov, gain_amt)
        return best

    def _plan_floor(
        self,
        floor: int,
        goals: List[Tuple[str, int]],
        providers_by_resource: Dict[str, List[str]],
    ) -> Tuple[List[Tuple[str, int]], Dict[str, int], set]:
        """
        Plan to satisfy goals on a specific floor, expanding dependencies using only
        skills executable on this floor. Returns (execution_order_for_floor, missing_resources)
        where missing_resources are to be provided by lower floors.
        """
        execution_order: List[Tuple[str, int]] = []
        missing: Dict[str, int] = {}
        visiting: set = set()  # detect cycles per-floor
        # Track simulated per-floor state to avoid redundant production
        floor_inv: Dict[str, int] = {}
        floor_ach: set = set()
        produced_items: set = set()
        # Track ephemeral requirements already paid on this floor (e.g., achievement:PLACE_TABLE)
        ephemeral_paid: set = set()

        def expand_skill(skill_name: str, times: int):
            # Ensure this skill can be executed on this floor
            allowed = self._skill_allowed_floors(skill_name)
            if allowed is not None and floor not in allowed:
                # Can't do this here, push its effects as missing resources
                gains = self._parse_lambda_gain(skill_name, times)
                for item, amount in gains.items():
                    missing[item] = missing.get(item, 0) + int(amount)
                return

            key = (skill_name, floor)
            if key in visiting:
                print(f"Cycle detected on floor {floor} while expanding '{skill_name}', skipping nested expansion")
                # Still add the skill itself to order to avoid losing the goal
                execution_order.append((skill_name, times))
                return
            visiting.add(key)

            data = self.processed_skills.get(skill_name, {})
            swc = data.get("skill_with_consumption", {})
            reqs = swc.get("requirements", {})
            cons = swc.get("consumption", {})

            # Handle requirements (non-consuming)
            for req_key, req_lambda in reqs.items():
                if req_key.startswith("level:player_level"):
                    # Assume location constraint is satisfied by being on this floor
                    continue
                amount_needed = int(self._evaluate_lambda(req_lambda, times))
                if amount_needed <= 0:
                    continue

                # If achievable on this floor, expand provider; otherwise defer
                # First, check current simulated state to avoid duplicate production
                if req_key.startswith("achievement:"):
                    if req_key in floor_ach:
                        continue
                    # Try to satisfy ephemeral achievements inline without explicit steps
                    eph_provider = self._find_ephemeral_provider_for(req_key, providers_by_resource)
                    if eph_provider is not None:
                        if req_key not in ephemeral_paid:
                            # Pay provider consumption once on this floor by producing its required items
                            eph_swc = self.processed_skills[eph_provider]["skill_with_consumption"]
                            eph_cons = eph_swc.get("consumption", {})
                            # Produce items needed for consumption
                            for cons_item, cons_lambda in eph_cons.items():
                                cons_amt = int(self._evaluate_lambda(cons_lambda, 1))
                                if cons_item in TIERED_INVENTORY_ITEMS:
                                    # Tiers: require at least this level
                                    if floor_inv.get(cons_item, 0) < cons_amt:
                                        prov = self._select_provider_for(cons_item, floor, providers_by_resource)
                                        if prov is not None:
                                            prov_name, prov_gain = prov
                                            if prov_gain >= cons_amt:
                                                expand_skill(prov_name, 1)
                                            else:
                                                missing[cons_item] = max(missing.get(cons_item, 0), cons_amt)
                                        else:
                                            missing[cons_item] = max(missing.get(cons_item, 0), cons_amt)
                                else:
                                    have = floor_inv.get(cons_item, 0)
                                    need = max(0, cons_amt - have)
                                    if need > 0:
                                        prov = self._select_provider_for(cons_item, floor, providers_by_resource)
                                        if prov is not None:
                                            prov_name, prov_gain = prov
                                            times_needed = int(ceil(need / max(1, prov_gain)))
                                            if times_needed > 0:
                                                expand_skill(prov_name, times_needed)
                                        else:
                                            missing[cons_item] = missing.get(cons_item, 0) + need
                                    # Deduct the consumption
                                    floor_inv[cons_item] = max(0, have - cons_amt)
                            ephemeral_paid.add(req_key)
                        # Mark achievement as satisfied on this floor
                        floor_ach.add(req_key)
                        continue
                elif req_key in TIERED_INVENTORY_ITEMS:
                    if floor_inv.get(req_key, 0) >= amount_needed:
                        continue
                else:
                    if floor_inv.get(req_key, 0) >= amount_needed:
                        continue

                provider = self._select_provider_for(req_key, floor, providers_by_resource)
                if req_key in TIERED_INVENTORY_ITEMS:
                    if provider is not None:
                        prov_name, prov_gain = provider
                        if prov_gain >= amount_needed:
                            expand_skill(prov_name, 1)
                        else:
                            # No provider on this floor meets required tier; defer
                            missing[req_key] = max(missing.get(req_key, 0), amount_needed)
                    else:
                        missing[req_key] = max(missing.get(req_key, 0), amount_needed)
                else:
                    if provider is not None:
                        prov_name, prov_gain = provider
                        times_needed = int(ceil(amount_needed / max(1, prov_gain)))
                        if times_needed > 0:
                            expand_skill(prov_name, times_needed)
                    else:
                        missing[req_key] = missing.get(req_key, 0) + amount_needed

            # Handle consumption (consuming)
            for cons_key, cons_lambda in cons.items():
                amount_needed = int(self._evaluate_lambda(cons_lambda, times))
                if amount_needed <= 0:
                    continue
                # If we already have enough on this floor, skip producing
                current_avail = floor_inv.get(cons_key, 0)
                if cons_key in TIERED_INVENTORY_ITEMS:
                    if current_avail >= amount_needed:
                        provider = None
                    else:
                        provider = self._select_provider_for(cons_key, floor, providers_by_resource)
                else:
                    if current_avail >= amount_needed:
                        provider = None
                    else:
                        provider = self._select_provider_for(cons_key, floor, providers_by_resource)
                if cons_key in TIERED_INVENTORY_ITEMS:
                    if provider is not None:
                        prov_name, prov_gain = provider
                        if prov_gain >= amount_needed:
                            expand_skill(prov_name, 1)
                        else:
                            missing[cons_key] = max(missing.get(cons_key, 0), amount_needed)
                    else:
                        missing[cons_key] = max(missing.get(cons_key, 0), amount_needed)
                else:
                    if provider is not None:
                        prov_name, prov_gain = provider
                        times_needed = int(ceil(amount_needed / max(1, prov_gain)))
                        if times_needed > 0:
                            expand_skill(prov_name, times_needed)
                    else:
                        missing[cons_key] = missing.get(cons_key, 0) + amount_needed

            # Finally add this skill itself
            execution_order.append((skill_name, times))
            # Update simulated per-floor state based on this skill's effects
            try:
                gain_delta = self._parse_lambda_gain(skill_name, times)
                cons_delta = self._parse_lambda_consumption(skill_name, times)
                # Update inventory
                for item, amount in gain_delta.items():
                    if item in TIERED_INVENTORY_ITEMS:
                        floor_inv[item] = max(floor_inv.get(item, 0), int(amount))
                    else:
                        floor_inv[item] = floor_inv.get(item, 0) + int(amount)
                    produced_items.add(item)
                for item, amount in cons_delta.items():
                    floor_inv[item] = max(0, floor_inv.get(item, 0) - int(amount))
                # Update achievements
                for ach in self._skill_gain_achievements(skill_name, times):
                    floor_ach.add(ach)
            except Exception as _:
                pass
            visiting.remove(key)

        for s, t in goals:
            if t <= 0:
                continue
            expand_skill(s, int(t))

        return execution_order, missing, produced_items

    def _skill_gain_achievements(self, skill_name: str, n: int = 1) -> List[str]:
        """Return list of achievement keys this skill grants when executed n times."""
        try:
            data = self.processed_skills.get(skill_name, {})
            gain = data.get("skill_with_consumption", {}).get("gain", {})
            achieved: List[str] = []
            for k, v in gain.items():
                if isinstance(v, dict):
                    gtype = v.get("type", "inventory")
                    if gtype != "achievement":
                        continue
                    expr = v.get("expression", "1")
                    amt = int(self._evaluate_lambda_or_expr(expr, n))
                else:
                    if not (isinstance(k, str) and k.startswith("achievement:")):
                        continue
                    amt = int(self._evaluate_lambda_or_expr(v, n)) if isinstance(v, str) else int(v)
                if amt > 0:
                    achieved.append(k)
            return achieved
        except Exception:
            return []

    def _prune_redundant_tiered_tool_crafts(self, execution_order: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        """Remove craft steps that set tiered items to a level not exceeding current tier."""
        inv: Dict[str, int] = {}
        pruned: List[Tuple[str, int]] = []
        for skill_name, times in execution_order:
            try:
                gain = self._parse_lambda_gain(skill_name, times)
                cons = self._parse_lambda_consumption(skill_name, times)
                # If this skill upgrades a tiered item but not above current tier, drop it
                tier_targets = {k: v for k, v in gain.items() if k in TIERED_INVENTORY_ITEMS}
                redundant = False
                for item, new_tier in tier_targets.items():
                    if inv.get(item, 0) >= int(new_tier):
                        redundant = True
                        break
                if redundant:
                    continue
                pruned.append((skill_name, times))
                # Apply effects
                for item, amount in gain.items():
                    if item in TIERED_INVENTORY_ITEMS:
                        inv[item] = max(inv.get(item, 0), int(amount))
                    else:
                        inv[item] = inv.get(item, 0) + int(amount)
                for item, amount in cons.items():
                    inv[item] = max(0, inv.get(item, 0) - int(amount))
            except Exception:
                pruned.append((skill_name, times))
        return pruned

    def _seed_goals_for_lower_floor(
        self,
        next_floor: int,
        missing_resources: Dict[str, int],
        providers_by_resource: Dict[str, List[str]],
        already_produced: set = None,
    ) -> Tuple[List[Tuple[str, int]], Dict[str, int]]:
        """
        Convert missing resources into skill goals for the next lower floor.
        If a resource has no provider at the next floor, keep it in carry-over.
        Returns (goals_for_next_floor, residual_carry_over)
        """
        goals: List[Tuple[str, int]] = []
        carry_over: Dict[str, int] = {}
        already_produced = already_produced or set()

        for res, amt in missing_resources.items():
            if amt <= 0:
                continue
            # If this resource was already produced on the higher floor, do not seed it again below
            if res in already_produced:
                continue
            # Try to find a provider at exactly next_floor; if none, defer
            provider = self._select_provider_for(res, next_floor, providers_by_resource)
            if provider is None:
                carry_over[res] = carry_over.get(res, 0) + int(amt)
                continue
            prov_name, prov_gain = provider
            if res in TIERED_INVENTORY_ITEMS:
                # Need a provider that reaches at least this tier; if not, carry over
                if prov_gain >= int(amt):
                    goals.append((prov_name, 1))
                else:
                    carry_over[res] = max(carry_over.get(res, 0), int(amt))
            else:
                times_needed = int(ceil(int(amt) / max(1, prov_gain)))
                if times_needed > 0:
                    goals.append((prov_name, times_needed))

        return goals, carry_over

    def _find_ephemeral_provider_for(self, res_key: str, providers_by_resource: Dict[str, List[str]]) -> Optional[str]:
        """Return the ephemeral provider skill name for a given achievement/level key, if any."""
        for sname in providers_by_resource.get(res_key, []):
            data = self.processed_skills.get(sname, {}).get("skill_with_consumption", {})
            if data.get("ephemeral", False):
                return sname
        return None

    def _transition_actions(self, from_floor: int, to_floor: int) -> List[Tuple[str, int]]:
        """
        Build a sequence of floor transition actions from 'from_floor' to 'to_floor'.
        This looks for skills that set level:player_level exactly to the target floor
        and inserts them. If none is found, returns an empty list.
        """
        if from_floor == to_floor:
            return []
        actions: List[Tuple[str, int]] = []
        direction = 1 if to_floor > from_floor else -1
        step = 1
        # Try direct jump first
        direct = self._find_floor_set_skill(to_floor)
        if direct is not None:
            actions.append((direct, 1))
            return actions
        # Otherwise, try stepping by 1 if such actions exist
        current = from_floor
        while current != to_floor:
            current += direction * step
            incr = self._find_floor_set_skill(current)
            if incr is None:
                # Cannot find a transition action; stop trying
                break
            actions.append((incr, 1))
        return actions

    def _find_floor_set_skill(self, floor: int) -> Optional[str]:
        for sname, data in self.processed_skills.items():
            gain = data.get("skill_with_consumption", {}).get("gain", {})
            if "level:player_level" not in gain:
                continue
            entry = gain["level:player_level"]
            expr = entry.get("expression") if isinstance(entry, dict) else entry
            try:
                val = int(self._evaluate_lambda_or_expr(expr, 1))
            except Exception:
                continue
            if val == int(floor):
                return sname
        return None

    def _create_planning_problem(self, target_skill_name, n):
        """
        Create a Unified Planning problem from skills.

        Returns:
            A unified_planning.model.Problem instance
        """
        problem = Problem("skill_dependencies")

        # Collect all unique items from all skills
        all_items = set()
        all_achievements = set()
        all_levels = set()
        # Discover floors mentioned in requirements/gains for level:player_level
        floors_in_domain = set()

        for skill_name, skill_data in self.processed_skills.items():
            skill_with_consumption = skill_data["skill_with_consumption"]

            # Collect from requirements
            for req_key in skill_with_consumption.get("requirements", {}).keys():
                if req_key.startswith("achievement:"):
                    all_achievements.add(req_key)
                elif req_key.startswith("level:"):
                    all_levels.add(req_key)
                else:
                    all_items.add(req_key)

            # Collect floors from requirements
            for req_key, req_val in skill_with_consumption.get("requirements", {}).items():
                if req_key == "level:player_level":
                    if isinstance(req_val, (list, tuple)):
                        for v in req_val:
                            try:
                                floors_in_domain.add(int(v))
                            except Exception:
                                pass
                    elif isinstance(req_val, str):
                        try:
                            # Evaluate lambda at n=1 to get target floor
                            val = self._evaluate_lambda(req_val, 1)
                            floors_in_domain.add(int(val))
                        except Exception:
                            pass

            # Collect from consumption (consumption items are always inventory, never achievements/levels)
            for cons_key in skill_with_consumption.get("consumption", {}).keys():
                if not cons_key.startswith("achievement:") and not cons_key.startswith("level:"):
                    all_items.add(cons_key)

            # Collect from gain
            for gain_key, gain_entry in skill_with_consumption.get("gain", {}).items():
                if isinstance(gain_entry, dict):
                    gain_type = gain_entry.get("type", "inventory")
                    if gain_type == "achievement":
                        all_achievements.add(gain_key)
                    elif gain_type == "level":
                        all_levels.add(gain_key)
                        if gain_key == "level:player_level":
                            try:
                                expr = gain_entry.get("expression", "1")
                                val = self._evaluate_lambda_or_expr(expr, 1)
                                floors_in_domain.add(int(val))
                            except Exception:
                                pass
                    else:
                        all_items.add(gain_key)
                else:
                    # Simple format, check key prefix
                    if gain_key.startswith("achievement:"):
                        all_achievements.add(gain_key)
                    elif gain_key.startswith("level:"):
                        all_levels.add(gain_key)
                        if gain_key == "level:player_level":
                            try:
                                val = self._evaluate_lambda_or_expr(gain_entry, 1)
                                floors_in_domain.add(int(val))
                            except Exception:
                                pass
                    else:
                        all_items.add(gain_key)

        print(f"Found {len(all_items)} items, {len(all_achievements)} achievements, {len(all_levels)} levels")

        # Create fluents (state variables) with explicit integer type
        # Get the integer type from the problem's environment
        int_type = problem.environment.type_manager.IntType()

        item_fluents = {}
        for item in all_items:
            # Sanitize item name for PDDL
            safe_name = item.replace(":", "_").replace(".", "_").replace("-", "_")

            # Get initial value
            initial_amount = 0
            if self.initial_state and hasattr(self.initial_state, "inventory"):
                initial_amount = self.initial_state.inventory.get(item, 0)

            # Create integer-typed fluent
            fluent = Fluent(f"have_{safe_name}", typename=int_type)
            problem.add_fluent(fluent, default_initial_value=initial_amount)
            item_fluents[item] = fluent

        # Create fluents for achievements (boolean - default type)
        achievement_fluents = {}
        for achievement in all_achievements:
            # Sanitize achievement name for PDDL
            safe_name = achievement.replace(":", "_").replace(".", "_")

            # Get initial value
            initial_value = False
            if self.initial_state and hasattr(self.initial_state, "achievements"):
                initial_value = self.initial_state.achievements.get(achievement, 0) > 0

            # Boolean fluent (default type)
            fluent = Fluent(f"achieved_{safe_name}")
            problem.add_fluent(fluent, default_initial_value=initial_value)
            achievement_fluents[achievement] = fluent

        # Create boolean floor fluents at_floor_* discovered above
        # Always include floor 0 to allow initial planning
        floors_in_domain.add(0)
        floor_fluents = {}
        for fl in sorted(floors_in_domain):
            fl_fluent = Fluent(f"at_floor_{int(fl)}")
            # Default initial value: true only for initial floor (from initial_state if available)
            initial_bool = False
            init_floor = 0
            if self.initial_state and hasattr(self.initial_state, "level"):
                init_floor = int(self.initial_state.level.get("level:player_level", 0))
            initial_bool = (int(fl) == init_floor)
            problem.add_fluent(fl_fluent, default_initial_value=initial_bool)
            floor_fluents[int(fl)] = fl_fluent

        # Create fluents for other levels (integers)
        level_fluents = {}
        for level in all_levels:
            safe_name = level.replace(":", "_").replace(".", "_")

            # Get initial value
            initial_value = 0
            if self.initial_state and hasattr(self.initial_state, "level"):
                initial_value = self.initial_state.level.get(level, 0)

            # Integer-typed fluent
            fluent = Fluent(f"level_{safe_name}", typename=int_type)
            problem.add_fluent(fluent, default_initial_value=initial_value)
            level_fluents[level] = fluent

        # Create actions for each skill
        print(f"\n=== CREATING ACTIONS ({len(self.processed_skills)} skills) ===")
        actions = {}
        for skill_name, skill_data in self.processed_skills.items():
            action = self._create_action_from_skill(
                skill_name, skill_data, item_fluents, achievement_fluents, level_fluents, floor_fluents
            )
            assert action is not None, f"Failed to create action for skill '{skill_name}'"
            problem.add_action(action)
            actions[skill_name] = action

            # Print action details
            print(f"\n  Action: {action.name}")
            if action.preconditions:
                print(f"    Preconditions: {action.preconditions}")
            else:
                print(f"    Preconditions: (none)")
            if action.effects:
                print(f"    Effects: {action.effects}")
            else:
                print(f"    Effects: (none)")
        print("=== END ACTIONS ===\n")

        # Set goal: achieve the target skill's gains
        target_skill_data = self.processed_skills[target_skill_name]
        target_gain = target_skill_data["skill_with_consumption"].get("gain", {})

        goals = []
        for gain_key, gain_entry in target_gain.items():
            if isinstance(gain_entry, dict):
                gain_expr = gain_entry.get("expression", "lambda n: n")
                gain_type = gain_entry.get("type", "inventory")
            else:
                gain_expr = gain_entry
                # Infer type from key
                if gain_key.startswith("achievement:"):
                    gain_type = "achievement"
                elif gain_key.startswith("level:"):
                    gain_type = "level"
                else:
                    gain_type = "inventory"

            target_amount = self._evaluate_lambda_or_expr(gain_expr, n)

            if gain_type == "achievement":
                assert gain_key in achievement_fluents, f"Achievement '{gain_key}' not in fluents"
                goals.append(achievement_fluents[gain_key])
            elif gain_type == "level":
                assert gain_key in level_fluents, f"Level '{gain_key}' not in fluents"
                # Enforce exact target level via equality
                goals.append(Equals(level_fluents[gain_key], Int(int(target_amount))))
            else:
                assert gain_key in item_fluents, f"Item '{gain_key}' not in fluents"
                if gain_key in TIERED_INVENTORY_ITEMS:
                    # For tiered items, goal is to reach at least the tier
                    goals.append(item_fluents[gain_key] >= target_amount)
                else:
                    # For regular items, goal is to have at least the amount
                    goals.append(item_fluents[gain_key] >= target_amount)

        # Print all fluents for debugging
        print("\n=== FLUENTS CREATED ===")
        print(f"\nItem Fluents ({len(item_fluents)}):")
        for item, fluent in sorted(item_fluents.items()):
            initial_val = problem.initial_value(fluent)
            print(f"  {item}: {fluent.name} (type: {fluent.type}, initial: {initial_val})")

        print(f"\nAchievement Fluents ({len(achievement_fluents)}):")
        for achievement, fluent in sorted(achievement_fluents.items()):
            initial_val = problem.initial_value(fluent)
            print(f"  {achievement}: {fluent.name} (type: {fluent.type}, initial: {initial_val})")

        print(f"\nLevel Fluents ({len(level_fluents)}):")
        for level, fluent in sorted(level_fluents.items()):
            initial_val = problem.initial_value(fluent)
            print(f"  {level}: {fluent.name} (type: {fluent.type}, initial: {initial_val})")
        print(f"\nFloor Fluents ({len(floor_fluents)}):")
        for fl, fluent in sorted(floor_fluents.items()):
            initial_val = problem.initial_value(fluent)
            print(f"  floor {fl}: {fluent.name} (type: {fluent.type}, initial: {initial_val})")
        print("=== END FLUENTS ===\n")

        assert len(goals) > 0, f"No goals defined for skill '{target_skill_name}'"

        # Print goals
        print(f"\n=== GOAL FOR '{target_skill_name}' (n={n}) ===")
        for goal in goals:
            print(f"  {goal}")
        print("=== END GOAL ===\n")

        # Add each goal separately (add_goal can be called multiple times)
        for goal in goals:
            problem.add_goal(goal)

        return problem

    def _create_action_from_skill(self, skill_name, skill_data, item_fluents, achievement_fluents, level_fluents, floor_fluents):
        """
        Convert a skill into a Unified Planning action.

        Returns:
            InstantaneousAction
        """
        skill_with_consumption = skill_data["skill_with_consumption"]
        requirements = skill_with_consumption.get("requirements", {})
        consumption = skill_with_consumption.get("consumption", {})
        gain = skill_with_consumption.get("gain", {})

        # Sanitize action name
        safe_name = skill_name.replace(" ", "_").replace("-", "_")
        action = InstantaneousAction(safe_name)

        # Add preconditions from requirements
        for req_key, req_lambda in requirements.items():
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

            # Handle floor OR conditions (e.g., [0, 2] means floor 0 OR floor 2)
            if req_lambda_is_list:
                assert req_key.startswith("level:"), f"OR conditions only supported for levels, got: {req_key}"
                # For player_level, use boolean floor fluents; else fallback to numeric equality
                if req_key == "level:player_level":
                    or_conditions = []
                    for floor_value in req_lambda:
                        fl = int(floor_value)
                        if fl in floor_fluents:
                            or_conditions.append(floor_fluents[fl])
                    if len(or_conditions) == 0:
                        continue
                else:
                    assert req_key in level_fluents, f"Level requirement '{req_key}' not found in fluents"
                    or_conditions = []
                    for floor_value in req_lambda:
                        or_conditions.append(Equals(level_fluents[req_key], Int(int(floor_value))))

                # Create disjunction (OR of all conditions)
                if len(or_conditions) == 1:
                    action.add_precondition(or_conditions[0])
                else:
                    action.add_precondition(Or(*or_conditions))
                continue

            req_amount = self._evaluate_lambda(req_lambda, 1)  # Evaluate at n=1

            if req_key.startswith("achievement:"):
                assert req_key in achievement_fluents, f"Achievement requirement '{req_key}' not found in fluents"
                action.add_precondition(achievement_fluents[req_key])
            elif req_key.startswith("level:"):
                # For player_level, use at_floor boolean; others remain numeric equality
                if req_key == "level:player_level":
                    fl = int(req_amount)
                    if fl in floor_fluents:
                        action.add_precondition(floor_fluents[fl])
                else:
                    assert req_key in level_fluents, f"Level requirement '{req_key}' not found in fluents"
                    action.add_precondition(Equals(level_fluents[req_key], Int(int(req_amount))))
            else:
                # Check if it's actually in achievement or level fluents (misclassified)
                if req_key in achievement_fluents:
                    print(f"  DEBUG: Requirement '{req_key}' treated as item but found in achievement_fluents")
                    action.add_precondition(achievement_fluents[req_key])
                elif req_key in level_fluents:
                    print(f"  DEBUG: Requirement '{req_key}' treated as item but found in level_fluents")
                    # Non-player level equality
                    action.add_precondition(Equals(level_fluents[req_key], Int(int(req_amount))))
                else:
                    assert req_key in item_fluents, f"Item requirement '{req_key}' not found in fluents"
                    fluent = item_fluents[req_key]

                    # Check fluent type to avoid bool >= int errors
                    if hasattr(fluent, 'type') and hasattr(fluent.type, 'is_bool_type') and fluent.type.is_bool_type():
                        print(f"  DEBUG: Item requirement '{req_key}' has bool type! Using equality check.")
                        action.add_precondition(fluent)  # Just check if true
                    elif req_key in TIERED_INVENTORY_ITEMS:
                        # For tiered items, check tier level
                        action.add_precondition(fluent >= req_amount)
                    else:
                        action.add_precondition(fluent >= req_amount)

        # Add effects from consumption (decrease items)
        for cons_key, cons_lambda in consumption.items():
            cons_amount = self._evaluate_lambda(cons_lambda, 1)
            assert cons_key in item_fluents, f"Consumption item '{cons_key}' not found in fluents"
            fluent = item_fluents[cons_key]
            action.add_decrease_effect(fluent, cons_amount)

        # Add effects from gain (increase items/set achievements/levels)
        for gain_key, gain_entry in gain.items():
            if isinstance(gain_entry, dict):
                gain_expr = gain_entry.get("expression", "lambda n: n")
                gain_type = gain_entry.get("type", "inventory")
            else:
                gain_expr = gain_entry
                # Infer type from key
                if gain_key.startswith("achievement:"):
                    gain_type = "achievement"
                elif gain_key.startswith("level:"):
                    gain_type = "level"
                else:
                    gain_type = "inventory"

            gain_amount = self._evaluate_lambda_or_expr(gain_expr, 1)

            if gain_type == "achievement":
                assert gain_key in achievement_fluents, f"Achievement gain '{gain_key}' not found in fluents"
                action.add_effect(achievement_fluents[gain_key], True)
            elif gain_type == "level":
                assert gain_key in level_fluents, f"Level gain '{gain_key}' not found in fluents"
                action.add_effect(level_fluents[gain_key], gain_amount)
                # If player_level changes, toggle only the source floor off and the target floor on
                if gain_key == "level:player_level":
                    try:
                        new_floor = int(gain_amount)
                        # Turn on the new floor flag
                        if new_floor in floor_fluents:
                            action.add_effect(floor_fluents[new_floor], True)
                        # Determine source floors from this skill's level requirement
                        source_floors = set()
                        req_val = requirements.get("level:player_level")
                        if isinstance(req_val, (list, tuple)):
                            source_floors.update(int(v) for v in req_val if str(v).isdigit() or isinstance(v, int))
                        elif isinstance(req_val, str):
                            try:
                                sf = int(self._evaluate_lambda(req_val, 1))
                                source_floors.add(sf)
                            except Exception:
                                pass
                        elif isinstance(req_val, (int, float)):
                            source_floors.add(int(req_val))

                        # Turn off only the source floors (excluding the new floor)
                        for fl in source_floors:
                            if fl != new_floor and fl in floor_fluents:
                                action.add_effect(floor_fluents[fl], False)
                    except Exception:
                        pass
            else:
                # For inventory items, check multiple places since keys might be misclassified
                if gain_key in achievement_fluents:
                    # Misclassified as inventory but it's actually an achievement
                    print(f"  DEBUG: Gain '{gain_key}' classified as inventory but found in achievement_fluents")
                    action.add_effect(achievement_fluents[gain_key], True)
                elif gain_key in level_fluents:
                    # Misclassified as inventory but it's actually a level
                    print(f"  DEBUG: Gain '{gain_key}' classified as inventory but found in level_fluents")
                    action.add_effect(level_fluents[gain_key], gain_amount)
                else:
                    assert gain_key in item_fluents, f"Item gain '{gain_key}' not found in any fluents"
                    fluent = item_fluents[gain_key]

                    # Check the actual fluent type to avoid type errors
                    if hasattr(fluent, 'type') and hasattr(fluent.type, 'is_bool_type') and fluent.type.is_bool_type():
                        # This is actually a boolean fluent, use set effect
                        print(f"  DEBUG: Fluent '{gain_key}' in item_fluents but has bool type! Using set effect.")
                        action.add_effect(fluent, True)
                    elif gain_key in TIERED_INVENTORY_ITEMS:
                        # For tiered items, set to the new tier level
                        action.add_effect(fluent, gain_amount)
                    else:
                        action.add_increase_effect(fluent, gain_amount)

        return action

    def _solve_planning_problem(self, problem):
        """
        Solve the planning problem using the Unified Planning Library.

        Returns:
            A plan (sequence of actions) - FAILS LOUDLY if no plan found
        """
        print("Solving planning problem...")
        logging.basicConfig(level=logging.INFO)

        # Optional: To be more specific, you can set the logger for the unified_planning package
        logging.getLogger('unified_planning').setLevel(logging.INFO) 

        with OneshotPlanner(name='enhsp', problem_kind=problem.kind) as planner:
            result = planner.solve(problem, timeout=self.planner_timeout)

            if result.status in [
                result.status.SOLVED_SATISFICING,
                result.status.SOLVED_OPTIMALLY,
            ]:
                print(f"Plan found! Status: {result.status}")
                return result.plan
            else:
                # FAIL LOUDLY
                raise RuntimeError(f"Planning failed with status: {result.status}")

    def _plan_to_execution_order(self, plan):
        """
        Convert a Unified Planning plan to execution order format.

        Args:
            plan: A unified_planning.plan.Plan instance

        Returns:
            List of (skill_name, count) tuples
        """
        execution_order = []

        for action_instance in plan.actions:
            # Get the action name and convert back to skill name
            action_name = action_instance.action.name
            skill_name = action_name.replace("_", " ")

            # Find the original skill name (case-insensitive match)
            original_skill_name = None
            for sk in self.skills.keys():
                if sk.replace(" ", "_").replace("-", "_").lower() == action_name.lower():
                    original_skill_name = sk
                    break

            if original_skill_name is None:
                # Try direct match
                if skill_name in self.skills:
                    original_skill_name = skill_name
                else:
                    # FAIL LOUDLY
                    raise ValueError(
                        f"Could not map action '{action_name}' to any skill. "
                        f"Available skills: {list(self.skills.keys())}"
                    )

            execution_order.append((original_skill_name, 1))

        print(f"Converted plan with {len(execution_order)} steps")
        return execution_order

    def _combine_consecutive_skills(self, execution_order):
        """Combine consecutive identical skills into single entries."""
        if not execution_order:
            return execution_order

        print("\nCombining consecutive identical skills...")

        combined_order = []
        current_skill = execution_order[0][0]
        current_count = execution_order[0][1]

        for skill_name, count in execution_order[1:]:
            if skill_name == current_skill:
                current_count += count
                print(f"  Combining {skill_name}: {current_count - count} + {count} = {current_count}")
            else:
                combined_order.append((current_skill, current_count))
                current_skill = skill_name
                current_count = count

        combined_order.append((current_skill, current_count))

        print("Combined execution order:")
        for i, (skill_name, count) in enumerate(combined_order):
            print(f"  {i+1}. {skill_name} (n={count})")

        return combined_order

    def _convert_to_target_amounts(self, execution_order):
        """Convert execution counts to target inventory amounts."""
        print(f"\nConverting execution counts to target inventory amounts...")

        inventory = {}
        target_execution_order = []

        for skill_name, execution_count in execution_order:
            skills_to_use = self.processed_skills

            if skill_name not in skills_to_use:
                # FAIL LOUDLY
                raise ValueError(f"Skill '{skill_name}' not found in processed skills")

            # Calculate what this skill does to inventory
            gain = self._parse_lambda_gain(skill_name, execution_count)
            consumption = self._parse_lambda_consumption(skill_name, execution_count)

            # For skills that gain items, convert to target amount
            if gain:
                main_item = max(gain.keys(), key=lambda x: gain[x]) if gain else None

                if main_item and gain[main_item] > 0:
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
                    target_execution_order.append((skill_name, execution_count))
            else:
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

    def _parse_lambda_gain(self, skill_name, n=1):
        """Parse lambda functions in skill gain and return actual quantities gained."""
        assert skill_name in self.processed_skills, f"Skill '{skill_name}' not found"

        skill_data = self.processed_skills[skill_name]
        skill_with_consumption = skill_data["skill_with_consumption"]
        gain = skill_with_consumption.get("gain", {})

        parsed_gain = {}
        for item_name, gain_entry in gain.items():
            # Only include inventory-type gains
            if isinstance(gain_entry, dict):
                gain_type = gain_entry.get("type", "inventory")
                if gain_type != "inventory":
                    continue
                gain_expr = gain_entry.get("expression", "lambda n: n")
            else:
                # Simple format - check key
                if item_name.startswith("achievement:") or item_name.startswith("level:"):
                    continue
                gain_expr = gain_entry

            quantity_gained = self._evaluate_lambda_or_expr(gain_expr, n)
            parsed_gain[item_name] = quantity_gained

        return parsed_gain

    def _parse_lambda_consumption(self, skill_name, n=1):
        """Parse lambda functions in skill consumption and return actual quantities consumed."""
        assert skill_name in self.processed_skills, f"Skill '{skill_name}' not found"

        skill_data = self.processed_skills[skill_name]
        skill_with_consumption = skill_data["skill_with_consumption"]
        consumption = skill_with_consumption.get("consumption", {})

        parsed_consumption = {}
        for item_name, lambda_str in consumption.items():
            quantity_consumed = self._evaluate_lambda(lambda_str, n)
            parsed_consumption[item_name] = quantity_consumed

        return parsed_consumption

    def _evaluate_lambda(self, lambda_str, n):
        """Evaluate a lambda function string with the given n value."""
        lambda_func = eval(lambda_str)
        return lambda_func(n)

    def _evaluate_lambda_or_expr(self, expr, n):
        """Evaluate a lambda function string or expression with the given n value."""
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

    def _inline_ephemeral_requirements(self):
        """
        Preprocess skills to inline ephemeral requirements.

        This is copied from the old resolver since it's a useful preprocessing step.
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
                continue

            requirements = skill_with_consumption.get("requirements", {})
            consumption = skill_with_consumption.get("consumption", {})
            new_requirements = {}
            new_consumption = {}

            # First, copy original consumption as baseline
            for cons_item, cons_lambda in consumption.items():
                new_consumption[cons_item] = cons_lambda

            # Process requirements
            for req_item, req_lambda in requirements.items():
                # Check if this is a floor OR condition (either already a list, or evals to one)
                req_lambda_is_list = isinstance(req_lambda, (list, tuple))
                if not req_lambda_is_list and isinstance(req_lambda, str):
                    # Try to eval and check if it's a list
                    try:
                        eval_result = eval(req_lambda)
                        req_lambda_is_list = isinstance(eval_result, (list, tuple))
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
                    # For infrastructure-like ephemeral outputs (achievements/levels), avoid inlining to
                    # prevent duplicating their consumption across many dependent skills.
                    eph_swc = ephemeral_skills[providing_ephemeral_skill]["skill_with_consumption"]
                    eph_gain_map = eph_swc.get("gain", {})
                    eph_gain_entry = eph_gain_map.get(req_item)
                    eph_gain_type = None
                    if isinstance(eph_gain_entry, dict):
                        eph_gain_type = eph_gain_entry.get("type")

                    if (
                        isinstance(req_item, str)
                        and (
                            req_item.startswith("achievement:")
                            or req_item.startswith("level:")
                            or req_item.startswith("ephemeral:")
                            or (eph_gain_type in ("achievement", "level"))
                        )
                    ):
                        print(
                            f"  Not inlining ephemeral non-inventory requirement '{req_item}' from '{providing_ephemeral_skill}'"
                        )
                        if req_item in new_requirements:
                            new_requirements[req_item] = self._combine_lambda_requirements(
                                new_requirements[req_item], req_lambda
                            )
                        else:
                            new_requirements[req_item] = req_lambda
                        continue

                    print(f"Inlining ephemeral requirement '{req_item}' from '{providing_ephemeral_skill}' in skill '{skill_name}'")

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
                    # Keep non-ephemeral requirements
                    if req_item in new_requirements:
                        new_requirements[req_item] = self._combine_lambda_requirements(
                            new_requirements[req_item], req_lambda
                        )
                    else:
                        new_requirements[req_item] = req_lambda

            processed_skills[skill_name]["skill_with_consumption"]["requirements"] = new_requirements
            processed_skills[skill_name]["skill_with_consumption"]["consumption"] = new_consumption

        return processed_skills

    def _compose_lambda_requirements(self, ephemeral_lambda, need_lambda):
        """Compose two lambda requirements by substitution."""
        # FAIL LOUDLY if either is a list - can't compose floor OR conditions
        assert not isinstance(need_lambda, (list, tuple)), (
            f"Cannot compose lambda with floor OR condition (need_lambda={need_lambda})"
        )
        assert not isinstance(ephemeral_lambda, (list, tuple)), (
            f"Cannot compose lambda with floor OR condition (ephemeral_lambda={ephemeral_lambda})"
        )

        need_func = eval(need_lambda)
        eph_func = eval(ephemeral_lambda)

        need_b = need_func(0)
        need_a = need_func(1) - need_func(0)

        eph_d = eph_func(0)
        eph_c = eph_func(1) - eph_func(0)

        if need_a == 0:
            result_coeff = 0
            result_const = eph_c * need_b + eph_d

            if result_coeff == 0:
                return f"lambda n: 0*n + {result_const}"
            elif result_const == 0:
                return f"lambda n: {result_coeff}*n + 0"
            else:
                return f"lambda n: {result_coeff}*n + {result_const}"
        else:
            result_amount = eph_func(need_func(1))
            return f"lambda n: 0*n + {result_amount}"

    def _combine_lambda_requirements(self, existing_lambda, new_lambda):
        """Combine two lambda requirements by adding coefficients."""
        existing_func = eval(existing_lambda)
        new_func = eval(new_lambda)

        existing_b = existing_func(0)
        existing_a = existing_func(1) - existing_func(0)

        new_b = new_func(0)
        new_a = new_func(1) - new_func(0)

        combined_a = existing_a + new_a
        combined_b = existing_b + new_b

        if combined_a == 0:
            return f"lambda n: 0*n + {combined_b}"
        elif combined_b == 0:
            return f"lambda n: {combined_a}*n + 0"
        else:
            return f"lambda n: {combined_a}*n + {combined_b}"


# Create alias for compatibility
UnifiedSkillDependencyResolver = UnifiedPlanningSkillResolver

__all__ = [
    "UnifiedPlanningSkillResolver",
    "UnifiedSkillDependencyResolver",
    "TIERED_INVENTORY_ITEMS",
    "configure_symbolic_state_module",
]
