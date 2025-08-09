"""
Skill Dependency Resolver

This module handles the complex logic of resolving skill dependencies by transforming
requirements at each level until all dependencies can be satisfied by basic skills.
"""


class SkillDependencyResolver:
    def __init__(self, skills, max_inventory_capacity=9):
        """
        Initialize the resolver with a dictionary of skills.
        
        Args:
            skills: Dict mapping skill_name -> skill_data
            max_inventory_capacity: Maximum number of items that can be held of each type
        """
        self.skills = skills
        self.max_inventory_capacity = max_inventory_capacity
    
    def resolve_dependencies(self, target_skill_name, n=1):
        """
        Build a dependency graph by transforming requirements at each level.
        Each level transforms requirements by replacing them with the requirements
        of skills that provide those items.
        
        Args:
            target_skill_name: Name of the skill to build dependencies for
            n: Number of times to apply this skill
            
        Returns:
            List of skills in execution order (skills may appear multiple times)
        """
        if target_skill_name not in self.skills:
            raise ValueError(f"Skill '{target_skill_name}' not found in skills")
        
        # Preprocess skills to inline ephemeral requirements
        processed_skills = self._inline_ephemeral_requirements()
        
        # Start with the target skill's requirements (after preprocessing)
        target_skill = processed_skills[target_skill_name]
        skill_with_consumption = target_skill["skill_with_consumption"]
        current_requirements = skill_with_consumption.get("requirements", {})
        
        print(f"Starting dependency resolution for skill '{target_skill_name}'")
        print(f"Initial requirements: {current_requirements}")
        
        # Keep transforming requirements until we reach basic skills
        # Also collect skills needed at each level
        level = 0
        skills_by_level = {}  # level -> [(skill_name, required_amount)]
        
        while current_requirements:
            level += 1
            print(f"\n--- Level {level} ---")
            print(f"Current requirements: {current_requirements}")
            
            skills_by_level[level] = []
            new_requirements = {}
            
            # For each current requirement, transform it
            for req_item, req_formula in current_requirements.items():
                providing_skill = self._find_skill_providing_item(req_item, processed_skills)
                
                if providing_skill is None:
                    print(f"Warning: No skill found that provides '{req_item}' - treating as basic requirement")
                    # Keep this as a basic requirement (can't be transformed further)
                    new_requirements[req_item] = req_formula
                    continue
                
                print(f"Requirement '{req_item}' provided by skill '{providing_skill}'")
                
                # Calculate how much of this skill we need
                required_amount = self._evaluate_lambda(req_formula, n)
                skills_by_level[level].append((providing_skill, required_amount, req_item))
                
                # Get the providing skill's requirements (use processed skills)
                providing_skill_data = processed_skills[providing_skill]
                providing_skill_requirements = providing_skill_data["skill_with_consumption"].get("requirements", {})
                
                if not providing_skill_requirements:
                    print(f"Skill '{providing_skill}' has no requirements - it's a basic skill")
                    # This is a basic skill, no further transformation needed
                    continue
                
                # Transform: replace this requirement with the providing skill's requirements
                for sub_req_item, sub_req_formula in providing_skill_requirements.items():
                    # Check if this requirement evaluates to zero - if so, skip it
                    sub_quantity_needed = self._evaluate_lambda(sub_req_formula, n)
                    if sub_quantity_needed <= 0:
                        print(f"Sub-requirement '{sub_req_item}' evaluates to {sub_quantity_needed}, skipping")
                        continue
                        
                    if sub_req_item in new_requirements:
                        print(f"Accumulating requirement for '{sub_req_item}'")
                        # Properly combine lambda functions when accumulating
                        combined_lambda = self._combine_lambda_requirements(
                            new_requirements[sub_req_item], sub_req_formula
                        )
                        new_requirements[sub_req_item] = combined_lambda
                        print(f"Combined requirement: '{sub_req_item}' = {combined_lambda}")
                    else:
                        new_requirements[sub_req_item] = sub_req_formula
                        print(f"Added new requirement: '{sub_req_item}' = {sub_req_formula}")
            
            # Update current requirements for next iteration
            current_requirements = new_requirements
            
            # Safety check to prevent infinite loops
            if level > 10:
                print("Warning: Dependency resolution depth exceeded 10 levels, stopping")
                break
        
        print(f"\nDependency resolution completed after {level} levels")
        print(f"Skills by level: {skills_by_level}")
        
        # Now build the execution order from deepest level to level 1
        execution_order = []
        planned_inventory = {}  # Track what will be available after planned executions
        
        # Process levels from deepest to shallowest (level 3 -> 2 -> 1)
        for current_level in sorted(skills_by_level.keys(), reverse=True):
            print(f"\nProcessing level {current_level}")
            for skill_name, required_amount, req_item in skills_by_level[current_level]:
                # Calculate how many times we need to run this skill
                if skill_name in self.skills:
                    skill_data = self.skills[skill_name]
                    gain = skill_data["skill_with_consumption"].get("gain", {})
                    if req_item in gain:
                        gain_per_execution = self._evaluate_lambda_or_expr(gain[req_item], 1)
                        times_needed = max(1, int(required_amount / gain_per_execution))
                        print(f"Adding {skill_name} (n={times_needed}) to execution order")
                        execution_order.append((skill_name, times_needed))
                        
                        # Update planned inventory
                        actual_gain = self._parse_lambda_gain(skill_name, times_needed)
                        actual_consumption = self._parse_lambda_consumption(skill_name, times_needed, processed_skills)
                        
                        for item, amount in actual_gain.items():
                            planned_inventory[item] = planned_inventory.get(item, 0) + amount
                            print(f"Updated planned inventory (gain): {item} = {planned_inventory[item]}")
                        
                        for item, amount in actual_consumption.items():
                            planned_inventory[item] = planned_inventory.get(item, 0) - amount
                            print(f"Updated planned inventory (consumption): {item} = {planned_inventory[item]}")
        
        # Store processed_skills for use in inventory constraints
        self._processed_skills = processed_skills
        
        # Add the target skill itself at the end
        execution_order.append((target_skill_name, n))
        
        # Filter out ephemeral skills (except the target skill) BEFORE inventory processing
        execution_order = self._filter_ephemeral_skills(execution_order, target_skill_name)

        
        
        # Apply inventory capacity constraints and reorder if necessary
        execution_order = self._apply_inventory_constraints(execution_order)
        
        # Skip combining consecutive skills since inventory constraints already optimized the order
        
        # Convert execution counts to target inventory amounts
        execution_order = self._convert_to_target_amounts(execution_order)
        
        return execution_order
    
    def _build_execution_order_recursive(self, target_skill_name, n, execution_order, visited, processed_skills=None, planned_inventory=None, depth=0):
        """
        Build the execution order by recursively resolving each requirement.
        Uses depth tracking to prevent infinite recursion while allowing skill reuse.
        
        Args:
            target_skill_name: Name of the skill to build execution order for
            n: Number of times to apply this skill
            execution_order: List to append execution steps to
            visited: Set of skills currently being processed (for cycle detection)
            processed_skills: Skills with ephemeral requirements inlined (optional, uses self.skills if None)
            planned_inventory: Dict tracking what will be available after planned executions
            depth: Current recursion depth
        """
        if depth > 20:
            print(f"Warning: Maximum recursion depth reached for {target_skill_name}")
            return
            
        # Use processed skills if provided, otherwise use self.skills
        skills_to_use = processed_skills if processed_skills is not None else self.skills
        
        # Initialize planned inventory if not provided
        if planned_inventory is None:
            planned_inventory = {}
            
        skill_key = f"{target_skill_name}_{n}"
        if skill_key in visited:
            print(f"Cycle detected: {target_skill_name} (n={n}) is already being processed, skipping")
            return
        
        visited.add(skill_key)
        print(f"{'  ' * depth}Building execution order for '{target_skill_name}' (n={n})")
        
        # Get the target skill's requirements from processed skills
        target_skill = skills_to_use[target_skill_name]
        skill_with_consumption = target_skill["skill_with_consumption"]
        requirements = skill_with_consumption.get("requirements", {})
        
        # Collect all prerequisite skills that need to be built
        prerequisite_skills_to_build = []
        
        # For each requirement, determine what prerequisite skills we need
        for req_item, req_formula in requirements.items():
            # Parse the lambda to get actual quantity needed
            quantity_needed = self._evaluate_lambda(req_formula, n)
            print(f"{'  ' * depth}Need {quantity_needed} of '{req_item}' for {n}x '{target_skill_name}'")
            
            # Check if we already have enough in planned inventory
            available_quantity = planned_inventory.get(req_item, 0)
            if available_quantity >= quantity_needed:
                print(f"{'  ' * depth}Already have {available_quantity} '{req_item}' planned, need {quantity_needed} - sufficient!")
                
                # Only reserve (subtract) if this item is actually consumed by the current skill
                current_skill_consumption = self._parse_lambda_consumption(target_skill_name, n, processed_skills)
                consumed_amount = current_skill_consumption.get(req_item, 0)
                if consumed_amount > 0:
                    # This item gets consumed, so reserve it based on actual consumption
                    planned_inventory[req_item] = available_quantity - consumed_amount
                    print(f"{'  ' * depth}Reserved {consumed_amount} '{req_item}' (consumable), remaining: {planned_inventory[req_item]}")
                else:
                    # This item is a tool/infrastructure, don't reserve it
                    print(f"{'  ' * depth}Tool/infrastructure '{req_item}' remains available for other skills")
                continue
                
            quantity_to_produce = quantity_needed - available_quantity
            print(f"{'  ' * depth}Have {available_quantity} '{req_item}' planned, need to produce {quantity_to_produce} more")
            
            # Find the skill that provides this item
            providing_skill = self._find_skill_providing_item(req_item, skills_to_use)
            
            if providing_skill is None:
                print(f"{'  ' * depth}Warning: No skill provides '{req_item}' - assuming it's available")
                continue
            
            # Calculate how many times we need to run the providing skill
            providing_skill_data = self.skills[providing_skill]
            providing_gain = providing_skill_data["skill_with_consumption"].get("gain", {})
            
            if req_item in providing_gain:
                gain_per_execution = self._evaluate_lambda_or_expr(providing_gain[req_item], 1)
                times_needed = max(1, int(quantity_to_produce / gain_per_execution))
                print(f"{'  ' * depth}Need to run '{providing_skill}' {times_needed} times to get {quantity_to_produce} '{req_item}'")
                
                # Add to prerequisites list instead of immediately processing
                prerequisite_skills_to_build.append((providing_skill, times_needed, req_item))
        
        # Now build all prerequisites in dependency order (deepest first)  
        for providing_skill, times_needed, req_item in prerequisite_skills_to_build:
            # Recursively resolve prerequisites for the providing skill FIRST
            self._build_execution_order_recursive(providing_skill, times_needed, execution_order, visited.copy(), processed_skills, planned_inventory, depth + 1)
            
            # Add the providing skill itself AFTER its prerequisites are resolved
            execution_order.append((providing_skill, times_needed))
            
            # Update planned inventory with what this skill will produce and consume
            actual_gain = self._parse_lambda_gain(providing_skill, times_needed)
            actual_consumption = self._parse_lambda_consumption(providing_skill, times_needed, processed_skills)
            
            # Add gains to planned inventory
            for item, amount in actual_gain.items():
                planned_inventory[item] = planned_inventory.get(item, 0) + amount
                print(f"{'  ' * depth}Updated planned inventory (gain): {item} = {planned_inventory[item]}")
            
            # Subtract consumption from planned inventory
            for item, amount in actual_consumption.items():
                planned_inventory[item] = planned_inventory.get(item, 0) - amount
                print(f"{'  ' * depth}Updated planned inventory (consumption): {item} = {planned_inventory[item]}")
        
        visited.remove(skill_key)  # Remove from visited when done processing
    
    def _apply_inventory_constraints(self, execution_order):
        """
        Apply inventory capacity constraints by actually modifying the execution order.
        Split collection skills and insert deferred collection nodes where capacity allows.
        
        Args:
            execution_order: List of (skill_name, count) tuples
            
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
            gain = self._parse_lambda_gain(skill_name, count)
            consumption = self._parse_lambda_consumption(skill_name, count, getattr(self, '_processed_skills', None))
            
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
                if current_amount + amount > self.max_inventory_capacity:
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
                        inventory[item] = inventory.get(item, 0) + amount
            else:
                # Normal execution - no capacity issues
                modified_execution.append((skill_name, count))
                
                # Update inventory
                for item, amount in consumption.items():
                    inventory[item] = max(0, inventory.get(item, 0) - amount)
                for item, amount in gain.items():
                    inventory[item] = inventory.get(item, 0) + amount
                
                print(f"Executed {skill_name} (n={count}), inventory: {inventory}")
            
            # After each skill, try to place deferred collections
            for item, deferred_amount in list(deferred_collections.items()):
                if deferred_amount > 0:
                    current_amount = inventory.get(item, 0)
                    can_collect = min(deferred_amount, self.max_inventory_capacity - current_amount)
                    
                    if can_collect > 0:
                        # Find the collection skill for this item
                        collection_skill = self._find_skill_providing_item(item, getattr(self, '_processed_skills', self.skills))
                        if collection_skill:
                            modified_execution.append((collection_skill, can_collect))
                            inventory[item] = current_amount + can_collect
                            deferred_collections[item] -= can_collect
                            print(f"Placed deferred collection: {collection_skill} (n={can_collect}) for '{item}'")
                            
                            if deferred_collections[item] == 0:
                                del deferred_collections[item]
        
        # Handle any remaining deferred collections at the end
        for item, deferred_amount in deferred_collections.items():
            if deferred_amount > 0:
                collection_skill = self._find_skill_providing_item(item, getattr(self, '_processed_skills', self.skills))
                if collection_skill:
                    modified_execution.append((collection_skill, deferred_amount))
                    print(f"Added final deferred collection: {collection_skill} (n={deferred_amount}) for '{item}'")
        
        print(f"\nFinal execution order with inventory constraints:")
        for i, (skill_name, count) in enumerate(modified_execution):
            print(f"  {i+1}. {skill_name} (n={count})")
        
        return modified_execution
    
    
    def _convert_to_target_amounts(self, execution_order):
        """
        Convert execution counts to target inventory amounts.
        For collection skills, n should mean target inventory level, not execution count.
        
        Args:
            execution_order: List of (skill_name, count) tuples where count is execution count
            
        Returns:
            List of (skill_name, target_amount) tuples where target_amount is desired inventory level
        """
        print(f"\nConverting execution counts to target inventory amounts...")
        
        # Track current inventory levels
        inventory = {}
        target_execution_order = []
        
        for skill_name, execution_count in execution_order:
            if skill_name not in self.skills:
                target_execution_order.append((skill_name, execution_count))
                continue
            
            # Calculate what this skill does to inventory
            gain = self._parse_lambda_gain(skill_name, execution_count)
            consumption = self._parse_lambda_consumption(skill_name, execution_count, getattr(self, '_processed_skills', None))
            
            # For skills that gain items, convert to target amount
            if gain:
                # Find the main item this skill produces
                main_item = max(gain.keys(), key=lambda x: gain[x]) if gain else None
                
                if main_item and gain[main_item] > 0:
                    # This is a collection/production skill
                    current_amount = inventory.get(main_item, 0)
                    target_amount = current_amount + gain[main_item]
                    
                    print(f"Skill '{skill_name}': execution_count={execution_count} -> target_amount={target_amount} for '{main_item}'")
                    print(f"  Current {main_item}: {current_amount}, will gain: {gain[main_item]}, target: {target_amount}")
                    
                    target_execution_order.append((skill_name, target_amount))
                else:
                    # Not a collection skill, keep execution count
                    target_execution_order.append((skill_name, execution_count))
            else:
                # No gain, keep execution count
                target_execution_order.append((skill_name, execution_count))
            
            # Update inventory after this skill
            for item, amount in gain.items():
                inventory[item] = inventory.get(item, 0) + amount
            for item, amount in consumption.items():
                inventory[item] = max(0, inventory.get(item, 0) - amount)
            
            print(f"  Inventory after '{skill_name}': {inventory}")
        
        print(f"\nFinal execution order with target amounts:")
        for i, (skill_name, target_amount) in enumerate(target_execution_order):
            print(f"  {i+1}. {skill_name} (target={target_amount})")
        
        return target_execution_order
    
    def _evaluate_lambda(self, lambda_str, n):
        """Evaluate a lambda function string with the given n value."""
        try:
            lambda_func = eval(lambda_str)
            return lambda_func(n)
        except Exception as e:
            print(f"Error evaluating lambda '{lambda_str}': {e}")
            return 0
    
    def _evaluate_lambda_or_expr(self, expr, n):
        """Evaluate a lambda function string or expression with the given n value."""
        try:
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
    
    def _find_skill_providing_item(self, item_name, skills_dict=None):
        """
        Find a skill that provides (gains) the specified item.
        
        Args:
            item_name: Name of the item to find a provider for
            skills_dict: Skills dictionary to search in (uses self.skills if None)
            
        Returns:
            Name of skill that provides the item, or None if not found
        """
        skills_to_search = skills_dict if skills_dict is not None else self.skills
        for skill_name, skill_data in skills_to_search.items():
            skill_with_consumption = skill_data["skill_with_consumption"]
            gain = skill_with_consumption.get("gain", {})
            
            if item_name in gain:
                return skill_name
        
        return None
    
    
    def _parse_lambda_consumption(self, skill_name, n=1, processed_skills=None):
        """
        Parse lambda functions in skill consumption and return actual quantities consumed.
        
        Args:
            skill_name: Name of the skill
            n: Number of times this skill will be executed (default: 1)
            processed_skills: Optional processed skills dict to use instead of self.skills
            
        Returns:
            Dict mapping item names to actual quantities consumed (scales with n)
        """
        skills_to_use = processed_skills if processed_skills is not None else self.skills
        
        if skill_name not in skills_to_use:
            return {}
        
        skill_data = skills_to_use[skill_name]
        skill_with_consumption = skill_data["skill_with_consumption"]
        consumption = skill_with_consumption.get("consumption", {})
        
        parsed_consumption = {}
        
        for item_name, lambda_str in consumption.items():
            try:
                # Evaluate the lambda function with n
                lambda_func = eval(lambda_str)
                quantity_consumed = lambda_func(n)
                parsed_consumption[item_name] = quantity_consumed
            except Exception as e:
                print(f"Error parsing consumption lambda '{lambda_str}' for item '{item_name}' in skill '{skill_name}': {e}")
                parsed_consumption[item_name] = 0
        
        return parsed_consumption
    
    def _parse_lambda_gain(self, skill_name, n=1):
        """
        Parse lambda functions or expressions in skill gain and return actual quantities gained.
        
        Args:
            skill_name: Name of the skill
            n: Number of times this skill will be executed (default: 1)
            
        Returns:
            Dict mapping item names to actual quantities gained
        """
        if skill_name not in self.skills:
            return {}
        
        skill_data = self.skills[skill_name]
        skill_with_consumption = skill_data["skill_with_consumption"]
        gain = skill_with_consumption.get("gain", {})
        
        parsed_gain = {}
        
        for item_name, gain_expr in gain.items():
            try:
                if isinstance(gain_expr, str):
                    if gain_expr == "n":
                        # Simple case: gain is just n
                        quantity_gained = n
                    elif gain_expr.startswith("lambda"):
                        # Lambda function case
                        lambda_func = eval(gain_expr)
                        quantity_gained = lambda_func(n)
                    else:
                        # Try to evaluate as expression
                        quantity_gained = eval(gain_expr.replace("n", str(n)))
                else:
                    # Numeric case
                    quantity_gained = gain_expr * n
                
                parsed_gain[item_name] = quantity_gained
            except Exception as e:
                print(f"Error parsing gain '{gain_expr}' for item '{item_name}' in skill '{skill_name}': {e}")
                parsed_gain[item_name] = 0
        
        return parsed_gain

    def _filter_ephemeral_skills(self, execution_order, target_skill_name):
        """
        Filter out ephemeral skills from the execution order, except for the target skill.
        
        Ephemeral skills affect the world state but don't produce inventory items, so they
        don't need to be executed unless they are the goal skill itself.
        
        Args:
            execution_order: List of (skill_name, count) tuples
            target_skill_name: Name of the target skill (should not be filtered)
            
        Returns:
            Filtered execution order with ephemeral skills removed
        """
        filtered_order = []
        
        for skill_name, count in execution_order:
            # Always keep the target skill, even if it's ephemeral
            if skill_name == target_skill_name:
                filtered_order.append((skill_name, count))
                continue
                
            # Check if skill is ephemeral
            if skill_name in self.skills:
                skill_data = self.skills[skill_name]
                skill_with_consumption = skill_data.get("skill_with_consumption", {})
                is_ephemeral = skill_with_consumption.get("ephemeral", False)
                
                if not is_ephemeral:
                    # Non-ephemeral skills are kept
                    filtered_order.append((skill_name, count))
                else:
                    # Ephemeral skills are filtered out (except target)
                    print(f"Filtering out ephemeral skill: {skill_name} (count={count})")
            else:
                # Unknown skills are kept (safer default)
                filtered_order.append((skill_name, count))
        
        print(f"\nFiltered execution order (ephemeral skills removed):")
        for i, (skill_name, count) in enumerate(filtered_order):
            print(f"  {i+1}. {skill_name} (count={count})")
        
        return filtered_order

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
                # Check if this requirement comes from an ephemeral skill
                providing_ephemeral_skill = None
                for ephemeral_name, ephemeral_data in ephemeral_skills.items():
                    ephemeral_gain = ephemeral_data["skill_with_consumption"].get("gain", {})
                    if req_item in ephemeral_gain:
                        providing_ephemeral_skill = ephemeral_name
                        break
                
                if providing_ephemeral_skill:
                    print(f"Inlining ephemeral requirement '{req_item}' from '{providing_ephemeral_skill}' in skill '{skill_name}'")
                    print(f"Need amount: {req_lambda}")
                    
                    # Get the ephemeral skill's requirements and consumption
                    ephemeral_requirements = ephemeral_skills[providing_ephemeral_skill]["skill_with_consumption"].get("requirements", {})
                    ephemeral_consumption = ephemeral_skills[providing_ephemeral_skill]["skill_with_consumption"].get("consumption", {})
                    
                    # Inline ephemeral requirements into our requirements using lambda composition
                    for eph_req_item, eph_req_lambda in ephemeral_requirements.items():
                        # Compose: ephemeral_lambda(need_lambda(n))
                        composed_lambda = self._compose_lambda_requirements(eph_req_lambda, req_lambda)
                        print(f"Composed requirement: {eph_req_item} = {composed_lambda}")
                        
                        if eph_req_item in new_requirements:
                            new_requirements[eph_req_item] = self._combine_lambda_requirements(
                                new_requirements[eph_req_item], composed_lambda
                            )
                        else:
                            new_requirements[eph_req_item] = composed_lambda
                    
                    # Inline ephemeral consumption into our consumption using lambda composition
                    for eph_cons_item, eph_cons_lambda in ephemeral_consumption.items():
                        # Compose: ephemeral_lambda(need_lambda(n))
                        composed_lambda = self._compose_lambda_requirements(eph_cons_lambda, req_lambda)
                        print(f"Composed consumption: {eph_cons_item} = {composed_lambda}")
                        
                        if eph_cons_item in new_consumption:
                            new_consumption[eph_cons_item] = self._combine_lambda_requirements(
                                new_consumption[eph_cons_item], composed_lambda
                            )
                        else:
                            new_consumption[eph_cons_item] = composed_lambda
                    
                    # Don't add the ephemeral requirement itself
                    print(f"Removed ephemeral requirement '{req_item}' and added its dependencies")
                else:
                    # Keep non-ephemeral requirements as-is
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

    def _add_constant_to_lambda(self, lambda_str, constant):
        """
        Add a constant to a lambda expression.
        Converts "lambda n: a*n + b" to "lambda n: a*n + (b + constant)"
        """
        try:
            func = eval(lambda_str)
            
            # Extract coefficients
            b = func(0)  # Constant term
            a = func(1) - func(0)  # Coefficient of n
            
            # Add constant to b term
            new_b = b + constant
            
            # Generate the new lambda
            if a == 0:
                return f"lambda n: 0*n + {new_b}"
            elif new_b == 0:
                return f"lambda n: {a}*n + 0"
            else:
                return f"lambda n: {a}*n + {new_b}"
                
        except Exception as e:
            print(f"Error adding constant {constant} to lambda '{lambda_str}': {e}")
            return lambda_str

    def _combine_lambda_requirements(self, existing_lambda, new_lambda):
        """
        Combine two lambda requirements when a skill has multiple sources for the same resource.
        
        This combines lambda expressions of the form "lambda n: a*n + b" by adding their coefficients.
        """
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