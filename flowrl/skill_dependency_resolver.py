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
        
        # Start with the target skill's requirements
        target_skill = self.skills[target_skill_name]
        skill_with_consumption = target_skill["skill_with_consumption"]
        current_requirements = skill_with_consumption.get("requirements", {})
        
        print(f"Starting dependency resolution for skill '{target_skill_name}'")
        print(f"Initial requirements: {current_requirements}")
        
        # Keep transforming requirements until we reach basic skills
        level = 0
        while current_requirements:
            level += 1
            print(f"\n--- Level {level} ---")
            print(f"Current requirements: {current_requirements}")
            
            new_requirements = {}
            
            # For each current requirement, transform it
            for req_item, req_formula in current_requirements.items():
                providing_skill = self._find_skill_providing_item(req_item)
                
                if providing_skill is None:
                    print(f"Warning: No skill found that provides '{req_item}' - treating as basic requirement")
                    # Keep this as a basic requirement (can't be transformed further)
                    new_requirements[req_item] = req_formula
                    continue
                
                print(f"Requirement '{req_item}' provided by skill '{providing_skill}'")
                
                # Get the providing skill's requirements
                providing_skill_data = self.skills[providing_skill]
                providing_skill_requirements = providing_skill_data["skill_with_consumption"].get("requirements", {})
                
                if not providing_skill_requirements:
                    print(f"Skill '{providing_skill}' has no requirements - it's a basic skill")
                    # This is a basic skill, no further transformation needed
                    continue
                
                # Transform: replace this requirement with the providing skill's requirements
                for sub_req_item, sub_req_formula in providing_skill_requirements.items():
                    if sub_req_item in new_requirements:
                        print(f"Accumulating requirement for '{sub_req_item}'")
                        # TODO: Properly combine lambda functions when accumulating
                        # For now, just keep the existing one
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
        
        # Now build the execution order by working backwards from the target skill
        execution_order = []
        self._build_execution_order_recursive(target_skill_name, n, execution_order, set())
        
        # Add the target skill itself at the end
        execution_order.append((target_skill_name, n))
        
        # Apply inventory capacity constraints and reorder if necessary
        execution_order = self._apply_inventory_constraints(execution_order)
        
        # Combine consecutive identical skills
        execution_order = self._combine_consecutive_skills(execution_order)
        
        return execution_order
    
    def _build_execution_order_recursive(self, target_skill_name, n, execution_order, visited, depth=0):
        """
        Build the execution order by recursively resolving each requirement.
        Uses depth tracking to prevent infinite recursion while allowing skill reuse.
        
        Args:
            target_skill_name: Name of the skill to build execution order for
            n: Number of times to apply this skill
            execution_order: List to append execution steps to
            visited: Set of skills currently being processed (for cycle detection)
            depth: Current recursion depth
        """
        if depth > 20:
            print(f"Warning: Maximum recursion depth reached for {target_skill_name}")
            return
            
        skill_key = f"{target_skill_name}_{n}"
        if skill_key in visited:
            print(f"Cycle detected: {target_skill_name} (n={n}) is already being processed, skipping")
            return
        
        visited.add(skill_key)
        print(f"{'  ' * depth}Building execution order for '{target_skill_name}' (n={n})")
        
        # Get the target skill's requirements
        target_skill = self.skills[target_skill_name]
        skill_with_consumption = target_skill["skill_with_consumption"]
        requirements = skill_with_consumption.get("requirements", {})
        
        # For each requirement, figure out how much we need and execute prerequisite skills
        for req_item, req_formula in requirements.items():
            # Parse the lambda to get actual quantity needed
            quantity_needed = self._evaluate_lambda(req_formula, n)
            print(f"{'  ' * depth}Need {quantity_needed} of '{req_item}' for {n}x '{target_skill_name}'")
            
            # Find the skill that provides this item
            providing_skill = self._find_skill_providing_item(req_item)
            
            if providing_skill is None:
                print(f"{'  ' * depth}Warning: No skill provides '{req_item}' - assuming it's available")
                continue
            
            # Calculate how many times we need to run the providing skill
            providing_skill_data = self.skills[providing_skill]
            providing_gain = providing_skill_data["skill_with_consumption"].get("gain", {})
            
            if req_item in providing_gain:
                gain_per_execution = self._evaluate_lambda_or_expr(providing_gain[req_item], 1)
                times_needed = max(1, int(quantity_needed / gain_per_execution))
                print(f"{'  ' * depth}Need to run '{providing_skill}' {times_needed} times to get {quantity_needed} '{req_item}'")
                
                # Recursively resolve prerequisites for the providing skill
                self._build_execution_order_recursive(providing_skill, times_needed, execution_order, visited.copy(), depth + 1)
                
                # Add the providing skill itself
                execution_order.append((providing_skill, times_needed))
        
        visited.remove(skill_key)  # Remove from visited when done processing
    
    def _apply_inventory_constraints(self, execution_order):
        """
        Apply inventory capacity constraints to the execution order.
        Reorder skills to ensure inventory never exceeds max capacity.
        
        Args:
            execution_order: List of (skill_name, count) tuples
            
        Returns:
            Reordered execution order that respects inventory constraints
        """
        print(f"\nApplying inventory constraints (max capacity: {self.max_inventory_capacity})")
        
        # Track current inventory levels
        inventory = {}
        reordered_execution = []
        pending_skills = execution_order.copy()
        
        while pending_skills:
            skill_executed = False
            
            for i, (skill_name, count) in enumerate(pending_skills):
                # Calculate what this skill would do to inventory
                gain = self._parse_lambda_gain(skill_name, count)
                consumption = self._parse_lambda_consumption(skill_name, count)
                
                # Calculate net change in inventory
                net_change = {}
                for item, amount in gain.items():
                    net_change[item] = net_change.get(item, 0) + amount
                for item, amount in consumption.items():
                    net_change[item] = net_change.get(item, 0) - amount
                
                # Check if this skill would exceed capacity
                would_exceed_capacity = False
                for item, change in net_change.items():
                    if change > 0:  # Only check items we're adding
                        current_amount = inventory.get(item, 0)
                        new_amount = current_amount + change
                        if new_amount > self.max_inventory_capacity:
                            print(f"Skill '{skill_name}' (n={count}) would exceed capacity for '{item}': {current_amount} + {change} = {new_amount} > {self.max_inventory_capacity}")
                            would_exceed_capacity = True
                            break
                
                if not would_exceed_capacity:
                    # Execute this skill
                    print(f"Executing: {skill_name} (n={count})")
                    
                    # Update inventory
                    for item, change in net_change.items():
                        inventory[item] = max(0, inventory.get(item, 0) + change)
                    
                    print(f"Inventory after {skill_name}: {inventory}")
                    
                    # Add to reordered execution and remove from pending
                    reordered_execution.append((skill_name, count))
                    pending_skills.pop(i)
                    skill_executed = True
                    break
            
            if not skill_executed:
                print("Warning: No skills can be executed without exceeding capacity!")
                print(f"Current inventory: {inventory}")
                print(f"Pending skills: {pending_skills}")
                # Add remaining skills anyway to avoid infinite loop
                reordered_execution.extend(pending_skills)
                break
        
        print(f"\nFinal execution order with inventory constraints:")
        for i, (skill_name, count) in enumerate(reordered_execution):
            print(f"  {i+1}. {skill_name} (n={count})")
        
        return reordered_execution
    
    def _parse_lambda_consumption(self, skill_name, n=1):
        """
        Parse lambda functions in skill consumption and return actual quantities consumed.
        
        Args:
            skill_name: Name of the skill
            n: Number of times this skill will be executed (default: 1)
            
        Returns:
            Dict mapping item names to actual quantities consumed
        """
        if skill_name not in self.skills:
            return {}
        
        skill_data = self.skills[skill_name]
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
    
    def _combine_consecutive_skills(self, execution_order):
        """
        Combine consecutive identical skills into single executions with higher counts.
        
        Args:
            execution_order: List of (skill_name, count) tuples
            
        Returns:
            Combined execution order with consecutive identical skills merged
        """
        if not execution_order:
            return execution_order
        
        print(f"\nCombining consecutive identical skills...")
        
        combined_order = []
        current_skill, current_count = execution_order[0]
        
        for i in range(1, len(execution_order)):
            skill_name, count = execution_order[i]
            
            if skill_name == current_skill:
                # Same skill, combine the counts
                print(f"Combining {current_skill} (n={current_count}) + {skill_name} (n={count}) = {current_skill} (n={current_count + count})")
                current_count += count
            else:
                # Different skill, add the current one and start new
                combined_order.append((current_skill, current_count))
                current_skill, current_count = skill_name, count
        
        # Add the last skill
        combined_order.append((current_skill, current_count))
        
        print(f"\nFinal combined execution order:")
        for i, (skill_name, count) in enumerate(combined_order):
            print(f"  {i+1}. {skill_name} (n={count})")
        
        return combined_order
    
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
    
    def _find_skill_providing_item(self, item_name):
        """
        Find a skill that provides (gains) the specified item.
        
        Args:
            item_name: Name of the item to find a provider for
            
        Returns:
            Name of skill that provides the item, or None if not found
        """
        for skill_name, skill_data in self.skills.items():
            skill_with_consumption = skill_data["skill_with_consumption"]
            gain = skill_with_consumption.get("gain", {})
            
            if item_name in gain:
                return skill_name
        
        return None
    
    def _parse_lambda_requirements(self, skill_name, n=1):
        """
        Parse lambda functions in skill requirements and return actual quantities needed.
        
        Args:
            skill_name: Name of the skill
            n: Number of times this skill will be executed (default: 1)
            
        Returns:
            Dict mapping item names to actual quantities needed
        """
        if skill_name not in self.skills:
            return {}
        
        skill_data = self.skills[skill_name]
        skill_with_consumption = skill_data["skill_with_consumption"]
        requirements = skill_with_consumption.get("requirements", {})
        
        parsed_requirements = {}
        
        for item_name, lambda_str in requirements.items():
            try:
                # Evaluate the lambda function with n
                lambda_func = eval(lambda_str)
                quantity_needed = lambda_func(n)
                parsed_requirements[item_name] = quantity_needed
            except Exception as e:
                print(f"Error parsing lambda '{lambda_str}' for item '{item_name}' in skill '{skill_name}': {e}")
                parsed_requirements[item_name] = 0
        
        return parsed_requirements
    
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