"""
Skill Dependency Resolver - Node-based Graph Implementation

This module handles skill dependencies by building a graph structure where each Node
represents a skill and its children represent the skills needed to fulfill its requirements.
The graph is then pruned to remove unnecessary executions of skills that produce 
non-consumable tools/infrastructure.
"""


TIERED_INVENTORY_ITEMS = {"pickaxe", "sword"}


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
        
    def add_child(self, child_node):
        """Add a child dependency node."""
        self.children.append(child_node)
        child_node.level = self.level + 1
        
    def __repr__(self):
        return f"Node({self.skill_name}, amount={self.amount_needed}, level={self.level}, children={len(self.children)})"


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
        
    def resolve_dependencies(self, target_skill_name, n=1, inline_ephemeral=False):
        """
        Build a dependency graph and prune it to remove unnecessary tool productions.

        Args:
            target_skill_name: Name of the skill to build dependencies for
            n: Number of times to apply this skill
            inline_ephemeral: Whether to inline ephemeral skills (for code gen) or keep them (for training order)

        Returns:
            List of skills in execution order
        """
        if target_skill_name not in self.skills:
            raise ValueError(f"Skill '{target_skill_name}' not found in skills")

        print(f"Building dependency graph for skill '{target_skill_name}' (n={n})")

        # Preprocess skills to inline ephemeral requirements (only if requested)
        if inline_ephemeral:
            processed_skills = self._inline_ephemeral_requirements()
        else:
            processed_skills = self.skills
        
        # Build the complete dependency graph using processed skills
        root_node = self._build_graph(target_skill_name, n, processed_skills=processed_skills)
        
        # Prune unnecessary tool/infrastructure productions from the graph
        self._prune_graph_nodes(root_node, processed_skills)
        
        # Convert pruned graph to execution order (level-order traversal for just-in-time resource collection)
        execution_order = self._graph_to_execution_order_levelwise(root_node)
        
        # Old approach (commented out for safety):
        # execution_order = self._graph_to_execution_order(root_node)
        
        # Apply inventory capacity constraints
        execution_order = self._apply_inventory_constraints(execution_order, processed_skills)
        
        # Combine consecutive identical skills
        execution_order = self._combine_consecutive_skills(execution_order)
        
        # Convert execution counts to target inventory amounts
        execution_order = self._convert_to_target_amounts(execution_order, processed_skills)
        
        return execution_order
    
    def _build_graph(self, skill_name, amount_needed, visited=None, level=0, processed_skills=None):
        """
        Recursively build the dependency graph.
        
        Args:
            skill_name: Name of the skill to build graph for
            amount_needed: How many times this skill is needed
            visited: Set to track cycles (for this branch only)
            level: Current depth level
            processed_skills: Skills with ephemeral requirements inlined
            
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
        
        # For each requirement, find the providing skill and build its subgraph
        for req_item, req_formula in requirements.items():
            # Calculate how much of this item we need
            quantity_needed = self._evaluate_lambda(req_formula, amount_needed)
            if quantity_needed <= 0:
                continue
                
            print(f"{'  ' * level}Need {quantity_needed} of '{req_item}'")
            
            # Find the skill that provides this item
            providing_skill = self._find_skill_providing_item(req_item, processed_skills)
            
            if providing_skill is None:
                print(f"{'  ' * level}Warning: No skill provides '{req_item}' - treating as basic requirement")
                continue
            
            # Calculate how many times we need to run the providing skill
            providing_skill_data = processed_skills[providing_skill]
            providing_gain = providing_skill_data["skill_with_consumption"].get("gain", {})
            
            if req_item in providing_gain:
                gain_per_execution = self._evaluate_lambda_or_expr(providing_gain[req_item], 1)
                times_needed = max(1, int(quantity_needed / gain_per_execution))
                
                print(f"{'  ' * level}Need to run '{providing_skill}' {times_needed} times")
                
                # Recursively build the subgraph for this providing skill
                child_node = self._build_graph(providing_skill, times_needed, visited.copy(), level + 1, processed_skills)
                node.add_child(child_node)
        
        visited.remove(cycle_key)
        return node
    
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
        
        # Process levels from deepest to shallowest (highest level number to 0)
        max_level = max(levels.keys()) if levels else 0
        for level in range(max_level, -1, -1):
            print(f"Processing level {level}:")
            level_nodes = levels[level]
            
            # Add all nodes at this level to execution order
            for node in level_nodes:
                execution_order.append((node.skill_name, node.amount_needed))
                print(f"  Added: {node.skill_name} (amount={node.amount_needed})")
        
        print("Raw execution order from graph (level-wise):")
        for i, (skill_name, amount) in enumerate(execution_order):
            print(f"  {i+1}. {skill_name} (amount={amount})")
        
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
        """
        inventory = {}
        redundant_nodes = []
        
        for node in execution_order_nodes:
            skill_name = node.skill_name
            amount = node.amount_needed

            if skill_name not in processed_skills:
                continue
            
            # Calculate what this skill would gain and consume
            gains = self._parse_lambda_gain(skill_name, amount, processed_skills)
            consumption = self._parse_lambda_consumption(skill_name, amount, processed_skills)
            
            # Check if this skill produces tools we already have enough of
            should_prune = False
            for item, amount_gained in gains.items():
                # Check if this item is ever consumed in the remaining execution order
                item_ever_consumed = self._is_item_consumed_later_in_nodes(execution_order_nodes, node, item, processed_skills)

                if item in TIERED_INVENTORY_ITEMS:
                    # For tiered items (pickaxe, sword), only prune if we already have this level or higher
                    current_level = inventory.get(item, 0)
                    target_level = amount_gained
                    if not item_ever_consumed and current_level >= target_level:
                        print(f"Node {skill_name} produces redundant {item}: have level {current_level}, target level {target_level}, never consumed")
                        should_prune = True
                        break
                else:
                    # For non-tiered items, prune if we already have some and it's never consumed
                    if not item_ever_consumed and inventory.get(item, 0) > 0:
                        print(f"Node {skill_name} produces redundant {item}: have {inventory.get(item, 0)}, never consumed")
                        should_prune = True
                        break
            
            if should_prune:
                redundant_nodes.append(node)
            else:
                # Update inventory
                for item, amount_consumed in consumption.items():
                    inventory[item] = inventory.get(item, 0) - amount_consumed
                for item, amount_gained in gains.items():
                    if item in TIERED_INVENTORY_ITEMS:
                        inventory[item] = max(inventory.get(item, 0), amount_gained)
                    else:
                        inventory[item] = inventory.get(item, 0) + amount_gained

                print(f"Keeping node {skill_name} (amount={amount}), inventory: {inventory}")

        return redundant_nodes
    
    def _is_item_consumed_later_in_nodes(self, execution_order_nodes, current_node, item, processed_skills):
        """Check if an item is consumed by any node that appears later in execution order."""
        found_current = False
        for node in execution_order_nodes:
            if node == current_node:
                found_current = True
                continue
            
            if found_current and node.skill_name in processed_skills:
                consumption = self._parse_lambda_consumption(node.skill_name, node.amount_needed, processed_skills)
                if item in consumption and consumption[item] > 0:
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
    
    
    def _parse_lambda_gain(self, skill_name, n=1, processed_skills=None):
        """Parse lambda functions in skill gain and return actual quantities gained."""
        skills_to_use = processed_skills if processed_skills is not None else self.skills
        
        if skill_name not in skills_to_use:
            return {}
        
        skill_data = skills_to_use[skill_name]
        skill_with_consumption = skill_data["skill_with_consumption"]
        gain = skill_with_consumption.get("gain", {})
        
        parsed_gain = {}
        for item_name, gain_expr in gain.items():
            try:
                quantity_gained = self._evaluate_lambda_or_expr(gain_expr, n)
                parsed_gain[item_name] = quantity_gained
            except Exception as e:
                print(f"Error parsing gain '{gain_expr}' for item '{item_name}' in skill '{skill_name}': {e}")
                parsed_gain[item_name] = 0

        return parsed_gain
    
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
        """Find a skill that provides (gains) the specified item."""
        skills_to_search = skills_dict if skills_dict is not None else self.skills

        for skill_name, skill_data in skills_to_search.items():
            skill_with_consumption = skill_data["skill_with_consumption"]
            gain = skill_with_consumption.get("gain", {})
            
            if item_name in gain:
                return skill_name

        return None

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
                        collection_skill = self._find_skill_providing_item(item, processed_skills)
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
                collection_skill = self._find_skill_providing_item(item, processed_skills)
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
