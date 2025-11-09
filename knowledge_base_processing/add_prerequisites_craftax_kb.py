"""
Phase 2: Add prerequisites to skills in the knowledge base.

This reads the output from process_craftax_kb_from_tutorial.py and adds
prerequisites to each skill by analyzing all available skills.

Usage:
    python knowledge_base_processing/add_prerequisites_craftax_kb.py
"""

import os
import sys
import json
import argparse
from functools import partial
import agentkit
from agentkit import Graph, SimpleDBNode
from flowrl.llm.compose_prompts import ComposeBasePrompt
from flowrl.llm.craftax.after_queries import *
import agentkit.llm_api

try:
    import llm_api
    get_query = llm_api.get_query
    llm_name = "yintat-gpt-4o"
except:
    get_query = agentkit.llm_api.get_query
    llm_name = "gpt-5"


class SaveSkillWithRequirementsAfterQuery(aq.JsonAfterQuery):
    """Parse JSON response and save complete skill structure."""
    def __init__(self, skill_index):
        super().__init__()
        self.type = dict
        self.skill_index = skill_index

    def post_process(self):
        print(f"\n{'='*80}")
        print(f"LLM RESPONSE for skill {self.skill_index}:")
        print(f"{'='*80}")
        print(self.node.result)
        print(f"{'='*80}\n")

        result = self.parse_json()[-1]
        if "updated_skills" not in self.node.db:
            self.node.db["updated_skills"] = {}
        self.node.db["updated_skills"][self.skill_index] = result


def generate_graph(db=None, skill_index=None):
    """Build agentkit graph for adding prerequisites."""
    LLM_API_FUNCTION_GPT4 = partial(get_query(llm_name), max_gen=8192)

    def build_graph(prompts, database={}):
        graph = Graph()
        edge_list = []
        order_list = []

        for _, node_info in prompts.items():
            key = node_prompt = node_info["prompt"]
            node = SimpleDBNode(
                key,
                node_prompt,
                graph,
                node_info["query"],
                node_info["compose"],
                database,
                after_query=(
                    node_info["after_query"]
                    if "after_query" in node_info.keys()
                    else None
                ),
                verbose=True,
            )
            graph.add_node(node)

            for dependency in node_info["dep"]:
                dependency_name = prompts[dependency]["prompt"]
                edge_list.append((dependency_name, key))

            if "after" in node_info.keys():
                for dependency in node_info["after"]:
                    dependency_name = prompts[dependency]["prompt"]
                    order_list.append((dependency_name, key))

        for edge in edge_list:
            graph.add_edge(*edge)
        for order in order_list:
            graph.add_order(*order)

        return graph

    prompts = return_prompts(LLM_API_FUNCTION_GPT4, db, skill_index=skill_index)
    graph = build_graph(prompts, db)
    return graph


def return_prompts(LLM_API_FUNCTION_GPT4, db, skill_index=None):
    """Define prompts for adding prerequisites to each skill."""
    prereq_prompts = {}

    skills_list = db.get("skills", [])

    # If skill_index is specified, only process that one skill
    if skill_index is not None:
        indices_to_process = [skill_index]
    else:
        indices_to_process = range(len(skills_list))

    # Create a prompt for each skill
    for i in indices_to_process:
        skill = skills_list[i]
        skill_name = skill.get("skill_name", f"skill_{i}")

        prereq_prompts[f"add_prereq_{i}"] = {
            "prompt": f"""
You are analyzing prerequisites for a skill in Craftax.

# All Available Skills
```
$db.all_skills$
```

# Current Skill to Analyze
```json
{json.dumps(skill, indent=2)}
```

# Tutorial Context
```
$db.tutorial_text$
```

# Task


## Step 0: Understand the Skill
Read the skill name and description carefully.
- A Skill is a reinforcement learning policy executed over potentially a long horizon
- Skill requirements are the recommended prerequisites enabling sucessful execution of the skill, and are fulfilled by other skills prior executing this skill

## Step 1: Extract Relevant Tutorial Information
Read the tutorial context and extract ALL information relevant to achieving the skill:
- What does the tutorial say about how to achieve this skill?
- What resources does the tutorial say are consumed?

## Step 2: Determine Requirements and Consumption
Based on the tutorial information:

**Requirements**: Everything needed for safe and successful execution of this skill
- Include both necessary and recommended requirements
- Consider what requirements are intuitively useful to have even if not strictly necessary
- Include location requirements using "level:player_level" if floor_availability is not empty, including when floor availability is [0] (overworld). In this special case, instead of a lambda expression, instead return the list
- Let X be some requirement Y be the state after executing the skill. We should only include X if it is useful to go from the skills initial state to Y. Whether it is only useful after reaching Y is irrelevant.
- Do not factor in the usefulness of a requirement for future skills after reaching Y
- Do NOT include transitive dependencies


**Consumption**: Only what gets used up during execution
- Include any of the beforelisted requirements that are likely to be consumed during the execution of the skill
- For variable consumption, use a realistic estimate, leaning slightly higher for a safe margin
- Do NOT include durable equipment that remains after use

Requirements are a SUPERSET of consumption.

## Step 3: Determine Ephemeral Status
Is the primary gain a PLACEMENT or PROXIMITY effect in the game world (not observable in inventory)?
- Ephemeral = true: The skill places an object in the world or creates adjacency/proximity (the effect is in the world state, not inventory)
- Ephemeral = false: The skill changes inventory, achievements, levels, or stats (observable in the symbolic state)

Floor transitions change level:player_level (observable in symbolic state) so they are NOT ephemeral.

Explain your reasoning.

# Format

After completing your analysis above, return the JSON:

```json
{{
  "skill_name": "",           # Keep same
  "description": "",          # Keep same
  "requirements": {{}},       # gain_key -> "lambda n: a*n + b"
  "consumption": {{}},        # gain_key -> "lambda n: a*n + b"
  "gain": {{}},               # Keep same
  "ephemeral": false          # true if spatial effect
}}
```

Lambda format: `"lambda n: a*n + b"` where n=executions, a=per-execution, b=base
Each key must match a gain key from available skills.

Important:
- For tiered items (pickaxe, sword, bow), tier 0 means "not present" - don't add requirements with tier 0
""",
            "dep": [],
            "compose": ComposeBasePrompt(),
            "query": LLM_API_FUNCTION_GPT4,
            "after_query": SaveSkillWithRequirementsAfterQuery(i),
        }

    for _, v in prereq_prompts.items():
        v["prompt"] = v["prompt"].strip()

    return prereq_prompts


def main():
    """Main function to add prerequisites to skills."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Add prerequisites to Craftax knowledge base skills')
    parser.add_argument('--skill-index', type=int, default=None,
                        help='Process only the skill at this index')
    parser.add_argument('--save-single', action='store_true',
                        help='When used with --skill-index, save only that updated skill into the verified knowledge base')
    args = parser.parse_args()

    # Get project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Paths
    input_json_path = os.path.join(project_root, "resources", "craftax_knowledgebase.json")
    output_json_path = os.path.join(project_root, "resources", "craftax_knowledgebase_verified.json")
    output_txt_path = os.path.join(project_root, "resources", "craftax_knowledgebase_verified.txt")
    tutorial_path = os.path.join(project_root, "resources", "tutorial.md")

    print(f"Loading knowledge base from: {input_json_path}")

    if not os.path.exists(input_json_path):
        print(f"ERROR: Knowledge base file not found at {input_json_path}")
        print("Please run process_craftax_kb_from_tutorial.py first")
        sys.exit(1)

    # Load existing knowledge base
    with open(input_json_path, 'r') as f:
        kb = json.load(f)

    skills_list = kb.get("skills", [])
    tutorial_context = kb.get("tutorial_context", {})

    # Load the original tutorial text (same as in process_craftax_kb_from_tutorial.py)
    tutorial_text = ""
    if os.path.exists(tutorial_path):
        try:
            with open(tutorial_path, 'r') as f:
                tutorial_text = f.read()
            print(f"Loaded original tutorial text from: {tutorial_path} ({len(tutorial_text)} chars)")
        except Exception as e:
            print(f"Warning: Failed to read tutorial text at {tutorial_path}: {e}")
    else:
        print(f"Warning: Tutorial file not found at {tutorial_path}; proceeding without raw tutorial text")

    print(f"Loaded {len(skills_list)} skills")

    # Validate skill index if provided
    if args.skill_index is not None:
        if args.skill_index < 0 or args.skill_index >= len(skills_list):
            print(f"ERROR: Invalid skill index {args.skill_index}. Valid range: 0-{len(skills_list)-1}")
            sys.exit(1)
        print(f"Processing only skill {args.skill_index}: '{skills_list[args.skill_index]['skill_name']}'")

    # Setup database with all skills
    db = {
        "skills": skills_list,
        "all_skills": json.dumps(skills_list, indent=2),
        # Keep extracted tutorial context for quick reference
        "tutorial_context": json.dumps(tutorial_context, indent=2),
        # Also include the full original tutorial text to enable richer deduction
        "tutorial_text": tutorial_text,
        "updated_skills": {},
    }

    # Generate graph and evaluate
    if args.skill_index is not None:
        print(f"Adding prerequisites for skill {args.skill_index} using LLM...")
    else:
        print("Adding prerequisites to all skills using LLM...")
    graph = generate_graph(db, skill_index=args.skill_index)
    answer = graph.evaluate()

    # Update skills with complete structure (requirements, consumption, ephemeral)
    for i, updated_skill in db["updated_skills"].items():
        skills_list[i].update(updated_skill)
        print(f"Updated skill '{skills_list[i]['skill_name']}' with requirements/consumption")

    # If debugging single skill, print details and exit without saving
    if args.skill_index is not None and not args.save_single:
        print("\n" + "="*80)
        print(f"DEBUG OUTPUT for skill {args.skill_index}:")
        print("="*80)
        print(json.dumps(skills_list[args.skill_index], indent=2))
        print("\nNot saving (debug mode). Re-run with --save-single to update only this skill, or run without --skill-index to save all skills.")
        return

    # If saving a single skill, update only that entry in the verified KB
    if args.skill_index is not None and args.save_single:
        target_idx = args.skill_index
        updated_skill = skills_list[target_idx]
        target_name = updated_skill.get('skill_name', f'skill_{target_idx}')

        # Load existing verified KB if present, else start from the base kb we loaded
        verified_kb = None
        if os.path.exists(output_json_path):
            try:
                with open(output_json_path, 'r') as f:
                    verified_kb = json.load(f)
                print(f"Loaded existing verified KB from: {output_json_path}")
            except Exception as e:
                print(f"Warning: Could not load verified KB ({e}); starting from base KB")

        if not verified_kb:
            verified_kb = kb

        verified_skills = verified_kb.get('skills', [])

        # Find by name first to avoid index drift
        found_j = None
        for j, s in enumerate(verified_skills):
            if s.get('skill_name') == target_name:
                found_j = j
                break

        if found_j is not None:
            verified_skills[found_j].update(updated_skill)
            print(f"Updated verified KB skill by name match: index {found_j} -> '{target_name}'")
        elif 0 <= target_idx < len(verified_skills):
            verified_skills[target_idx].update(updated_skill)
            print(f"Updated verified KB skill by index: {target_idx} -> '{target_name}'")
        else:
            verified_skills.append(updated_skill)
            print(f"Appended skill to verified KB (no name/index match): '{target_name}'")

        verified_kb['skills'] = verified_skills

        # Save updated verified KB (JSON)
        print(f"Saving updated single-skill verified knowledge base to: {output_json_path}")
        with open(output_json_path, 'w') as f:
            json.dump(verified_kb, f, indent=2)

        # Also regenerate the pretty text file
        print(f"Saving human-readable version to: {output_txt_path}")
        with open(output_txt_path, 'w') as f:
            f.write("CRAFTAX KNOWLEDGE BASE WITH PREREQUISITES (Verified)\n")
            f.write("=" * 80 + "\n\n")

            def write_dict(d, indent=0):
                prefix = "  " * indent
                for key, value in d.items():
                    if isinstance(value, dict):
                        f.write(f"{prefix}{key}:\n")
                        write_dict(value, indent + 1)
                    elif isinstance(value, list):
                        f.write(f"{prefix}{key}:\n")
                        for item in value:
                            if isinstance(item, dict):
                                write_dict(item, indent + 1)
                                f.write("\n")
                            else:
                                f.write(f"{prefix}  - {item}\n")
                    else:
                        f.write(f"{prefix}{key}: {value}\n")

            write_dict(verified_kb)

        print("\nSingle-skill update complete!")
        print(f"Skill: {target_name}")
        print(f"JSON: {output_json_path}")
        print(f"Text: {output_txt_path}")
        return

    # Save updated knowledge base
    kb["skills"] = skills_list

    print(f"Saving updated knowledge base to: {output_json_path}")
    with open(output_json_path, 'w') as f:
        json.dump(kb, f, indent=2)

    # Save as pretty text
    print(f"Saving human-readable version to: {output_txt_path}")
    with open(output_txt_path, 'w') as f:
        f.write("CRAFTAX KNOWLEDGE BASE WITH PREREQUISITES (Verified)\n")
        f.write("=" * 80 + "\n\n")

        def write_dict(d, indent=0):
            """Recursively write dictionary as formatted text."""
            prefix = "  " * indent
            for key, value in d.items():
                if isinstance(value, dict):
                    f.write(f"{prefix}{key}:\n")
                    write_dict(value, indent + 1)
                elif isinstance(value, list):
                    f.write(f"{prefix}{key}:\n")
                    for item in value:
                        if isinstance(item, dict):
                            write_dict(item, indent + 1)
                            f.write("\n")
                        else:
                            f.write(f"{prefix}  - {item}\n")
                else:
                    f.write(f"{prefix}{key}: {value}\n")

        write_dict(kb)

    print("\nPrerequisite addition complete!")
    print(f"JSON: {output_json_path}")
    print(f"Text: {output_txt_path}")


if __name__ == "__main__":
    main()
