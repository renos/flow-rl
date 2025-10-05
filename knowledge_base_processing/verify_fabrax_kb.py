import os
import sys
import json
from functools import partial
import agentkit
from agentkit import Graph, SimpleDBNode
from flowrl.llm.compose_prompts import ComposeBasePrompt
from flowrl.llm.fabrax.after_queries import *
import agentkit.llm_api
# LLM setup (same as original script)
# LLM setup (same as original script)
try:
    import llm_api
    get_query = llm_api.get_query
    llm_name = "yintat-gpt-4o"
except:
    get_query = agentkit.llm_api.get_query
    llm_name = "gpt-5"


class SaveVerificationAfterQuery(aq.BaseAfterQuery):
    def __init__(self, action_name, subaction_name):
        super().__init__()
        self.type = str
        self.action_name = action_name
        self.subaction_name = subaction_name

    def post_process(self):
        if "verified_requirements" not in self.node.db:
            self.node.db["verified_requirements"] = {}
        if self.action_name not in self.node.db["verified_requirements"]:
            self.node.db["verified_requirements"][self.action_name] = {}
        
        self.node.db["verified_requirements"][self.action_name][self.subaction_name] = self.node.result


def generate_verification_graph(db=None):
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

            if "shorthand" in node_info.keys() and node_info["shorthand"] is not None:
                database["shorthands"][key] = node_info["shorthand"]

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

    prompts = return_verification_prompts(LLM_API_FUNCTION_GPT4, db)
    graph = build_graph(prompts, db)
    return graph


def return_verification_prompts(LLM_API_FUNCTION_GPT4, db):
    verification_prompts = {}

    # Load the knowledge base from the database
    kb_json = db["manual"]

    # Create a separate verification prompt for each action-subaction combination
    prompt_counter = 0
    for action_name, action_data in kb_json["action"].items():
        for subaction_name, requirements in action_data.items():
            prompt_key = f"verify_{action_name}_{subaction_name}_{prompt_counter}"
            
            verification_prompts[prompt_key] = {
                "prompt": f"""
You are verifying requirements for the action {action_name} with subaction {subaction_name} in the game Crafter.

You have access to the following information:
1. BlockType enum: INVALID=0, OUT_OF_BOUNDS=1, GRASS=2, WATER=3, STONE=4, TREE=5, WOOD=6, PATH=7, COAL=8, IRON=9, DIAMOND=10, CRAFTING_TABLE=11, FURNACE=12, SAND=13, LAVA=14, PLANT=15, RIPE_PLANT=16

2. Action enum: NOOP=0, LEFT=1, RIGHT=2, UP=3, DOWN=4, DO=5, SLEEP=6, PLACE_STONE=7, PLACE_TABLE=8, PLACE_FURNACE=9, PLACE_PLANT=10, MAKE_WOOD_PICKAXE=11, MAKE_STONE_PICKAXE=12, MAKE_IRON_PICKAXE=13, MAKE_WOOD_SWORD=14, MAKE_STONE_SWORD=15, MAKE_IRON_SWORD=16

The current requirements to verify are:
{requirements}

For each requirement, determine if it is:
- VERIFIED: Can be directly inferred from the enum information or is logically certain
- ASSUMPTION: An educated guess that cannot be confirmed from the available information

Return your response as a JSON list where each requirement is either:
- "VERIFIED: [requirement text]" (if you can confirm it from the available information)
- "ASSUMPTION: [requirement text]" (if it's an educated guess)

Be conservative - when in doubt, mark as assumption.
                """,
                "dep": [],
                "compose": ComposeBasePrompt(),
                "query": LLM_API_FUNCTION_GPT4,
                "after_query": SaveVerificationAfterQuery(action_name, subaction_name),
            }
            prompt_counter += 1

    for _, v in verification_prompts.items():
        v["prompt"] = v["prompt"].strip()
    return verification_prompts


def main():
    """Main function to verify each item in the knowledge base."""
    
    # Setup database
    db = {"shorthands": {}}

    # Load the knowledge base from file
    knowledge_base_path = os.path.join(os.path.dirname(__file__), "../resources/fabrax_knowledgebase.json")
    with open(knowledge_base_path, "r") as f:
        knowledge_base_json = json.load(f)
    
    db["manual"] = knowledge_base_json

    # Generate verification graph
    graph = generate_verification_graph(db)

    # Run verification
    answer = graph.evaluate()

    # Reconstruct the verified knowledge base
    verified_kb = {"action": {}}
    for action_name, action_data in db["verified_requirements"].items():
        verified_kb["action"][action_name] = {}
        for subaction_name, verified_requirements in action_data.items():
            # Parse the JSON response from the LLM
            try:
                parsed_requirements = json.loads(verified_requirements)
                verified_kb["action"][action_name][subaction_name] = parsed_requirements
            except json.JSONDecodeError:
                # If JSON parsing fails, keep the raw response
                verified_kb["action"][action_name][subaction_name] = verified_requirements

    # Save verified knowledge base
    output_path = os.path.join(os.path.dirname(__file__), "../resources/fabrax_knowledgebase_verified.json")
    with open(output_path, "w") as f:
        json.dump(verified_kb, f, indent=2)

    print(f"Verified knowledge base saved to: {output_path}")

    # Also save as pretty text
    text_output_path = os.path.join(os.path.dirname(__file__), "../resources/fabrax_knowledgebase_verified.txt")
    with open(text_output_path, "w") as f:
        f.write("VERIFIED FABRAX KNOWLEDGE BASE\n")
        f.write("=" * 50 + "\n\n")
        for action_name, action_data in verified_kb["action"].items():
            f.write(f"Action: {action_name}\n")
            f.write("-" * 30 + "\n")
            for subaction_name, requirements in action_data.items():
                f.write(f"  Subaction: {subaction_name}\n")
                if isinstance(requirements, list):
                    for req in requirements:
                        f.write(f"    - {req}\n")
                else:
                    f.write(f"    - {requirements}\n")
                f.write("\n")
            f.write("\n")

    print(f"Verified knowledge base (text format) saved to: {text_output_path}")

    return verified_kb


if __name__ == "__main__":
    # Call the main function
    main()