import os
import sys

# Get the current file's directory
# current_dir = os.path.dirname(os.path.abspath(__file__))

# # Change to parent directory
# parent_dir = os.path.dirname(current_dir)
# os.chdir(parent_dir)

import flowrl
from flowrl.llm.compose_prompts import ComposeBasePrompt, ComposeReasoningPrompt
from flowrl.llm.craftax_classic.after_queries import *
import agentkit

from agentkit import Graph, SimpleDBNode

from functools import partial
import agentkit.llm_api

# import llm_api
from functools import partial

from flowrl.llm.compose_prompts import ComposeReasoningPrompt
from flowrl.llm.craftax_classic.after_queries import *

try:
    import llm_api

    get_query = llm_api.get_query
    llm_name = "yintat-gpt-4o"
except:
    get_query = agentkit.llm_api.get_query
    llm_name = "gpt-5"
    # llm_name = "google-gemini-2.0-pro-exp-02-05"
    # llm_name = "google-gemini-2.0-flash-thinking-exp"


class SaveManualAfterQuery(aq.BaseAfterQuery):
    def __init__(self):
        super().__init__()
        self.type = str
        # self.required_keys = [
        #     "manual",
        # ]
        # self.length = len(self.keys)

    def post_process(self):
        self.node.db["manual"] = self.node.result


def generate_graph(db=None, return_inventory_graph=False):

    LLM_API_FUNCTION_GPT4 = partial(get_query(llm_name), max_gen=8192)

    def build_graph(prompts, database={}):

        # Create graph
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

    prompts = return_prompts(LLM_API_FUNCTION_GPT4)
    graph = build_graph(prompts, db)
    return graph


def return_prompts(LLM_API_FUNCTION_GPT4):
    inventory_prompts = {}

    inventory_prompts["predict_item_count"] = {
        "prompt": """
        
You are an expert writing a game manual for the game Craftax. The goal is for the player to have all information needed to play the game.
You are given the following information:

Craftax ICML Paper
```
$db.paper$
```

Craftax Game Logic
```
$db.game_logic$
```


## Review requirements
1. Explains all game rules, mechanics, and systems

## Invidiaul Achievement Decomposition
For every achievement or objective indiviaully:
   - Lists all prerequisites
   - Details each individual step required
   - Explains any specific techniques or approaches needed
   - Provides specific quantities, requirements, and conditions wherever relevant


The manual should be thorough enough that even inexperienced players can follow along without having to guess or experiment.

        """,
        "dep": [],
        "compose": ComposeBasePrompt(),
        "query": LLM_API_FUNCTION_GPT4,
        "after_query": SaveManualAfterQuery(),
    }

    for _, v in inventory_prompts.items():
        v["prompt"] = v["prompt"].strip()
    return inventory_prompts


def main(game_logic_path, paper_path):
    """Main function to analyze game files and create a manual."""

    # Setup LLM function
    db = {}

    # Load game logic and paper files
    with open(game_logic_path, "r") as f:
        game_logic = f.read()
    with open(paper_path, "r") as f:
        paper = f.read()

    db["game_logic"] = game_logic
    db["paper"] = paper
    graph = generate_graph(db)

    answer = graph.evaluate()

    # # Save outputs
    with open("/home/renos/flow-rl/resources/craftax_manual.txt", "w") as f:
        f.write(db["manual"])

    # with open("agent_prompt_template.txt", "w") as f:
    #     f.write(agent_prompt)

    # return {
    #     "manual": manual,
    #     "agent_prompt": agent_prompt
    # }


if __name__ == "__main__":
    # Paths to the input files
    game_logic_path = "/home/renos/flow-rl/resources/craftax_game_logic.py"
    paper_path = "/home/renos/flow-rl/resources/craftax_paper.tex"

    # Call the main function
    main(game_logic_path, paper_path)
