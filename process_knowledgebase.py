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
    llm_name = "gpt-4.1"
    # llm_name = "google-gemini-2.0-pro-exp-02-05"
    # llm_name = "google-gemini-2.0-flash-thinking-exp"


class SaveManualAfterQuery(aq.JsonAfterQuery):
    def __init__(self):
        super().__init__()
        self.type = dict
        # self.required_keys = [
        #     "manual",
        # ]
        # self.length = len(self.keys)

    def post_process(self):
        self.node.db["knowledgebase"] = self.parse_json()[-1]


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
        
Make a knowledge base for the game Crafter. The goal of this knowledge base is to serve as a source of verified information for an agent playing the game. For each action, enumerate the possible requirements for performing that action. Return your response in a json format

Make the hierarchy: action which will itself be a dictionary with the action name as the key, and then within that dictionary, list each instance of the action as a separate key, with the value being a list of requirements for that action in that context.


@struct.dataclass
class Inventory:
    wood: int = 0
    stone: int = 0
    coal: int = 0
    iron: int = 0
    diamond: int = 0
    sapling: int = 0
    wood_pickaxe: int = 0
    stone_pickaxe: int = 0
    iron_pickaxe: int = 0
    wood_sword: int = 0
    stone_sword: int = 0
    iron_sword: int = 0
#max inventory size is 9 for each item

# ENUMS
class BlockType(Enum):
    INVALID = 0
    OUT_OF_BOUNDS = 1
    GRASS = 2
    WATER = 3
    STONE = 4
    TREE = 5
    WOOD = 6
    PATH = 7
    COAL = 8
    IRON = 9
    DIAMOND = 10
    CRAFTING_TABLE = 11
    FURNACE = 12
    SAND = 13
    LAVA = 14
    PLANT = 15
    RIPE_PLANT = 16

class Action(Enum):
    NOOP = 0  #
    LEFT = 1  # a
    RIGHT = 2  # d
    UP = 3  # w
    DOWN = 4  # s
    DO = 5  # space
    SLEEP = 6  # tab
    PLACE_STONE = 7  # r
    PLACE_TABLE = 8  # t
    PLACE_FURNACE = 9  # f
    PLACE_PLANT = 10  # p
    MAKE_WOOD_PICKAXE = 11  # 1
    MAKE_STONE_PICKAXE = 12  # 2
    MAKE_IRON_PICKAXE = 13  # 3
    MAKE_WOOD_SWORD = 14  # 4
    MAKE_STONE_SWORD = 15  # 5
    MAKE_IRON_SWORD = 16  # 6

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

    with open("/home/renos/flow-rl/resources/craftax_classic_knowledgebase.json", "w") as f:
        json.dump(db["knowledgebase"], f, indent=2)

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
