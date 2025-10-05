import os
import sys

import flowrl
from flowrl.llm.compose_prompts import ComposeBasePrompt, ComposeReasoningPrompt
from flowrl.llm.fabrax.after_queries import *
import agentkit
import json

from agentkit import Graph, SimpleDBNode

from functools import partial
import agentkit.llm_api

from flowrl.llm.compose_prompts import ComposeReasoningPrompt
from flowrl.llm.fabrax.after_queries import *

try:
    import llm_api

    get_query = llm_api.get_query
    llm_name = "yintat-gpt-4o"
except:
    get_query = agentkit.llm_api.get_query
    llm_name = "gpt-5"


class SaveManualAfterQuery(aq.JsonAfterQuery):
    def __init__(self):
        super().__init__()
        self.type = dict

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

Make a knowledge base for the game Fabrax. Fabrax is based on Craftax Classic but with extended crafting branches including metallurgy, glass & ceramics, and chemistry & alchemy.

The goal of this knowledge base is to serve as a source of verified information for an agent playing the game. For each action, enumerate the possible requirements for performing that action. Return your response in a json format.

Make the hierarchy: action which will itself be a dictionary with the action name as the key, and then within that dictionary, list each instance of the action as a separate key, with the value being a list of requirements for that action in that context.

IMPORTANT: Fabrax does NOT use tier-based tools. Instead of pickaxe levels (1=wood, 2=stone, 3=iron), each pickaxe type is a separate inventory item with its own name.

@struct.dataclass
class Inventory:
    # Basic resources
    wood: int = 0
    stone: int = 0
    coal: int = 0
    iron: int = 0
    diamond: int = 0
    sapling: int = 0

    # New Fabrax resources
    copper: int = 0
    tin: int = 0
    sand: int = 0
    clay: int = 0
    limestone: int = 0
    leather: int = 0

    # Pickaxes (separate items, NOT tier-based)
    wood_pickaxe: int = 0
    stone_pickaxe: int = 0
    iron_pickaxe: int = 0
    bronze_pickaxe: int = 0
    steel_pickaxe: int = 0

    # Swords (separate items, NOT tier-based)
    wood_sword: int = 0
    stone_sword: int = 0
    iron_sword: int = 0
    bronze_sword: int = 0
    steel_sword: int = 0

    # Intermediate materials
    iron_bar: int = 0
    steel_bar: int = 0
    bronze_bar: int = 0
    glass: int = 0
    bottle: int = 0
    lens: int = 0
    telescope: int = 0
    brick: int = 0
    lime: int = 0
    mortar: int = 0
    tar: int = 0
    fertilizer: int = 0
    flux: int = 0

# Max inventory size is 9 for each item

# BLOCK TYPES (extends Craftax Classic)
class BlockType(Enum):
    # Core blocks (Craftax Classic)
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

    # New Fabrax blocks
    COPPER = 17
    TIN = 18
    CLAY = 19
    LIMESTONE = 20
    ANVIL = 21
    KILN = 22
    COMPOSTER = 23
    ALCHEMY_BENCH = 24
    GLASS_WINDOW = 25
    MASONRY_WALL = 26

# ACHIEVEMENTS (54 total: 0-21 from Classic + 22-53 new Fabrax)
# Classic achievements (0-21): Same as Craftax Classic
# Metallurgy branch (22-33): anvil, bellows, iron_bar, copper, tin, steel_bar, bronze_bar, steel/bronze pickaxes & swords, harden_edge
# Glass & Ceramics (34-46): sand, kiln, glass, clay, brick, bottle, lens, limestone, lime, telescope, mortar, window, masonry
# Chemistry & Alchemy (47-53): composter, tar, fertilizer, alchemy_bench, tonics, flux

        """,
        "dep": [],
        "compose": ComposeBasePrompt(),
        "query": LLM_API_FUNCTION_GPT4,
        "after_query": SaveManualAfterQuery(),
    }

    for _, v in inventory_prompts.items():
        v["prompt"] = v["prompt"].strip()
    return inventory_prompts


def main(progression_guide_path):
    """Main function to analyze game files and create a manual."""

    # Setup LLM function
    db = {}

    # Load Fabrax progression guide
    with open(progression_guide_path, "r") as f:
        progression_guide = f.read()

    db["progression_guide"] = progression_guide
    graph = generate_graph(db)

    answer = graph.evaluate()

    output_path = os.path.join(os.path.dirname(__file__), "../resources/fabrax_knowledgebase.json")
    with open(output_path, "w") as f:
        json.dump(db["knowledgebase"], f, indent=2)

    print(f"Knowledge base generated at: {output_path}")


if __name__ == "__main__":
    # Path to the Fabrax progression guide
    progression_guide_path = os.path.join(
        os.path.dirname(__file__),
        "../Craftax/craftax/fabrax/fabrax_progression_guide.md"
    )

    # Call the main function
    main(progression_guide_path)
