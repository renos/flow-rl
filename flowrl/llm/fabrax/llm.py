import agentkit

from agentkit import Graph, SimpleDBNode

from functools import partial
import agentkit.llm_api

# import llm_api
from functools import partial

from flowrl.llm.compose_prompts import ComposeReasoningPrompt
from flowrl.llm.fabrax.after_queries import *
from flowrl.llm.utils import get_ctxt
from .prompts.main import return_prompts as return_simplified_prompts
from .prompts.inventory import return_prompts as return_inventory_prompts

try:
    import llm_api

    get_query = llm_api.get_query
    llm_name = "yintat-gpt-4o"
except:
    get_query = agentkit.llm_api.get_query
    llm_name = "gpt-5"
    # llm_name = "google-gemini-2.0-pro-exp-02-05"
    # llm_name = "google-gemini-2.0-flash-thinking-exp"


def generate_graph(db=None, return_inventory_graph=False):

    MANUAL = get_ctxt()

    LLM_API_FUNCTION_GPT4 = partial(get_query(llm_name), max_gen=4096)

    def build_graph(prompts, database={}):
        if "shorthands" not in database.keys():
            database["shorthands"] = {}
        if "skills" not in database.keys():
            database["skills"] = {}
        if "knowledge_base" not in database.keys():
            database["knowledge_base"] = {}

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

    if db is None:
        # Load the knowledgebase
        import json
        import os
        knowledgebase_path = os.path.join(os.path.dirname(__file__), "../../../resources/fabrax_knowledgebase_verified.json")
        try:
            with open(knowledgebase_path, 'r') as f:
                knowledgebase = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Could not find knowledgebase at {knowledgebase_path}")
            knowledgebase = {}

        db = {
            "manual": MANUAL,
            "knowledge_base": json.dumps(knowledgebase, indent=2),
        }

    # prompts, temp_prompts = return_dense_prompts(LLM_API_FUNCTION_GPT4)
    # prompts, temp_prompts = return_sparse_prompts(LLM_API_FUNCTION_GPT4)
    prompts, temp_prompts = return_simplified_prompts(LLM_API_FUNCTION_GPT4)
    for _, v in prompts.items():
        v["prompt"] = v["prompt"].strip()
    for _, v in temp_prompts.items():
        v["prompt"] = v["prompt"].strip()

    graph = build_graph(prompts, db)
    # Fabrax achievements (IDs 0-21 from Craftax Classic + new 22-53)
    achievements_dict = {
        # Core Craftax Classic achievements (0-21)
        "Achievements/collect_wood": 0.0,
        "Achievements/place_table": 0.0,
        "Achievements/eat_cow": 0.0,
        "Achievements/collect_sapling": 0.0,
        "Achievements/collect_drink": 0.0,
        "Achievements/make_wood_pickaxe": 0.0,
        "Achievements/make_wood_sword": 0.0,
        "Achievements/place_plant": 0.0,
        "Achievements/defeat_zombie": 0.0,
        "Achievements/collect_stone": 0.0,
        "Achievements/place_stone": 0.0,
        "Achievements/eat_plant": 0.0,
        "Achievements/defeat_skeleton": 0.0,
        "Achievements/make_stone_pickaxe": 0.0,
        "Achievements/make_stone_sword": 0.0,
        "Achievements/wake_up": 0.0,
        "Achievements/place_furnace": 0.0,
        "Achievements/collect_coal": 0.0,
        "Achievements/collect_iron": 0.0,
        "Achievements/collect_diamond": 0.0,
        "Achievements/make_iron_pickaxe": 0.0,
        "Achievements/make_iron_sword": 0.0,
        # Metallurgy branch (22-33)
        "Achievements/place_anvil": 0.0,
        "Achievements/make_bellows": 0.0,
        "Achievements/smelt_iron_bar": 0.0,
        "Achievements/collect_copper": 0.0,
        "Achievements/collect_tin": 0.0,
        "Achievements/make_steel_bar": 0.0,
        "Achievements/make_bronze_bar": 0.0,
        "Achievements/forge_steel_pickaxe": 0.0,
        "Achievements/forge_bronze_pickaxe": 0.0,
        "Achievements/harden_edge": 0.0,
        "Achievements/forge_steel_sword": 0.0,
        "Achievements/forge_bronze_sword": 0.0,
        # Glass & Ceramics branch (34-46)
        "Achievements/collect_sand": 0.0,
        "Achievements/place_kiln": 0.0,
        "Achievements/smelt_glass": 0.0,
        "Achievements/collect_clay": 0.0,
        "Achievements/fire_brick": 0.0,
        "Achievements/make_glass_bottle": 0.0,
        "Achievements/make_lens": 0.0,
        "Achievements/collect_limestone": 0.0,
        "Achievements/make_lime": 0.0,
        "Achievements/make_telescope": 0.0,
        "Achievements/make_mortar": 0.0,
        "Achievements/place_window": 0.0,
        "Achievements/place_wall_masonry": 0.0,
        # Chemistry & Alchemy branch (47-53)
        "Achievements/place_composter": 0.0,
        "Achievements/make_tar": 0.0,
        "Achievements/make_fertilizer": 0.0,
        "Achievements/place_alchemy_bench": 0.0,
        "Achievements/brew_tonic": 0.0,
        "Achievements/brew_stone_skin": 0.0,
        "Achievements/make_flux": 0.0,
    }
    db["current"] = {
        "status": "PlayerStatus(health=9, food=9, drink=9, energy=9)",
        # Fabrax has explicit pickaxe names (wood_pickaxe, stone_pickaxe, iron_pickaxe, bronze_pickaxe, steel_pickaxe)
        # instead of tier-based pickaxe levels
        "inventory": "Inventory(wood=0, stone=0, coal=0, iron=0, diamond=0, sapling=0, copper=0, tin=0, sand=0, clay=0, limestone=0, leather=0, wood_pickaxe=0, stone_pickaxe=0, iron_pickaxe=0, bronze_pickaxe=0, steel_pickaxe=0, wood_sword=0, stone_sword=0, iron_sword=0, bronze_sword=0, steel_sword=0, iron_bar=0, steel_bar=0, bronze_bar=0, glass=0, bottle=0, lens=0, telescope=0, brick=0, lime=0, mortar=0, tar=0, fertilizer=0, flux=0)",
        "closest_blocks": "Statistical distribution of blocks observed from this position (averaged across multiple gameplay trajectories):\n- grass (at least one: 100.0%, at least two: 100.0%, at least three: 100.0%, at least four: 100.0%, five or more: 100.0%)\n- tree (at least one: 99.1%, at least two: 96.0%, at least three: 89.2%, at least four: 78.6%, five or more: 52.5%)\n- sand (at least one: 93.7%, at least two: 87.6%, at least three: 79.3%, at least four: 67.0%, five or more: 55.8%)\n- stone (at least one: 91.7%, at least two: 89.8%, at least three: 86.1%, at least four: 80.8%, five or more: 74.8%)\n- water (at least one: 68.7%, at least two: 62.0%, at least three: 54.9%, at least four: 47.4%, five or more: 42.9%)\n- path (at least one: 53.5%, at least two: 39.1%, at least three: 29.1%, at least four: 22.2%, five or more: 17.0%)\n- coal (at least one: 30.9%, at least two: 8.2%, at least three: 1.8%, at least four: 0.5%, five or more: 0.1%)\n- iron (at least one: 23.9%, at least two: 5.7%, at least three: 1.3%, at least four: 0.4%)\n- copper (at least one: 20.5%, at least two: 4.5%, at least three: 0.9%, at least four: 0.2%)\n- tin (at least one: 19.8%, at least two: 4.2%, at least three: 0.8%, at least four: 0.1%)\n- clay (at least one: 15.3%, at least two: 3.1%, at least three: 0.6%)\n- limestone (at least one: 25.1%, at least two: 6.8%, at least three: 1.5%, at least four: 0.3%)\n- plant (at least one: 21.5%, at least two: 4.1%, at least three: 0.7%, at least four: 0.1%)\n- lava (at least one: 10.0%, at least two: 5.7%, at least three: 3.8%, at least four: 2.5%, five or more: 1.8%)\n- diamond (at least one: 1.7%, at least two: 0.1%)\n",
        "achievements_completed": achievements_dict,
        "completed_subtasks": [],
        "num_skills": 0,
    }
    db["prompts"] = prompts
    db["temp_prompts"] = temp_prompts

    if not return_inventory_graph:
        return db, graph

    inventory_prompts = return_inventory_prompts(LLM_API_FUNCTION_GPT4)
    inventory_graph = build_graph(inventory_prompts, db)

    return db, graph, inventory_graph, _
