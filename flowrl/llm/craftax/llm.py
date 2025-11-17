import agentkit

from agentkit import Graph, SimpleDBNode

from functools import partial
import agentkit.llm_api

# import llm_api
from functools import partial

from flowrl.llm.compose_prompts import ComposeReasoningPrompt
from flowrl.llm.craftax.after_queries import *
from flowrl.llm.utils import get_ctxt
from .prompts.main import return_prompts as return_simplified_prompts
from .prompts.inventory import return_prompts as return_inventory_prompts

try:
    import llm_api

    get_query = llm_api.get_query
    default_llm_name = "yintat-gpt-4o"
except:
    get_query = agentkit.llm_api.get_query
    default_llm_name = "gpt-5"
    # default_llm_name = "google-gemini-2.0-pro-exp-02-05"
    # default_llm_name = "google-gemini-2.0-flash-thinking-exp"


def generate_graph(db=None, return_inventory_graph=False, llm_name=None):
    """
    Generate the skill learning graph.

    Args:
        db: Database dict (optional)
        return_inventory_graph: Whether to return inventory graph (optional)
        llm_name: Name of LLM to use (e.g., "gpt-5", "gpt-4o-mini"). If None, uses default.
    """
    if llm_name is None:
        llm_name = default_llm_name

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
        # Create minimal database for code generation graph
        # Note: knowledgebase is loaded separately in Flow.select_next_skill_from_knowledgebase()
        db = {
            "manual": MANUAL,
        }

    # prompts, temp_prompts = return_dense_prompts(LLM_API_FUNCTION_GPT4)
    # prompts, temp_prompts = return_sparse_prompts(LLM_API_FUNCTION_GPT4)
    prompts, temp_prompts = return_simplified_prompts(LLM_API_FUNCTION_GPT4)
    for _, v in prompts.items():
        v["prompt"] = v["prompt"].strip()
    for _, v in temp_prompts.items():
        v["prompt"] = v["prompt"].strip()

    graph = build_graph(prompts, db)
    achievements_dict = {
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
        "Achievements/make_arrow": 0.0,
        "Achievements/make_torch": 0.0,
        "Achievements/place_torch": 0.0,
        "Achievements/collect_sapphire": 0.0,
        "Achievements/collect_ruby": 0.0,
        "Achievements/make_diamond_pickaxe": 0.0,
        "Achievements/make_diamond_sword": 0.0,
        "Achievements/make_iron_armour": 0.0,
        "Achievements/make_diamond_armour": 0.0,
        "Achievements/enter_gnomish_mines": 0.0,
        "Achievements/enter_dungeon": 0.0,
        "Achievements/enter_sewers": 0.0,
        "Achievements/enter_vault": 0.0,
        "Achievements/enter_troll_mines": 0.0,
        "Achievements/enter_fire_realm": 0.0,
        "Achievements/enter_ice_realm": 0.0,
        "Achievements/enter_graveyard": 0.0,
        "Achievements/defeat_gnome_warrior": 0.0,
        "Achievements/defeat_gnome_archer": 0.0,
        "Achievements/defeat_orc_solider": 0.0,
        "Achievements/defeat_orc_mage": 0.0,
        "Achievements/defeat_lizard": 0.0,
        "Achievements/defeat_kobold": 0.0,
        "Achievements/defeat_knight": 0.0,
        "Achievements/defeat_archer": 0.0,
        "Achievements/defeat_troll": 0.0,
        "Achievements/defeat_deep_thing": 0.0,
        "Achievements/defeat_pigman": 0.0,
        "Achievements/defeat_fire_elemental": 0.0,
        "Achievements/defeat_frost_troll": 0.0,
        "Achievements/defeat_ice_elemental": 0.0,
        "Achievements/damage_necromancer": 0.0,
        "Achievements/defeat_necromancer": 0.0,
        "Achievements/eat_bat": 0.0,
        "Achievements/eat_snail": 0.0,
        "Achievements/find_bow": 0.0,
        "Achievements/fire_bow": 0.0,
        "Achievements/learn_fireball": 0.0,
        "Achievements/cast_fireball": 0.0,
        "Achievements/learn_iceball": 0.0,
        "Achievements/cast_iceball": 0.0,
        "Achievements/open_chest": 0.0,
        "Achievements/drink_potion": 0.0,
        "Achievements/enchant_sword": 0.0,
        "Achievements/enchant_armour": 0.0,
    }
    db["current"] = {
        "status": "PlayerStatus(health=9, food=9, drink=9, energy=9, mana=9)",
        "inventory": "Inventory(wood=0, stone=0, coal=0, iron=0, diamond=0, sapling=0, pickaxe=0, sword=0, bow=0, arrows=0, armour=jnp.array([0, 0, 0, 0]), torches=0, ruby=0, sapphire=0, potions=jnp.array([0, 0, 0, 0, 0, 0]), books=0)",
        "closest_blocks": "Statistical distribution of blocks observed from this position (averaged across multiple gameplay trajectories):\n- grass (at least one: 100.0%, at least two: 100.0%, at least three: 100.0%, at least four: 100.0%, five or more: 100.0%)\n- tree (at least one: 99.1%, at least two: 96.0%, at least three: 89.2%, at least four: 78.6%, five or more: 52.5%)\n- sand (at least one: 93.7%, at least two: 87.6%, at least three: 79.3%, at least four: 67.0%, five or more: 55.8%)\n- stone (at least one: 91.7%, at least two: 89.8%, at least three: 86.1%, at least four: 80.8%, five or more: 74.8%)\n- water (at least one: 68.7%, at least two: 62.0%, at least three: 54.9%, at least four: 47.4%, five or more: 42.9%)\n- path (at least one: 53.5%, at least two: 39.1%, at least three: 29.1%, at least four: 22.2%, five or more: 17.0%)\n- coal (at least one: 30.9%, at least two: 8.2%, at least three: 1.8%, at least four: 0.5%, five or more: 0.1%)\n- iron (at least one: 23.9%, at least two: 5.7%, at least three: 1.3%, at least four: 0.4%)\n- plant (at least one: 21.5%, at least two: 4.1%, at least three: 0.7%, at least four: 0.1%)\n- lava (at least one: 10.0%, at least two: 5.7%, at least three: 3.8%, at least four: 2.5%, five or more: 1.8%)\n- diamond (at least one: 1.7%, at least two: 0.1%)\n- wall (at least one: 15.0%, at least two: 8.0%, at least three: 4.0%, at least four: 2.0%, five or more: 1.0%)\n- sapphire (at least one: 0.8%, at least two: 0.05%)\n- ruby (at least one: 0.9%, at least two: 0.06%)\n- chest (at least one: 2.1%, at least two: 0.3%, at least three: 0.05%)\n",
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

    return db, graph, inventory_graph, None