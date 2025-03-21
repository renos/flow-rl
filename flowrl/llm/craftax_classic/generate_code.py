from flowrl.llm.parse_code import process_generated_code
from flowrl.llm.craftax_classic.verify_code import verify_function
import os


def generate_py_file(data, file_path="generated_file.py"):

    with open(file_path, "w") as f:
        f.write("from craftax_classic.constants import *\n")
        f.write("from craftax_classic.envs.craftax_state import Inventory\n")
        f.write("import jax\n")

    task_count = 0  # Initialize the counter for task tracking

    for i, (key, response) in enumerate(data.items()):
        if i > 0:  # Skip the first item, as it generates tasks
            functions, exception = process_generated_code(response)
            if exception != "":
                print(False, exception, False, exception)
                continue

            # Verify functions to ensure they are correct
            (
                is_task_done_successful,
                is_task_done_exception,
                task_reward_successful,
                task_reward_exception,
            ) = verify_function(functions)

            # Print the verification results
            print(
                is_task_done_successful,
                is_task_done_exception,
                task_reward_successful,
                task_reward_exception,
            )

            # If verification fails, print the functions for debugging
            if not is_task_done_successful or not task_reward_successful:
                print(functions[0])
                print(functions[1])
                continue  # Skip adding invalid functions to the file

            # Replace function names with unique names
            functions[0] = functions[0].replace(
                "task_is_done", f"task_{task_count}_is_done"
            )
            functions[1] = functions[1].replace(
                "task_reward", f"task_{task_count}_reward"
            )

            # Write the modified functions into the Python file
            with open(file_path, "a") as f:
                f.write(functions[0] + "\n")
                f.write(functions[1] + "\n")

            task_count += 1  # Increment the counter for the next set of functions


def validate_code(generated_code):
    functions, exception = process_generated_code(generated_code)
    if exception != "":
        return (
            None,
            exception,
        )
    else:
        (
            is_task_done_successful,
            is_task_done_exception,
            task_reward_successful,
            task_reward_exception,
        ) = verify_function(functions)
    if not is_task_done_successful or not task_reward_successful:
        return (
            None,
            f"Task done exception `{is_task_done_exception}` Task reward exception `{task_reward_exception}`",
        )
    return functions, ""


def generate_validated_py(functions, file_path, task_num):
    if task_num == 0:
        with open(file_path, "w") as f:
            f.write("from craftax.craftax_classic.constants import *\n")
            f.write(
                "from craftax.craftax_classic.envs.craftax_state import Inventory\n"
            )
            f.write("import jax\n")

    all_functions = ""
    for i in range(3):
        all_functions += f"{functions[i]}\n"

    functions[0] = functions[0].replace("task_is_done", f"task_{task_num}_is_done")
    functions[1] = functions[1].replace("task_reward", f"task_{task_num}_reward")
    functions[2] = functions[2].replace(
        "task_network_number", f"task_{task_num}_network_number"
    )

    with open(file_path, "a") as f:
        f.write(functions[0] + "\n")
        f.write(functions[1] + "\n")
        f.write(functions[2] + "\n")
    return all_functions


from flax import struct
from dataclasses import dataclass, field
from typing import Tuple

# from craftax_classic.envs.craftax_state import EnvState, Inventory, Mobs
# @struct.dataclass
# class Inventory:
#     wood: Tuple[int, ...] = field(default_factory=lambda: tuple(0 for _ in range(10)))
#     stone: Tuple[int, ...] = field(default_factory=lambda: tuple(0 for _ in range(10)))
#     coal: Tuple[int, ...] = field(default_factory=lambda: tuple(0 for _ in range(10)))
#     iron: Tuple[int, ...] = field(default_factory=lambda: tuple(0 for _ in range(10)))
#     diamond: Tuple[int, ...] = field(default_factory=lambda: tuple(0 for _ in range(10)))
#     sapling: Tuple[int, ...] = field(default_factory=lambda: tuple(0 for _ in range(10)))
#     wood_pickaxe: Tuple[int, ...] = field(default_factory=lambda: tuple(0 for _ in range(10)))
#     stone_pickaxe: Tuple[int, ...] = field(default_factory=lambda: tuple(0 for _ in range(10)))
#     iron_pickaxe: Tuple[int, ...] = field(default_factory=lambda: tuple(0 for _ in range(10)))
#     wood_sword: Tuple[int, ...] = field(default_factory=lambda: tuple(0 for _ in range(10)))
#     stone_sword: Tuple[int, ...] = field(default_factory=lambda: tuple(0 for _ in range(10)))
#     iron_sword: Tuple[int, ...] = field(default_factory=lambda: tuple(0 for _ in range(10)))

#     def __str__(self):
#         def format_item(name, lst):
#             return f"{name}: " + ", ".join([f"P({i})={p}" for i, p in enumerate(lst)])

#         return (
#             "Inventory:\n"
#             f"{format_item('wood', self.wood)}\n"
#             f"{format_item('stone', self.stone)}\n"
#             f"{format_item('coal', self.coal)}\n"
#             f"{format_item('iron', self.iron)}\n"
#             f"{format_item('diamond', self.diamond)}\n"
#             f"{format_item('sapling', self.sapling)}\n"
#             f"{format_item('wood_pickaxe', self.wood_pickaxe)}\n"
#             f"{format_item('stone_pickaxe', self.stone_pickaxe)}\n"
#             f"{format_item('iron_pickaxe', self.iron_pickaxe)}\n"
#             f"{format_item('wood_sword', self.wood_sword)}\n"
#             f"{format_item('stone_sword', self.stone_sword)}\n"
#             f"{format_item('iron_sword', self.iron_sword)}\n"
#         )


# @struct.dataclass
# class Inventory:
#     wood: int = 0
#     stone: Tuple[int, ...] = field(default_factory=lambda: tuple(0 for _ in range(10)))
#     coal: Tuple[int, ...] = field(default_factory=lambda: tuple(0 for _ in range(10)))
#     iron: Tuple[int, ...] = field(default_factory=lambda: tuple(0 for _ in range(10)))
#     diamond: Tuple[int, ...] = field(
#         default_factory=lambda: tuple(0 for _ in range(10))
#     )
#     sapling: Tuple[int, ...] = field(
#         default_factory=lambda: tuple(0 for _ in range(10))
#     )
#     wood_pickaxe: Tuple[int, ...] = field(
#         default_factory=lambda: tuple(0 for _ in range(10))
#     )
#     stone_pickaxe: Tuple[int, ...] = field(
#         default_factory=lambda: tuple(0 for _ in range(10))
#     )
#     iron_pickaxe: Tuple[int, ...] = field(
#         default_factory=lambda: tuple(0 for _ in range(10))
#     )
#     wood_sword: Tuple[int, ...] = field(
#         default_factory=lambda: tuple(0 for _ in range(10))
#     )
#     stone_sword: Tuple[int, ...] = field(
#         default_factory=lambda: tuple(0 for _ in range(10))
#     )
#     iron_sword: Tuple[int, ...] = field(
#         default_factory=lambda: tuple(0 for _ in range(10))
#     )

#     def __str__(self):
#         def format_item(name, lst):
#             mean_num = round(sum(i * p for i, p in enumerate(lst)), 1)
#             return f"{name}: {mean_num}"


#         return (
#             "Inventory:\n"
#             f"{format_item('wood', self.wood)}\n"
#             f"{format_item('stone', self.stone)}\n"
#             f"{format_item('coal', self.coal)}\n"
#             f"{format_item('iron', self.iron)}\n"
#             f"{format_item('diamond', self.diamond)}\n"
#             f"{format_item('sapling', self.sapling)}\n"
#             f"{format_item('wood_pickaxe', self.wood_pickaxe)}\n"
#             f"{format_item('stone_pickaxe', self.stone_pickaxe)}\n"
#             f"{format_item('iron_pickaxe', self.iron_pickaxe)}\n"
#             f"{format_item('wood_sword', self.wood_sword)}\n"
#             f"{format_item('stone_sword', self.stone_sword)}\n"
#             f"{format_item('iron_sword', self.iron_sword)}\n"
#         )
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


def create_inventory_from_array(inventory_array):
    return Inventory(
        wood=round(inventory_array[0]),
        stone=round(inventory_array[1]),
        coal=round(inventory_array[2]),
        iron=round(inventory_array[3]),
        diamond=round(inventory_array[4]),
        sapling=round(inventory_array[5]),
        wood_pickaxe=round(inventory_array[6]),
        stone_pickaxe=round(inventory_array[7]),
        iron_pickaxe=round(inventory_array[8]),
        wood_sword=round(inventory_array[9]),
        stone_sword=round(inventory_array[10]),
        iron_sword=round(inventory_array[11]),
    )
