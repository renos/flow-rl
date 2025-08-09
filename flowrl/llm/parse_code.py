import ast

import numpy as np
import json
import logging
import os
import re


def get_function_name_and_arguments(code_string):
    # Parse the code string into an AST
    module = ast.parse(code_string)

    # Find the function definitions
    function_defs = [node for node in module.body if isinstance(node, ast.FunctionDef)]

    # If there are no function definitions, return None
    if not function_defs:
        return None

    # For simplicity, we'll just return the signature of the first function definition
    function_def = function_defs[0]

    input_lst = []
    # Construct the function signature (within object class)
    for arg in function_def.args.args:
        input_lst.append(arg.arg)
    return function_def.name, input_lst


def process_generated_code(
    response_cur,
    expected_functions=["task_is_done", "task_reward", "task_network_number"],
    expected_signatures={
        "task_is_done": [
            "inventory",
            "inventory_diff",
            "closest_blocks",
            "closest_blocks_prev",
            "player_intrinsics",
            "player_intrinsics_diff",
            "achievements",
            "n",
        ],
        "task_reward": [
            "inventory_diff",
            "closest_blocks",
            "closest_blocks_prev",
            "player_intrinsics_diff",
            "achievements_diff",
            "health_penalty",
        ],
        "task_network_number": [],
    },
):
    # Regex patterns to extract python code enclosed in GPT response
    patterns = [
        r"```python(.*?)```",
        r"```(.*?)```",
        r'"""(.*?)"""',
        r'""(.*?)""',
        r'"(.*?)"',
    ]
    for pattern in patterns:
        code_string = re.search(pattern, response_cur, re.DOTALL)
        if code_string is not None:
            code_string = code_string.group(1).strip()
            break
    code_string = response_cur if not code_string else code_string

    functions = []
    function_lines = []
    indent = 0
    in_comment = False

    for line in code_string.split("\n"):
        stripped_line = line.lstrip()
        current_indent = len(line) - len(stripped_line)

        if re.match(r"def .+\(.*\):", stripped_line):  # New function definition
            if (
                function_lines
            ):  # If we've been collecting lines, save the current function
                functions.append("\n".join(function_lines))
                function_lines = []  # Clear function lines

            indent = current_indent  # Update the indentation level
            function_lines.append(line)  # Start the new function block

        # elif stripped_line.startswith("\"\"\"") and stripped_line.endswith("\"\"\""):
        #     function_lines.append(line)  # Add entire comment
        #     continue

        # # Case 2: If the start has triple double quotes, add and mark in_comment as True
        # elif stripped_line.startswith("\"\"\""):
        #     in_comment = True
        #     function_lines.append(line)  # Start the comment block
        #     continue

        # # Case 3: If the end has triple double quotes and we're in a comment, add and mark in_comment as False
        # elif stripped_line.endswith("\"\"\"") and in_comment:
        #     function_lines.append(line)  # End the comment block
        #     in_comment = False
        #     continue

        # # Case 4: If you're in a comment, keep adding lines until the comment block ends
        # elif in_comment:
        #     function_lines.append(line)  # Add to the ongoing comment block
        #     continue
        elif function_lines and current_indent > indent:  # If we're within a function
            function_lines.append(line)  # Add the line to the current function block
    # handle case where the file ends but we're still in a function
    if function_lines:
        functions.append("\n".join(function_lines))

    for function in functions:
        print(function)

    # Add the Eureka Reward Signature to the environment code
    exception = ""
    try:
        name_list, input_list = [], []
        for function in functions:
            signature, input_lst = get_function_name_and_arguments(function)
            if signature:
                name_list.append(signature)
                input_list.append(input_lst)

        # Check if the expected functions are present, and if the signatures match
        for expected_function in expected_functions:
            if expected_function not in name_list:
                raise Exception(
                    f"Expected function {expected_function} not found in the generated code!"
                )
            expected_inputs = expected_signatures[expected_function]
            actual_inputs = input_list[name_list.index(expected_function)]
            if len(expected_inputs) != len(actual_inputs):
                raise Exception(
                    f"Expected number of inputs for function {expected_function} does not match the signature!"
                )
            for expected_input, actual_input in zip(
                expected_inputs, input_list[name_list.index(expected_function)]
            ):
                if expected_input != actual_input:
                    raise Exception(
                        f"Expected input {input} not found in the signature of function {expected_function}!"
                    )

    except Exception as e:
        exception += str(e)
    return functions, exception
