from craftax.craftax_classic.constants import BlockType
from craftax.craftax.constants import BlockType as BlockType_craftax
import jax.numpy as jnp
import jax


def extract_first_valid_subsequence(env_states, actions, start_value=12, end_value=13):
    player_state = env_states.player_state.reshape(-1)

    # Find the first occurrence of end_value (13)
    end_indices = jnp.where(player_state == end_value)[0]

    # If no end value is found, return None
    if len(end_indices) == 0:
        return None, None

    # Get the first end_idx
    end_idx = end_indices[0]

    # Search backwards from this end_idx to find consecutive start_values (12s)
    preceding_segment = player_state[:end_idx]

    # Start from the end of the preceding segment
    start_idx = end_idx - 1

    # Walk backwards until we find a non-start value or reach the beginning
    while start_idx >= 0 and preceding_segment[start_idx] == start_value:
        start_idx -= 1

    # Adjust start_idx to point to the first element of the consecutive sequence
    start_idx += 1

    # If we didn't find any start values, return None
    print(start_idx, end_idx)
    if start_idx >= end_idx:
        return None, None

    # Return the slice of the entire env_states for this range, including both start and end values
    return (
        jax.tree_map(
            lambda x: (
                x[start_idx + 1 : end_idx + 1] if hasattr(x, "__getitem__") else x
            ),
            env_states,
        ),
        actions[start_idx + 1 : end_idx + 1],
    )


from craftax.craftax_classic.constants import (
    Action,
)


def explain_inventory_changes(subseq, actions):
    """
    Explain inventory changes in natural language for each timestep,
    including the action performed when inventory changes
    """
    explanations = []
    inventory_diff = subseq.inventory_diff

    # Get all item names from the inventory structure
    item_names = inventory_diff.__dict__.keys()

    # Find the length of the sequence by checking the first item array
    first_item = next(iter(item_names))
    seq_length = len(getattr(inventory_diff, first_item))

    # Iterate through each timestep
    for t in range(seq_length):
        timestep_changes = []

        # Check each item type in the inventory
        for item_name in item_names:
            item_array = getattr(inventory_diff, item_name)

            # If this item had a change at this timestep
            if item_array[t] != 0:
                change = item_array[t]
                if change > 0:
                    timestep_changes.append(f"Gained {int(change)} {item_name}")
                else:
                    timestep_changes.append(f"Lost {int(abs(change))} {item_name}")

        # If there were any changes at this timestep, add an explanation with the action
        if timestep_changes:
            # Get the action index for this timestep
            action_idx = actions[t]

            # Convert action index to action name
            action_name = Action._member_names_[action_idx]

            explanation = f"Timestep {t}: Action: {action_name}, " + ", ".join(
                timestep_changes
            )
            explanations.append(explanation)

    return explanations


def explain_trajectory(env_states, actions, start_state):

    subseq, actions_subseq = extract_first_valid_subsequence(
        env_states, actions, start_value=start_state, end_value=start_state + 1
    )
    explanation = explain_inventory_changes(subseq, actions_subseq)
    return explanation


def describe_block_environment(closest_blocks, game="craftax"):
    """
    Generates a natural language description of the blocks in the player's vicinity.

    Args:
        closest_blocks: A 2D array of shape (num_block_types, 5) where each row corresponds
                       to a block type, and each column represents the percentage of time
                       the player has seen X or more of that block type.
        block_type_enum: The BlockType enum mapping indices to block types

    Returns:
        A string describing the player's surroundings in natural language
    """
    # Get the number of block types
    num_block_types = closest_blocks.shape[0]

    # Store blocks and their frequency information
    visible_blocks = []

    # Process each block type
    for block_idx in range(num_block_types):
        # Skip INVALID and OUT_OF_BOUNDS types as they're not actual blocks
        if block_idx <= 1:
            continue

        # Get the name of this block type
        if game == "craftax":
            block_name = BlockType_craftax(block_idx).name.replace("_", " ").lower()
        else:
            block_name = BlockType(block_idx).name.replace("_", " ").lower()

        # Check if this block is present (>0%)
        if closest_blocks[block_idx][0] > 0:
            # Find the maximum count with non-zero presence
            max_count_idx = 4  # Start with assumption of 5+ blocks
            while max_count_idx >= 0 and closest_blocks[block_idx][max_count_idx] == 0:
                max_count_idx -= 1

            # Create description of block presence
            presence_percentages = []
            for count_idx in range(max_count_idx + 1):
                # Only include non-zero percentages
                if closest_blocks[block_idx][count_idx] > 0:
                    count_desc = ""
                    if count_idx == 0:
                        count_desc = "at least one"
                    elif count_idx == 1:
                        count_desc = "at least two"
                    elif count_idx == 2:
                        count_desc = "at least three"
                    elif count_idx == 3:
                        count_desc = "at least four"
                    elif count_idx == 4:
                        count_desc = "five or more"

                    percentage = closest_blocks[block_idx][count_idx] * 100
                    presence_percentages.append(f"{count_desc}: {percentage:.1f}%")

            # Add this block to the visible blocks list
            visible_blocks.append(
                {
                    "name": block_name,
                    "main_percentage": closest_blocks[block_idx][0] * 100,
                    "description": ", ".join(presence_percentages),
                }
            )

    # Sort blocks by their primary presence percentage (highest first)
    visible_blocks.sort(key=lambda x: x["main_percentage"], reverse=True)

    # Generate the natural language description
    if not visible_blocks:
        return "You don't see any blocks in your surroundings."

    description = "Statistical distribution of blocks observed from this position (averaged across multiple gameplay trajectories):\n"

    for block in visible_blocks:
        description += f"- {block['name']} ({block['description']})\n"

    return description
