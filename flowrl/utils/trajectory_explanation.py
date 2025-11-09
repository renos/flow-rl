from craftax.craftax_classic.constants import BlockType, Achievement as ClassicAchievement
from craftax.craftax.constants import (
    BlockType as BlockType_craftax,
    Achievement as CraftaxAchievement,
)
import jax.numpy as jnp
import jax
import numpy as np


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


from craftax.craftax_classic.constants import Action as ClassicAction
from craftax.craftax.constants import Action as CraftaxAction


def _resolve_achievement_enum(num_entries):
    """Return the achievement Enum matching the array length."""
    try:
        if num_entries == len(CraftaxAchievement):
            return CraftaxAchievement
    except Exception:
        pass
    try:
        if num_entries == len(ClassicAchievement):
            return ClassicAchievement
    except Exception:
        pass
    return None


def explain_inventory_changes(subseq, actions, game="craftax"):
    """
    Explain inventory changes in natural language for each timestep,
    including the action performed when inventory changes

    Args:
        subseq: Subsequence of environment states
        actions: Array of actions taken
        game: Either "craftax" or "craftax_classic" to determine which Action enum to use
    """
    explanations = []
    inventory_diff = subseq.inventory_diff
    achievements_diff = getattr(subseq, "achievements_diff", None)
    achievements = getattr(subseq, "achievements", None)

    # Level / stat fields that may exist on Craftax environments
    level_fields = []
    for field_name in [
        "player_level",
        "player_strength",
        "player_dexterity",
        "player_intelligence",
    ]:
        if hasattr(subseq, field_name):
            level_fields.append(field_name)
    has_monsters_killed = hasattr(subseq, "monsters_killed")

    # Get all item names from the inventory structure
    item_names = inventory_diff.__dict__.keys()

    # Find the length of the sequence by checking the first item array
    first_item = next(iter(item_names))
    seq_length = len(getattr(inventory_diff, first_item))

    achievement_enum = None
    if achievements_diff is not None:
        try:
            num_achievements = achievements_diff.shape[-1]
        except AttributeError:
            num_achievements = len(achievements_diff[0]) if len(achievements_diff) > 0 else 0
        achievement_enum = _resolve_achievement_enum(num_achievements)

    # Iterate through each timestep
    for t in range(seq_length):
        timestep_changes = []

        # Check each item type in the inventory
        for item_name in item_names:
            item_array = getattr(inventory_diff, item_name)

            # Handle array fields (armour, potions) - must have shape and be non-scalar
            is_array = hasattr(item_array[t], 'shape') and item_array[t].shape != ()

            if is_array:
                # For array fields, check each element
                changes = item_array[t]
                for idx, change in enumerate(changes):
                    if change != 0:
                        if change > 0:
                            timestep_changes.append(f"Gained {int(change)} {item_name}[{idx}]")
                        else:
                            timestep_changes.append(f"Lost {int(abs(change))} {item_name}[{idx}]")
            else:
                # Handle scalar fields (wood, stone, etc.) and 0-d arrays
                if item_array[t] != 0:
                    change = item_array[t]
                    if change > 0:
                        timestep_changes.append(f"Gained {int(change)} {item_name}")
                    else:
                        timestep_changes.append(f"Lost {int(abs(change))} {item_name}")

        # Check achievements unlocked at this timestep
        if achievements_diff is not None:
            ach_step = achievements_diff[t]
            if hasattr(ach_step, "shape"):
                ach_indices = jnp.where(ach_step)[0].tolist()
            else:
                ach_indices = [i for i, val in enumerate(ach_step) if val]
            if len(ach_indices) > 0:
                names = []
                for idx in ach_indices:
                    if achievement_enum is not None:
                        names.append(achievement_enum(idx).name.lower())
                    else:
                        names.append(f"achievement_{int(idx)}")
                timestep_changes.append(
                    "Unlocked achievements: " + ", ".join(names)
                )

        # Track level/stat increases
        for field_name in level_fields:
            field_values = getattr(subseq, field_name)
            # For the first timestep we cannot compare with previous step, so skip
            if t == 0:
                continue
            prev_val = field_values[t - 1]
            curr_val = field_values[t]
            if curr_val > prev_val:
                field_readable = field_name.replace("player_", "").replace("_", " ")
                timestep_changes.append(
                    f"Increased {field_readable} to {int(curr_val)}"
                )

        # Special case: descending floors in Craftax – annotate with monsters_killed on previous floor
        if has_monsters_killed and "player_level" in level_fields and t > 0:
            lvl_vals = getattr(subseq, "player_level")
            prev_lvl = int(lvl_vals[t - 1])
            curr_lvl = int(lvl_vals[t])
            if curr_lvl > prev_lvl:
                try:
                    mk_row = subseq.monsters_killed[t - 1]
                    # Support both 1D per-level vector and flattened arrays
                    kills_prev_floor = int(mk_row[prev_lvl]) if hasattr(mk_row, "__getitem__") else int(mk_row)
                except Exception:
                    kills_prev_floor = -1
                timestep_changes.append(
                    f"Descended to level {curr_lvl} (monsters_killed on level {prev_lvl}: {kills_prev_floor})"
                )

        # If there were any changes at this timestep, add an explanation with the action
        if timestep_changes:
            # Get the action index for this timestep
            action_idx = int(actions[t])

            # Convert action index to action name based on game type
            try:
                if game == "craftax":
                    action_name = CraftaxAction(action_idx).name
                else:
                    action_name = ClassicAction(action_idx).name
            except (ValueError, IndexError):
                action_name = f"UNKNOWN_ACTION_{action_idx}"

            explanation = f"Timestep {t}: Action: {action_name}, " + ", ".join(
                timestep_changes
            )
            explanations.append(explanation)

    return explanations


def explain_trajectory(env_states, actions, start_state, game="craftax"):
    """
    Extract and explain a trajectory segment.

    Args:
        env_states: Environment states
        actions: Array of actions
        start_state: Starting state value
        game: Either "craftax" or "craftax_classic"
    """
    subseq, actions_subseq = extract_first_valid_subsequence(
        env_states, actions, start_value=start_state, end_value=start_state + 1
    )
    explanation = explain_inventory_changes(subseq, actions_subseq, game=game)
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
