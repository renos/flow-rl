import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

from craftax.craftax_env import make_craftax_env_from_name


class InteractivePositionDebugger:
    def __init__(self):
        self.current_env = None
        self.current_state = None
        self.env_params = None
        self.OBS_DIM = None
        self.BlockType = None
        self.Action = None
        self.ItemType = None  # Only for non-Classic Craftax
        self.env_name = ""

    def load_env(self, env_name, seed=42):
        """Load an environment for debugging."""
        try:
            self.env_name = env_name
            # Use auto-reset envs for simple stepping in notebooks
            self.current_env = make_craftax_env_from_name(env_name, auto_reset=True)
            self.env_params = self.current_env.default_params

            # Reset environment
            rng = jax.random.PRNGKey(seed)
            obs, self.current_state = self.current_env.reset(rng, self.env_params)

            # Import correct constants and action enums
            if "Classic" in env_name:
                from craftax.craftax_classic.constants import OBS_DIM, BlockType, Action

                self.ItemType = None
            else:
                from craftax.craftax.constants import (
                    OBS_DIM,
                    BlockType,
                    Action,
                    ItemType,
                )

                self.ItemType = ItemType

            self.OBS_DIM = OBS_DIM
            self.BlockType = BlockType
            self.Action = Action

            print(f"✅ Loaded {env_name}")
            print(f"OBS_DIM: {OBS_DIM}")
            print(f"Starting position: {self.current_state.player_position}")
            self.render_current_state()

        except Exception as e:
            print(f"❌ Error loading {env_name}: {e}")

    def get_block_name(self, block_id):
        """Get readable block name from block ID."""
        try:
            return list(self.BlockType)[int(block_id)].name
        except Exception:
            return f"B{int(block_id)}"

    def get_short_block_name(self, block_id):
        """Get shortened block name for display."""
        name = self.get_block_name(block_id)
        short_names = {
            "GRASS": "GRS",
            "WATER": "H2O",
            "STONE": "STN",
            "TREE": "TRE",
            "WOOD": "WOD",
            "COAL": "COL",
            "IRON": "IRN",
            "DIAMOND": "DIA",
            "CRAFTING_TABLE": "TBL",
            "FURNACE": "FUR",
            "SAND": "SND",
            "LAVA": "LVA",
            "PLANT": "PLT",
            "RIPE_PLANT": "RPL",
            "OUT_OF_BOUNDS": "OOB",
            "INVALID": "INV",
            "PATH": "PTH",
        }
        return short_names.get(name, name[:3])

    def move(self, action):
        """Execute an action and update state."""
        if self.current_env is None:
            print("❌ No environment loaded! Use load_env() first")
            return

        # Convert string action to enum if needed
        if isinstance(action, str):
            action = getattr(self.Action, action.upper())
        if hasattr(action, "value"):
            action = action.value

        # Execute action
        rng = jax.random.PRNGKey(np.random.randint(0, 1_000_000))
        old_pos = jnp.array(self.current_state.player_position).copy()

        obs, self.current_state, reward, done, info = self.current_env.step(
            rng, self.current_state, action, self.env_params
        )

        new_pos = self.current_state.player_position

        clear_output(wait=True)
        print(f"🎮 Action executed! Position: {tuple(map(int, old_pos.tolist()))} → {tuple(map(int, new_pos.tolist()))}")
        if float(reward) != 0.0:
            print(f"💰 Reward: {float(reward):.3f}")

        self.render_current_state()

    def teleport_to_grass_near_down_ladder(self):
        """Teleport player to the nearest GRASS tile adjacent to the down staircase on the current floor.

        If no GRASS nearby, fall back to nearest PATH tile. Classic envs do not support ladders and will skip.
        """
        if self.current_env is None or self.current_state is None:
            print("❌ No environment loaded! Use load_env() first")
            return

        if "Classic" in self.env_name:
            print("⚠️ Classic env has no ladders/floors; teleport not available.")
            return

        try:
            lvl = int(self.current_state.player_level)
            world_map = self.current_state.map[lvl]

            # Locate down ladder position
            ladder_yx = None
            if hasattr(self.current_state, "down_ladders"):
                ladder_yx = jnp.array(self.current_state.down_ladders[lvl])
            elif hasattr(self.current_state, "item_map") and self.ItemType is not None:
                item_layer = self.current_state.item_map[lvl]
                ys, xs = jnp.where(item_layer == self.ItemType.LADDER_DOWN.value)
                if ys.size > 0:
                    ladder_yx = jnp.array([ys[0], xs[0]])

            if ladder_yx is None:
                print("⚠️ No down ladder found on this floor.")
                return

            # Print ladder counts on this floor (if item_map available)
            if hasattr(self.current_state, "item_map") and self.ItemType is not None:
                item_layer = self.current_state.item_map[lvl]
                down_n = int(jnp.sum(item_layer == self.ItemType.LADDER_DOWN.value))
                up_n = int(jnp.sum(item_layer == self.ItemType.LADDER_UP.value))
                blocked_n = int(
                    jnp.sum(item_layer == self.ItemType.LADDER_DOWN_BLOCKED.value)
                )
                total_n = down_n + up_n + blocked_n
                print(
                    f"🔎 Ladder tiles on floor {lvl}: down={down_n}, up={up_n}, blocked={blocked_n}, total={total_n}"
                )

            # Masks for candidate landing tiles
            grass_mask = world_map == self.BlockType.GRASS.value
            path_mask = world_map == self.BlockType.PATH.value

            def pick_nearest(mask):
                ys, xs = jnp.where(mask)
                if ys.size == 0:
                    return None
                dy = ys - ladder_yx[0]
                dx = xs - ladder_yx[1]
                d2 = dy * dy + dx * dx
                k = jnp.argmin(d2)
                return jnp.array([ys[k], xs[k]])

            target = pick_nearest(grass_mask)
            if target is None:
                target = pick_nearest(path_mask)

            if target is None:
                print("⚠️ No suitable GRASS/PATH tile found to teleport.")
                return

            # Teleport by replacing player_position in the env state
            self.current_state = self.current_state.replace(
                player_position=target.astype(jnp.int32)
            )

            clear_output(wait=True)
            print(
                f"🧭 Teleported to near ladder: {tuple(map(int, target.tolist()))}"
            )
            self.render_current_state()

        except Exception as e:
            print(f"❌ Teleport failed: {e}")

    def render_current_state(self):
        """Render current state with relative positions."""
        if self.current_state is None:
            return

        # Get current map layer
        if hasattr(self.current_state, "player_level"):
            current_map = self.current_state.map[self.current_state.player_level]
        else:
            current_map = self.current_state.map

        # Extract local view around player
        player_pos = jnp.array(self.current_state.player_position)
        half_obs = jnp.array(self.OBS_DIM) // 2

        start_pos = player_pos - half_obs
        end_pos = player_pos + half_obs + 1

        # Clamp to bounds
        map_shape = jnp.array(current_map.shape)
        actual_start = jnp.maximum(start_pos, 0)
        actual_end = jnp.minimum(end_pos, map_shape)

        local_map = current_map[
            actual_start[0] : actual_end[0], actual_start[1] : actual_end[1]
        ]

        # Offset from clamping
        offset_y = int(actual_start[0] - start_pos[0])
        offset_x = int(actual_start[1] - start_pos[1])

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        # Map view
        ax1.imshow(np.array(local_map), cmap="tab20", alpha=0.7)
        ax1.set_title(
            f"{self.env_name}\nOBS_DIM: {tuple(self.OBS_DIM)}\nPlayer: {tuple(map(int, player_pos.tolist()))}"
        )
        ax1.grid(True, alpha=0.3)

        # Player marker at center (account for clipping)
        center_y = int(half_obs[0]) + offset_y
        center_x = int(half_obs[1]) + offset_x
        if 0 <= center_y < local_map.shape[0] and 0 <= center_x < local_map.shape[1]:
            ax1.plot(
                center_x,
                center_y,
                "ro",
                markersize=20,
                label="Player",
                markeredgecolor="yellow",
                markeredgewidth=3,
            )

        # Optional local item overlay window (for ladder labels)
        item_local = None
        lvl_for_items = None
        if self.ItemType is not None and hasattr(self.current_state, "item_map"):
            lvl_for_items = int(self.current_state.player_level) if hasattr(self.current_state, "player_level") else 0
            item_layer = self.current_state.item_map[lvl_for_items]
            item_local = item_layer[actual_start[0] : actual_end[0], actual_start[1] : actual_end[1]]

        # Per-cell annotations
        for i in range(local_map.shape[0]):
            for j in range(local_map.shape[1]):
                block_id = int(local_map[i, j])
                block_name = self.get_short_block_name(block_id)
                # If Craftax (non-Classic), override with ladder label when present (use local item window)
                if item_local is not None:
                    item_val = int(item_local[i, j])
                    if item_val in (
                        self.ItemType.LADDER_DOWN.value,
                        self.ItemType.LADDER_UP.value,
                        self.ItemType.LADDER_DOWN_BLOCKED.value,
                    ):
                        block_name = "LAD"
                rel_x = j - center_x
                rel_y = i - center_y
                text_color = (
                    "white" if block_id in [0, 1, self.BlockType.STONE.value, self.BlockType.COAL.value, self.BlockType.IRON.value, self.BlockType.DIAMOND.value] else "black"
                )
                ax1.text(
                    j,
                    i - 0.15,
                    block_name,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                    weight="bold",
                )
                ax1.text(
                    j,
                    i + 0.15,
                    f"({rel_x},{rel_y})",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=text_color,
                    weight="bold",
                )

        ax1.legend()

        # Relative positions panel
        ax2.text(
            0.02,
            0.98,
            f"CLOSEST BLOCKS ({self.env_name}):",
            transform=ax2.transAxes,
            fontweight="bold",
            fontsize=12,
        )

        if hasattr(self.current_state, "closest_blocks"):
            closest = self.current_state.closest_blocks
            total_blocks = int(closest.shape[0])
            y_pos = 0.94

            # Show first 10 blocks
            max_blocks = min(7, total_blocks)
            for block_id in range(max_blocks):
                positions = closest[block_id]
                block_name = self.get_block_name(block_id)

                ax2.text(
                    0.02,
                    y_pos,
                    f"{block_id}: {block_name}",
                    transform=ax2.transAxes,
                    fontweight="bold",
                    fontsize=9,
                )
                y_pos -= 0.025

                valid_count = 0
                max_k = min(5, int(positions.shape[1]))
                for i in range(max_k):
                    x, y = float(positions[0, i]), float(positions[1, i])
                    is_valid = abs(x) < 30 and abs(y) < 30
                    if is_valid:
                        valid_count += 1
                    color = "green" if is_valid else "red"
                    status = "✓" if is_valid else "✗"
                    ax2.text(
                        0.04,
                        y_pos,
                        f"  [{i}]: ({x:.0f},{y:.0f}) {status}",
                        transform=ax2.transAxes,
                        fontsize=7,
                        color=color,
                    )
                    y_pos -= 0.02

                total_positions = int(positions.shape[1])
                ax2.text(
                    0.04,
                    y_pos,
                    f"  → {valid_count}/{total_positions} valid",
                    transform=ax2.transAxes,
                    fontsize=7,
                    color=("green" if valid_count > 0 else "red"),
                    style="italic",
                )
                y_pos -= 0.03

            # Add separator and show last 3 blocks (e.g., ladders)
            if total_blocks > max_blocks:
                ax2.text(
                    0.02,
                    y_pos,
                    "..." + "─" * 30,
                    transform=ax2.transAxes,
                    fontsize=8,
                    color="gray",
                    style="italic",
                )
                y_pos -= 0.03

                last_3_start = max(0, total_blocks - 3)
                for block_id in range(last_3_start, total_blocks):
                    positions = closest[block_id]
                    block_name = self.get_block_name(block_id)

                    ax2.text(
                        0.02,
                        y_pos,
                        f"{block_id}: {block_name}",
                        transform=ax2.transAxes,
                        fontweight="bold",
                        fontsize=9,
                        color="blue",  # Different color to distinguish last items
                    )
                    y_pos -= 0.025

                    valid_count = 0
                    max_k = min(5, int(positions.shape[1]))
                    for i in range(max_k):
                        x, y = float(positions[0, i]), float(positions[1, i])
                        is_valid = abs(x) < 30 and abs(y) < 30
                        if is_valid:
                            valid_count += 1
                        color = "green" if is_valid else "red"
                        status = "✓" if is_valid else "✗"
                        ax2.text(
                            0.04,
                            y_pos,
                            f"  [{i}]: ({x:.0f},{y:.0f}) {status}",
                            transform=ax2.transAxes,
                            fontsize=7,
                            color=color,
                        )
                        y_pos -= 0.02

                    total_positions = int(positions.shape[1])
                    ax2.text(
                        0.04,
                        y_pos,
                        f"  → {valid_count}/{total_positions} valid",
                        transform=ax2.transAxes,
                        fontsize=7,
                        color=("green" if valid_count > 0 else "red"),
                        style="italic",
                    )
                    y_pos -= 0.03
        else:
            ax2.text(
                0.02,
                0.5,
                "No closest_blocks found",
                transform=ax2.transAxes,
                fontsize=12,
            )

        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis("off")

        plt.tight_layout()
        plt.show()

        # Print legend
        print("\n📋 BLOCK LEGEND:")
        unique_blocks = jnp.unique(local_map)
        legend_text = []
        for b in unique_blocks:
            full_name = self.get_block_name(int(b))
            short_name = self.get_short_block_name(int(b))
            legend_text.append(f"{short_name}={full_name}")

        for i in range(0, len(legend_text), 4):
            print("  " + " | ".join(legend_text[i : i + 4]))


# Create debugger instance
debugger = InteractivePositionDebugger()


# Action shortcuts
def up():
    debugger.move("UP")


def down():
    debugger.move("DOWN")


def left():
    debugger.move("LEFT")


def right():
    debugger.move("RIGHT")


def do():
    debugger.move("DO")


def noop():
    debugger.move("NOOP")


def tp_to_ladder_grass():
    debugger.teleport_to_grass_near_down_ladder()


# Multi-step movements
def move_sequence(actions):
    """Execute a sequence of actions."""
    for action in actions:
        debugger.move(action)


def walk_right(steps=5):
    """Walk right for multiple steps."""
    for i in range(steps):
        print(f"Step {i + 1}/{steps}")
        debugger.move("RIGHT")


def walk_up(steps=5):
    """Walk up for multiple steps."""
    for i in range(steps):
        print(f"Step {i + 1}/{steps}")
        debugger.move("UP")


# Load environments helpers
def load_classic():
    debugger.load_env("Craftax-Classic-Symbolic-v1")


def load_craftax():
    debugger.load_env("Craftax-Symbolic-v1")


def load_fabrax():
    try:
        debugger.load_env("Fabrax-Symbolic-v1")
    except Exception:
        print("❌ Fabrax not available, loading Classic instead")
        debugger.load_env("Craftax-Classic-Symbolic-v1")


def show_controls():
    """Show available controls."""
    print("=" * 60)
    print("🎮 INTERACTIVE POSITION DEBUGGER CONTROLS")
    print("=" * 60)
    print("\n📱 LOAD ENVIRONMENTS:")
    print("load_classic()    # Load Craftax Classic")
    print("load_craftax()    # Load Craftax")
    print("load_fabrax()     # Load Fabrax")
    print("\n🕹️  MOVEMENT:")
    print("up(), down(), left(), right()  # Move one step")
    print("do()                          # Interact with environment")
    print("noop()                        # Do nothing")
    print("\n🧭 TELEPORT:")
    print("tp_to_ladder_grass()          # Teleport near down staircase (Craftax)")
    print("\n🚶 MULTI-STEP:")
    print("walk_right(5)                 # Walk right 5 steps")
    print("walk_up(3)                    # Walk up 3 steps")
    print("move_sequence(['UP','UP','RIGHT','DOWN'])  # Execute sequence")
    print("\n🔍 WHAT YOU'LL SEE:")
    print("• Each grid cell shows: BLOCK_NAME on top, (x,y) coordinates below")
    print("• Red player marker = your position")
    print("• Right panel shows closest_blocks with ✓ = valid, ✗ = invalid (30,30)")
    print("• Block legend printed below the visualization")
    print("\n💡 USAGE:")
    print("1. Load an environment: load_classic() or load_craftax()")
    print("2. Move around: up(), right(), walk_right(10)")
    print("3. Teleport near ladder to debug relative positions easily")
