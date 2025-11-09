"""
Staircase Dungeon environment with configurable grid layouts.

The default setup mirrors the original 10x10 grid where each floor contains two
staircases (type 2 and type 3) placed at random tiles. One staircase TYPE per
floor is correct (90% success, 10% death) while the other is lethal. The correct
TYPE pattern is fixed across episodes but must be discovered through trial and
error, with new staircase locations sampled each episode.

This module also supports corridor-style floors (e.g. 50x1) where staircases are
anchored to opposite ends of the corridor while the agent spawns somewhere in
between them, producing much longer horizons without changing the underlying
decision problem.

Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=NOOP
"""

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces
from typing import Tuple, Optional
import chex


@struct.dataclass
class EnvState:
    """State of the staircase environment."""
    # All grids for all floors (generated once on reset)
    grids: jnp.ndarray  # (num_floors, grid_height, grid_width) - 0=empty, 2=stair_type_2, 3=stair_type_3

    # Agent state
    agent_pos: jnp.ndarray  # (2,) - (x, y) position
    current_floor: int  # 0 to 29

    # Episode info
    timestep: int
    done: bool


@struct.dataclass
class EnvParams:
    """Parameters for the staircase environment."""
    max_timesteps: int = 2000
    success_prob: float = 0.9  # Probability of success on correct staircase


@struct.dataclass
class StaticEnvParams:
    """Static parameters that must be known at compile time."""
    num_floors: int = 30
    grid_height: int = 10
    grid_width: int = 10
    place_stairs_at_ends: bool = False
    # Which staircase TYPE is correct for each floor
    # True = Type 2 staircase is correct, False = Type 3 staircase is correct
    # Default pattern generated with random.seed(42)
    correct_stair_pattern: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.array([
            True, True, False, True, True, True, True, True, False, True,
            True, True, True, True, True, True, False, True, False, False,
            True, True, False, False, False, True, True, False, True, True
        ], dtype=bool)
    )


class StaircaseEnv(environment.Environment):
    """Simple staircase dungeon environment."""

    def __init__(self, static_params: Optional[StaticEnvParams] = None):
        super().__init__()
        if static_params is None:
            static_params = StaticEnvParams()
        if static_params.place_stairs_at_ends and static_params.grid_width < 3:
            raise ValueError("place_stairs_at_ends requires grid_width >= 3")
        self.static_params = static_params

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(
        self,
        rng: chex.PRNGKey,
        state: EnvState,
        action: int,
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Execute one timestep."""

        # Get current floor grid
        current_grid = state.grids[state.current_floor]

        grid_height = self.static_params.grid_height
        grid_width = self.static_params.grid_width

        # Calculate new position based on action
        # Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=NOOP
        dx = jnp.array([0, 0, -1, 1, 0], dtype=jnp.int32)[action]
        dy = jnp.array([-1, 1, 0, 0, 0], dtype=jnp.int32)[action]

        new_x = jnp.clip(state.agent_pos[0] + dx, 0, grid_width - 1).astype(jnp.int32)
        new_y = jnp.clip(state.agent_pos[1] + dy, 0, grid_height - 1).astype(jnp.int32)
        new_pos = jnp.array([new_x, new_y], dtype=jnp.int32)

        # Check what's at the new position
        tile_type = current_grid[new_y, new_x]

        # Check if agent stepped on a staircase (type 2 or type 3)
        on_stair_type_2 = tile_type == 2
        on_stair_type_3 = tile_type == 3
        on_any_stair = on_stair_type_2 | on_stair_type_3

        # Determine which staircase type is correct for this floor
        # If correct_stair_pattern[floor] is True, then type 2 is correct
        # If correct_stair_pattern[floor] is False, then type 3 is correct
        type_2_is_correct = self.static_params.correct_stair_pattern[state.current_floor]

        # Check if agent stepped on the correct staircase type
        on_correct_stair = (on_stair_type_2 & type_2_is_correct) | (on_stair_type_3 & (~type_2_is_correct))
        on_wrong_stair = (on_stair_type_2 & (~type_2_is_correct)) | (on_stair_type_3 & type_2_is_correct)

        # Roll for success if on correct staircase
        rng, success_rng = jax.random.split(rng)
        success_roll = jax.random.uniform(success_rng) < params.success_prob

        # Determine outcome
        survived_correct = on_correct_stair & success_roll
        died_correct = on_correct_stair & (~success_roll)
        died_wrong = on_wrong_stair
        died = died_correct | died_wrong

        # Progress to next floor if survived correct staircase
        next_floor = jnp.where(
            survived_correct,
            state.current_floor + 1,
            state.current_floor
        ).astype(jnp.int32)

        # Check if completed all floors
        won = next_floor >= self.static_params.num_floors

        # Done if died or won or timeout
        done = died | won | (state.timestep >= params.max_timesteps)

        # Calculate reward
        # +1 for progressing to next floor, -1 for dying, +10 for winning
        reward = jnp.where(
            won,
            jnp.float32(10.0),
            jnp.where(
                survived_correct,
                jnp.float32(1.0),
                jnp.where(died, jnp.float32(-1.0), jnp.float32(0.0))
            )
        ).astype(jnp.float32)

        # If progressed to next floor, spawn agent at random position on new floor
        # Otherwise, agent moves to new_pos on current floor
        rng, spawn_rng = jax.random.split(rng)
        next_floor_index = jnp.minimum(next_floor, self.static_params.num_floors - 1)
        next_floor_grid = state.grids[next_floor_index]

        def _random_spawn(key):
            key, x_key = jax.random.split(key)
            key, y_key = jax.random.split(key)
            spawn_x = jax.random.randint(x_key, (), 0, grid_width, dtype=jnp.int32)
            spawn_y = jax.random.randint(y_key, (), 0, grid_height, dtype=jnp.int32)
            spawn_tile = next_floor_grid[spawn_y, spawn_x]
            is_stair = (spawn_tile == 2) | (spawn_tile == 3)
            spawn_x = jnp.where(is_stair, (spawn_x + 2) % grid_width, spawn_x)
            spawn_y = jnp.where(is_stair, (spawn_y + 2) % grid_height, spawn_y)
            return jnp.array([spawn_x, spawn_y], dtype=jnp.int32)

        def _corridor_spawn(key):
            inner_low = 1
            inner_high = grid_width - 1
            key, pos_key = jax.random.split(key)
            spawn_x = jax.random.randint(pos_key, (), inner_low, inner_high)
            spawn_y = jnp.int32(grid_height // 2)
            return jnp.array([spawn_x, spawn_y], dtype=jnp.int32)

        spawn_pos = (
            _corridor_spawn(spawn_rng)
            if self.static_params.place_stairs_at_ends
            else _random_spawn(spawn_rng)
        )

        # Use spawn position if progressed, otherwise use movement position
        final_pos = jnp.where(survived_correct, spawn_pos, new_pos).astype(jnp.int32)

        # Update state
        new_state = EnvState(
            grids=state.grids,  # Grids don't change, we just index differently
            agent_pos=final_pos,
            current_floor=next_floor,
            timestep=state.timestep + 1,
            done=done.astype(jnp.int32),
        )

        # Track which floors were reached/completed
        # floor_reached: binary array where floors 0 through current_floor are marked as reached
        # Only reported when episode ends (done=True)
        floor_indices = jnp.arange(self.static_params.num_floors)
        floors_reached_this_episode = (floor_indices <= state.current_floor).astype(jnp.float32)
        # Only report when episode actually ends
        floor_reached = floors_reached_this_episode * done

        # floor_completed: one-hot of current floor, only when successfully progressing
        floor_completed = jax.nn.one_hot(state.current_floor, self.static_params.num_floors) * survived_correct

        # Info dict
        info = {
            "current_floor": state.current_floor,
            "won": won,
            "died": died,
            "on_staircase": on_any_stair,
            "floor_reached": floor_reached,  # Binary array of floors reached (0 to current_floor) when episode ends
            "floor_completed": floor_completed,  # One-hot of floors successfully completed
        }

        return self.get_obs(new_state), new_state, reward, done, info

    def reset_env(
        self,
        rng: chex.PRNGKey,
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState]:
        """Reset the environment."""

        # Use the correct staircase pattern from static params
        correct_stair_pattern = self.static_params.correct_stair_pattern

        # Generate ALL floors at once using vmap
        rng, floor_rng = jax.random.split(rng)
        floor_rngs = jax.random.split(floor_rng, self.static_params.num_floors)
        floor_indices = jnp.arange(self.static_params.num_floors)

        # Use vmap to generate all floors in parallel
        grids = jax.vmap(self._generate_floor, in_axes=(0, 0, None))(
            floor_rngs, floor_indices, correct_stair_pattern
        )

        # Spawn agent at random position on floor 0
        grid_height = self.static_params.grid_height
        grid_width = self.static_params.grid_width

        def _random_spawn(rng_key):
            rng_key, agent_rng_x = jax.random.split(rng_key)
            rng_key, agent_rng_y = jax.random.split(rng_key)
            agent_x = jax.random.randint(agent_rng_x, (), 0, grid_width)
            agent_y = jax.random.randint(agent_rng_y, (), 0, grid_height)
            spawn_tile = grids[0, agent_y, agent_x]
            on_stair = (spawn_tile == 2) | (spawn_tile == 3)
            agent_x = jnp.where(on_stair, (agent_x + 2) % grid_width, agent_x)
            agent_y = jnp.where(on_stair, (agent_y + 2) % grid_height, agent_y)
            return rng_key, jnp.array([agent_x, agent_y], dtype=jnp.int32)

        def _corridor_spawn(rng_key):
            inner_low = 1
            inner_high = grid_width - 1
            rng_key, pos_key = jax.random.split(rng_key)
            agent_x = jax.random.randint(pos_key, (), inner_low, inner_high)
            agent_y = jnp.int32(grid_height // 2)
            return rng_key, jnp.array([agent_x, agent_y], dtype=jnp.int32)

        rng, spawn_key = jax.random.split(rng)
        rng, agent_pos = (
            _corridor_spawn(spawn_key)
            if self.static_params.place_stairs_at_ends
            else _random_spawn(spawn_key)
        )

        state = EnvState(
            grids=grids,
            agent_pos=agent_pos,
            current_floor=jnp.int32(0),
            timestep=jnp.int32(0),
            done=jnp.int32(0),
        )

        return self.get_obs(state), state

    def _generate_floor(
        self,
        rng: chex.PRNGKey,
        floor: int,
        correct_stair_pattern: jnp.ndarray,
    ) -> jnp.ndarray:
        """Generate a single floor with two staircases. Returns just the grid.

        Always places one staircase of type 2 and one of type 3.
        The correct_stair_pattern determines which TYPE is correct on each floor,
        not which physical position.
        """

        grid_height = self.static_params.grid_height
        grid_width = self.static_params.grid_width

        # Create empty grid
        grid = jnp.zeros((grid_height, grid_width), dtype=jnp.int32)

        if self.static_params.place_stairs_at_ends:
            rng, orient_rng = jax.random.split(rng)
            type2_left = jax.random.bernoulli(orient_rng)
            mid_row = min(grid_height // 2, grid_height - 1)
            left_x = 0
            right_x = grid_width - 1
            type2_x = jnp.where(type2_left, left_x, right_x)
            type3_x = jnp.where(type2_left, right_x, left_x)
            grid = grid.at[mid_row, type2_x].set(2)
            grid = grid.at[mid_row, type3_x].set(3)
            return grid

        # Randomly place two staircases when not using the corridor layout
        rng, stair1_rng, stair2_rng = jax.random.split(rng, 3)
        stair1_rng, stair1_x_key, stair1_y_key = jax.random.split(stair1_rng, 3)
        stair2_rng, stair2_x_key, stair2_y_key = jax.random.split(stair2_rng, 3)

        # First staircase position (type 2)
        stair1_x = jax.random.randint(stair1_x_key, (), 0, grid_width)
        stair1_y = jax.random.randint(stair1_y_key, (), 0, grid_height)

        # Second staircase position (type 3) - ensure different from first
        stair2_x = jax.random.randint(stair2_x_key, (), 0, grid_width)
        stair2_y = jax.random.randint(stair2_y_key, (), 0, grid_height)

        # If same position, shift second staircase
        same_pos = (stair1_x == stair2_x) & (stair1_y == stair2_y)
        stair2_x = jnp.where(same_pos, (stair2_x + 1) % grid_width, stair2_x)
        stair2_y = jnp.where(same_pos, (stair2_y + 1) % grid_height, stair2_y)

        # Always place type 2 at first position, type 3 at second position
        grid = grid.at[stair1_y, stair1_x].set(2)  # Type 2 staircase
        grid = grid.at[stair2_y, stair2_x].set(3)  # Type 3 staircase

        return grid

    def get_obs(self, state: EnvState) -> chex.Array:
        """Get observation from state."""
        # Get current floor grid
        current_grid = state.grids[state.current_floor]

        # Add agent to grid for observation
        obs_grid = current_grid.at[state.agent_pos[1], state.agent_pos[0]].set(1)

        # Flatten grid and concatenate with floor number
        flat_grid = obs_grid.flatten()
        floor_one_hot = jax.nn.one_hot(state.current_floor, self.static_params.num_floors)
        obs = jnp.concatenate([flat_grid, floor_one_hot])
        return obs.astype(jnp.float32)

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check if episode is done."""
        return state.done

    @property
    def name(self) -> str:
        return "StaircaseDungeon-v1"

    @property
    def num_actions(self) -> int:
        return 5  # UP, DOWN, LEFT, RIGHT, NOOP

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        return spaces.Discrete(5)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        obs_size = (
            self.static_params.grid_height * self.static_params.grid_width
            + self.static_params.num_floors
        )
        return spaces.Box(
            0.0,
            3.0,
            (obs_size,),
            dtype=jnp.float32,
        )

    def discount(self, state: EnvState, params: EnvParams) -> float:
        """Discount factor."""
        return 1.0 - state.done.astype(jnp.float32)
