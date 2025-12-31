"""
Long-horizon MiniGrid environments with increasing wall complexity.

These environments are designed for testing long-horizon prediction capabilities
by using custom map arrays with varying wall complexity.

All environments:
- Grid size: Determined by the map array
- Max steps: Configurable (default: 256)
- Random agent and goal positioning with distance constraints

References:
    - MiniGrid Tutorial: https://minigrid.farama.org/content/create_env_tutorial/
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall
from minigrid.minigrid_env import MiniGridEnv


class PositionSampler:
    """
    Utility class for sampling agent and goal positions in MiniGrid environments.

    Inspired by the position sampling logic in wall and antmaze environments,
    this sampler:
    - Pre-computes valid (walkable) grid positions
    - Randomly samples positions with distance constraints
    - Ensures both positions are reachable
    """

    def __init__(self, grid: Grid, width: int, height: int):
        """
        Initialize the position sampler.

        Args:
            grid: The MiniGrid Grid object
            width: Grid width
            height: Grid height
        """
        self.grid = grid
        self.width = width
        self.height = height
        self.valid_positions = self._compute_valid_positions()

    def _compute_valid_positions(self) -> List[Tuple[int, int]]:
        """
        Pre-compute all valid (walkable) positions in the grid.
        Similar to antmaze's collision checking approach.

        Returns:
            List of (x, y) tuples representing valid positions
        """
        valid_pos = []
        for x in range(1, self.width - 1):  # Exclude outer walls
            for y in range(1, self.height - 1):
                cell = self.grid.get(x, y)
                # Position is valid if it's empty (None)
                if cell is None:
                    valid_pos.append((x, y))
        return valid_pos

    def sample_positions(
        self,
        min_distance: Optional[int] = None,
        max_distance: Optional[int] = None,
        max_attempts: int = 1000,
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Sample start and goal positions with distance constraints.

        Similar to wall environment's approach but adapted for discrete grids.
        Uses Manhattan distance for grid-based environments.

        Args:
            min_distance: Minimum Manhattan distance between start and goal
            max_distance: Maximum Manhattan distance between start and goal
            max_attempts: Maximum number of sampling attempts

        Returns:
            Tuple of (start_pos, goal_pos) where each is (x, y)
        """
        if len(self.valid_positions) < 2:
            raise ValueError("Not enough valid positions to sample from")

        for _ in range(max_attempts):
            # Randomly sample two different positions
            indices = np.random.choice(
                len(self.valid_positions), size=2, replace=False
            )
            start_pos = self.valid_positions[indices[0]]
            goal_pos = self.valid_positions[indices[1]]

            # Calculate Manhattan distance
            manhattan_dist = abs(start_pos[0] - goal_pos[0]) + \
                           abs(start_pos[1] - goal_pos[1])

            # Check distance constraints
            if min_distance is not None and manhattan_dist < min_distance:
                continue
            if max_distance is not None and manhattan_dist > max_distance:
                continue

            return start_pos, goal_pos

        # Fallback: return random positions without constraints
        indices = np.random.choice(
            len(self.valid_positions), size=2, replace=False
        )
        return self.valid_positions[indices[0]], self.valid_positions[indices[1]]


class CustomMapEnv(MiniGridEnv):
    """
    Custom map environment that creates a maze from a 2D array.

    - Grid size: Determined by the map array
    - Wall pattern: Defined by the input map (1=wall, 0=passable)
    - Max steps: Configurable (default: 256)
    - Random agent and goal positioning inspired by wall/antmaze environments

    Example usage:
        map_array = [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1]
        ]
        env = gym.make('MiniGrid-CustomMap-v0',
                       map_array=map_array,
                       randomize_start_goal=True,
                       min_distance=5)
    """

    def __init__(
        self,
        map_array: Optional[List[List[int]]] = None,
        agent_start_pos: Optional[tuple] = None,
        agent_start_dir: int = 0,
        goal_pos: Optional[tuple] = None,
        max_steps: int = 256,
        randomize_start_goal: bool = False,
        min_distance: Optional[int] = None,
        max_distance: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize Custom Map environment.

        Args:
            map_array: 2D array where 1=wall, 0=passable.
                      If None, creates a simple 5x5 maze.
            agent_start_pos: Starting position of the agent (default: auto-placed)
                           Ignored if randomize_start_goal=True
            agent_start_dir: Starting direction of the agent (default: 0, facing right)
            goal_pos: Position of the goal (default: bottom-right corner)
                     Ignored if randomize_start_goal=True
            max_steps: Maximum steps per episode (default: 256)
            randomize_start_goal: If True, randomly sample start and goal positions
                                 each reset (inspired by wall/antmaze)
            min_distance: Minimum Manhattan distance between start and goal
                         (only used if randomize_start_goal=True)
            max_distance: Maximum Manhattan distance between start and goal
                         (only used if randomize_start_goal=True)
            **kwargs: Additional arguments passed to MiniGridEnv
        """
        # Use default map if none provided
        if map_array is None:
            map_array = [
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1]
            ]

        # Convert to numpy array and validate
        self.map_array = np.array(map_array, dtype=int)
        if len(self.map_array.shape) != 2:
            raise ValueError("map_array must be a 2D array")

        # Check that all outer cells are walls
        height, width = self.map_array.shape
        if not (np.all(self.map_array[0, :] == 1) and
                np.all(self.map_array[-1, :] == 1) and
                np.all(self.map_array[:, 0] == 1) and
                np.all(self.map_array[:, -1] == 1)):
            raise ValueError("All outer cells of map_array must be walls (1)")

        # Store configuration
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.goal_pos = goal_pos
        self.randomize_start_goal = randomize_start_goal
        self.min_distance = min_distance
        self.max_distance = max_distance

        # Determine grid size from map
        self.height = height
        self.width = width
        self.size = max(width, height)  # MiniGrid uses square grids

        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,
            max_steps=max_steps,
            see_through_walls=False,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "navigate to the green goal square through the custom maze"

    def _gen_grid(self, width, height):
        """Generate the grid layout from the map array."""
        # Create an empty grid
        self.grid = Grid(width, height)

        # Place walls according to the map array
        for y in range(height):
            for x in range(width):
                if self.map_array[y, x] == 1:
                    self.grid.set(x, y, Wall())

        # Handle position sampling
        if self.randomize_start_goal:
            # Use PositionSampler to randomly place agent and goal
            sampler = PositionSampler(self.grid, width, height)
            start_pos, goal_pos = sampler.sample_positions(
                min_distance=self.min_distance,
                max_distance=self.max_distance,
            )

            # Place goal
            self.put_obj(Goal(), *goal_pos)

            # Place agent
            self.agent_pos = start_pos
            self.agent_dir = self.agent_start_dir
        else:
            # Use fixed positions (original behavior)
            # Place the goal
            if self.goal_pos is not None:
                goal_x, goal_y = self.goal_pos
                if self.map_array[goal_y, goal_x] == 1:
                    raise ValueError(f"Goal position {self.goal_pos} is a wall in the map")
                self.put_obj(Goal(), goal_x, goal_y)
            else:
                # Default: place goal at bottom-right passable cell
                for y in range(height - 2, 0, -1):
                    for x in range(width - 2, 0, -1):
                        if self.map_array[y, x] == 0:
                            self.put_obj(Goal(), x, y)
                            break
                    else:
                        continue
                    break

            # Place the agent
            if self.agent_start_pos is not None:
                start_x, start_y = self.agent_start_pos
                if self.map_array[start_y, start_x] == 1:
                    raise ValueError(f"Agent start position {self.agent_start_pos} is a wall in the map")
                self.agent_pos = self.agent_start_pos
                self.agent_dir = self.agent_start_dir
            else:
                # Default: place agent at top-left passable cell
                for y in range(1, height - 1):
                    for x in range(1, width - 1):
                        if self.map_array[y, x] == 0:
                            self.agent_pos = (x, y)
                            self.agent_dir = self.agent_start_dir
                            break
                    else:
                        continue
                    break
