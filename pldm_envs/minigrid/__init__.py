"""MiniGrid environment package for PLDM."""

from gymnasium.envs.registration import register
from functools import partial
from pldm_envs.minigrid.configs.mazes import l1maze, l2maze, l3maze
from pldm_envs.minigrid.envs import CustomMapEnv

# Register custom map environment (generic)
register(
    id="MiniGrid-CustomMap-v0",
    entry_point="pldm_envs.minigrid.envs:CustomMapEnv",
)

# Register long-horizon environments with predefined mazes
# Level 1: Simple maze with 3 horizontal walls
register(
    id="MiniGrid-LongHorizon-Level1-v0",
    entry_point="pldm_envs.minigrid.envs:CustomMapEnv",
    kwargs={
        "map_array": l1maze,
        "randomize_start_goal": True,
        "min_distance": 10,
        "highlight": False,  # Disable agent view highlight
    },
)

# Level 2: Medium complexity maze with scattered obstacles
register(
    id="MiniGrid-LongHorizon-Level2-v0",
    entry_point="pldm_envs.minigrid.envs:CustomMapEnv",
    kwargs={
        "map_array": l2maze,
        "randomize_start_goal": True,
        "min_distance": 15,
        "highlight": False,  # Disable agent view highlight
    },
)

# Level 3: Complex maze with dense wall structure
register(
    id="MiniGrid-LongHorizon-Level3-v0",
    entry_point="pldm_envs.minigrid.envs:CustomMapEnv",
    kwargs={
        "map_array": l3maze,
        "randomize_start_goal": True,
        "min_distance": 20,
        "highlight": False,  # Disable agent view highlight
    },
)
