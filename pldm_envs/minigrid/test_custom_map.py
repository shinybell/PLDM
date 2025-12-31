"""
Test script for CustomMapEnv.

This script demonstrates how to create custom mazes using 2D arrays.
"""

import sys
import gymnasium as gym
import numpy as np
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper

# Register custom environments
import pldm_envs.minigrid


def test_custom_map(map_array, description="Custom Map", n_episodes=3):
    """
    Test a custom map environment.

    Args:
        map_array: 2D array where 1=wall, 0=passable
        description: Description of the map
        n_episodes: Number of episodes to run
    """
    print("=" * 80)
    print(f"Testing: {description}")
    print("=" * 80)

    # Print the map
    print("\nMap Layout (1=wall, 0=passable):")
    for row in map_array:
        print("  " + " ".join(str(x) for x in row))

    # Create environment
    env = gym.make('MiniGrid-CustomMap-v0', map_array=map_array)
    env = RGBImgObsWrapper(env, tile_size=8)
    env = ImgObsWrapper(env)

    print(f"\nEnvironment created:")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Grid size: {env.unwrapped.width}x{env.unwrapped.height}")
    print(f"  Max steps: {env.unwrapped.max_steps}")

    episode_lengths = []
    episode_rewards = []

    for ep_idx in range(n_episodes):
        obs, info = env.reset(seed=42 + ep_idx)
        done = False
        step_count = 0
        total_reward = 0.0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1
            total_reward += reward

        episode_lengths.append(step_count)
        episode_rewards.append(total_reward)

        print(f"\n  Episode {ep_idx + 1}: {step_count} steps, reward={total_reward:.4f}")

    env.close()

    # Statistics
    print("\n" + "-" * 80)
    print("Statistics:")
    print(f"  Average episode length: {np.mean(episode_lengths):.1f} steps")
    print(f"  Average reward: {np.mean(episode_rewards):.4f}")
    print("-" * 80 + "\n")


def main():
    """Test various custom maps."""
    print("\n" + "=" * 80)
    print("CUSTOM MAP ENVIRONMENT TEST")
    print("=" * 80 + "\n")

    # Test 1: Simple corridor (your example)
    map1 = [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]
    ]
    test_custom_map(map1, "Simple Corridor (5x5)", n_episodes=3)

    # Test 2: Larger maze
    map2 = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 1, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 0, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 1, 1, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]
    test_custom_map(map2, "Medium Maze (10x10)", n_episodes=3)

    # Test 3: Room with central obstacle
    map3 = [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1]
    ]
    test_custom_map(map3, "Room with Central Obstacle (7x7)", n_episodes=3)

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED!")
    print("=" * 80 + "\n")

    print("Usage example:")
    print("-" * 80)
    print("""
import gymnasium as gym
import pldm_envs.minigrid

# Define your custom map
my_map = [
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1]
]

# Create the environment
env = gym.make('MiniGrid-CustomMap-v0', map_array=my_map)

# Or with custom agent/goal positions
env = gym.make(
    'MiniGrid-CustomMap-v0',
    map_array=my_map,
    agent_start_pos=(1, 1),  # (x, y)
    goal_pos=(3, 3),         # (x, y)
    max_steps=100
)
    """)
    print("-" * 80 + "\n")


if __name__ == "__main__":
    main()
