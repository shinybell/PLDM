#!/usr/bin/env python3
"""
Test script for LongHorizon Level 1-3 environments.

This script tests that the Level 1-3 environments are correctly registered
and use the maze configurations from mazes.py with random positioning.
"""

import numpy as np
import gymnasium as gym

# Import to trigger environment registration
import pldm_envs.minigrid


def test_level_env(env_id, level_name, expected_min_distance):
    """Test a single level environment."""
    print(f"\n{'='*60}")
    print(f"Testing {level_name}: {env_id}")
    print(f"{'='*60}")

    # Create environment
    env = gym.make(env_id)
    unwrapped_env = env.unwrapped

    print(f"Grid size: {unwrapped_env.width}x{unwrapped_env.height}")
    print(f"Max steps: {env.spec.max_episode_steps if env.spec else 'N/A'}")
    print(f"Expected min distance: {expected_min_distance}")

    # Test multiple episodes
    distances = []
    positions = []

    for i in range(10):
        obs, info = env.reset()
        agent_pos = unwrapped_env.agent_pos

        # Find goal position
        goal_pos = None
        for x in range(unwrapped_env.width):
            for y in range(unwrapped_env.height):
                cell = unwrapped_env.grid.get(x, y)
                if cell is not None and cell.type == 'goal':
                    goal_pos = (x, y)
                    break
            if goal_pos:
                break

        manhattan_dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
        distances.append(manhattan_dist)
        positions.append((agent_pos, goal_pos))

        if i < 5:  # Print first 5 episodes
            print(f"  Episode {i+1}: Agent={agent_pos}, Goal={goal_pos}, Distance={manhattan_dist}")

    print(f"\nDistance statistics (10 episodes):")
    print(f"  Min: {min(distances)}")
    print(f"  Max: {max(distances)}")
    print(f"  Mean: {np.mean(distances):.2f}")
    print(f"  Std: {np.std(distances):.2f}")

    # Verify all distances meet minimum requirement
    all_valid = all(d >= expected_min_distance for d in distances)
    if all_valid:
        print(f"✓ All distances >= {expected_min_distance}")
    else:
        print(f"✗ Some distances < {expected_min_distance}")
        for i, d in enumerate(distances):
            if d < expected_min_distance:
                print(f"  Episode {i+1}: {d} < {expected_min_distance}")

    env.close()
    return all_valid


def main():
    print("\n" + "#"*60)
    print("# LongHorizon Level 1-3 Environment Tests")
    print("# Using predefined mazes from configs/mazes.py")
    print("#"*60)

    results = []

    # Test Level 1
    results.append(test_level_env(
        "MiniGrid-LongHorizon-Level1-v0",
        "Level 1 (Simple)",
        expected_min_distance=10
    ))

    # Test Level 2
    results.append(test_level_env(
        "MiniGrid-LongHorizon-Level2-v0",
        "Level 2 (Medium)",
        expected_min_distance=15
    ))

    # Test Level 3
    results.append(test_level_env(
        "MiniGrid-LongHorizon-Level3-v0",
        "Level 3 (Complex)",
        expected_min_distance=20
    ))

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Level 1: {'✓ PASS' if results[0] else '✗ FAIL'}")
    print(f"Level 2: {'✓ PASS' if results[1] else '✗ FAIL'}")
    print(f"Level 3: {'✓ PASS' if results[2] else '✗ FAIL'}")

    if all(results):
        print("\n✓ All level environments working correctly!")
    else:
        print("\n✗ Some level environments failed")

    print("="*60 + "\n")


if __name__ == "__main__":
    main()
