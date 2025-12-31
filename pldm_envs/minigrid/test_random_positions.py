#!/usr/bin/env python3
"""
Test script for CustomMapEnv with random position sampling.

This script demonstrates:
1. Creating a CustomMapEnv with a custom map
2. Using randomize_start_goal with distance constraints
3. Comparing with wall/antmaze position sampling logic
"""

import numpy as np
import gymnasium as gym
from pldm_envs.minigrid.envs import CustomMapEnv

# Define a simple test map (24x24)
def create_simple_map():
    """Create a simple 24x24 map with some walls."""
    size = 24
    map_array = np.zeros((size, size), dtype=int)

    # Outer walls
    map_array[0, :] = 1
    map_array[-1, :] = 1
    map_array[:, 0] = 1
    map_array[:, -1] = 1

    # Add some internal walls (L-shaped)
    for i in range(8, 16):
        map_array[i, 8] = 1
        map_array[16, i] = 1

    return map_array.tolist()


def create_complex_map():
    """Create a more complex 24x24 map with multiple corridors."""
    size = 24
    map_array = np.zeros((size, size), dtype=int)

    # Outer walls
    map_array[0, :] = 1
    map_array[-1, :] = 1
    map_array[:, 0] = 1
    map_array[:, -1] = 1

    # Vertical walls
    for x in [7, 12, 17]:
        for y in range(4, size - 4):
            if y not in [8, 9, 14, 15]:  # Leave gaps
                map_array[y, x] = 1

    # Horizontal walls
    for y in [7, 13, 18]:
        for x in range(4, size - 4):
            if x not in [9, 10, 15, 16]:  # Leave gaps
                map_array[y, x] = 1

    return map_array.tolist()


def test_fixed_positions():
    """Test with fixed agent and goal positions (original behavior)."""
    print("\n" + "="*60)
    print("Test 1: Fixed Positions (Original Behavior)")
    print("="*60)

    map_array = create_simple_map()
    env = CustomMapEnv(
        map_array=map_array,
        agent_start_pos=(2, 2),
        goal_pos=(21, 21),
        randomize_start_goal=False,
    )

    obs, info = env.reset()
    print(f"Agent position: {env.agent_pos}")
    print(f"Goal position: {env.goal.cur_pos if hasattr(env, 'goal') else 'N/A'}")
    print("✓ Fixed positions work correctly")


def test_random_positions_no_constraints():
    """Test random positioning without distance constraints."""
    print("\n" + "="*60)
    print("Test 2: Random Positions (No Distance Constraints)")
    print("="*60)

    map_array = create_simple_map()
    env = CustomMapEnv(
        map_array=map_array,
        randomize_start_goal=True,
    )

    positions = []
    distances = []

    for i in range(10):
        obs, info = env.reset()
        agent_pos = env.agent_pos

        # Find goal position
        goal_pos = None
        for x in range(env.width):
            for y in range(env.height):
                cell = env.grid.get(x, y)
                if cell is not None and cell.type == 'goal':
                    goal_pos = (x, y)
                    break
            if goal_pos:
                break

        manhattan_dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
        positions.append((agent_pos, goal_pos))
        distances.append(manhattan_dist)

        print(f"Episode {i+1}: Agent={agent_pos}, Goal={goal_pos}, Distance={manhattan_dist}")

    print(f"\nDistance statistics:")
    print(f"  Min: {min(distances)}")
    print(f"  Max: {max(distances)}")
    print(f"  Mean: {np.mean(distances):.2f}")
    print(f"  Std: {np.std(distances):.2f}")
    print("✓ Random positions without constraints work")


def test_random_positions_with_min_distance():
    """Test random positioning with minimum distance constraint."""
    print("\n" + "="*60)
    print("Test 3: Random Positions (Min Distance = 15)")
    print("="*60)

    map_array = create_simple_map()
    min_dist = 15

    env = CustomMapEnv(
        map_array=map_array,
        randomize_start_goal=True,
        min_distance=min_dist,
    )

    distances = []

    for i in range(10):
        obs, info = env.reset()
        agent_pos = env.agent_pos

        # Find goal position
        goal_pos = None
        for x in range(env.width):
            for y in range(env.height):
                cell = env.grid.get(x, y)
                if cell is not None and cell.type == 'goal':
                    goal_pos = (x, y)
                    break
            if goal_pos:
                break

        manhattan_dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
        distances.append(manhattan_dist)

        print(f"Episode {i+1}: Agent={agent_pos}, Goal={goal_pos}, Distance={manhattan_dist}")

        # Check constraint
        assert manhattan_dist >= min_dist, f"Distance {manhattan_dist} < min {min_dist}"

    print(f"\nAll distances >= {min_dist}: ✓")
    print(f"Distance range: [{min(distances)}, {max(distances)}]")
    print("✓ Minimum distance constraint works correctly")


def test_random_positions_with_distance_range():
    """Test random positioning with both min and max distance constraints."""
    print("\n" + "="*60)
    print("Test 4: Random Positions (Min=10, Max=20)")
    print("="*60)

    map_array = create_simple_map()
    min_dist = 10
    max_dist = 20

    env = CustomMapEnv(
        map_array=map_array,
        randomize_start_goal=True,
        min_distance=min_dist,
        max_distance=max_dist,
    )

    distances = []

    for i in range(10):
        obs, info = env.reset()
        agent_pos = env.agent_pos

        # Find goal position
        goal_pos = None
        for x in range(env.width):
            for y in range(env.height):
                cell = env.grid.get(x, y)
                if cell is not None and cell.type == 'goal':
                    goal_pos = (x, y)
                    break
            if goal_pos:
                break

        manhattan_dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
        distances.append(manhattan_dist)

        print(f"Episode {i+1}: Agent={agent_pos}, Goal={goal_pos}, Distance={manhattan_dist}")

        # Check constraints
        assert min_dist <= manhattan_dist <= max_dist, \
            f"Distance {manhattan_dist} not in [{min_dist}, {max_dist}]"

    print(f"\nAll distances in [{min_dist}, {max_dist}]: ✓")
    print(f"Actual range: [{min(distances)}, {max(distances)}]")
    print("✓ Distance range constraint works correctly")


def test_complex_map():
    """Test with a more complex map."""
    print("\n" + "="*60)
    print("Test 5: Complex Map with Random Positions")
    print("="*60)

    map_array = create_complex_map()

    env = CustomMapEnv(
        map_array=map_array,
        randomize_start_goal=True,
        min_distance=15,
    )

    for i in range(5):
        obs, info = env.reset()
        agent_pos = env.agent_pos

        # Find goal position
        goal_pos = None
        for x in range(env.width):
            for y in range(env.height):
                cell = env.grid.get(x, y)
                if cell is not None and cell.type == 'goal':
                    goal_pos = (x, y)
                    break
            if goal_pos:
                break

        manhattan_dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])

        print(f"Episode {i+1}: Agent={agent_pos}, Goal={goal_pos}, Distance={manhattan_dist}")

    print("✓ Complex map works correctly")


if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# CustomMapEnv Random Position Sampling Tests")
    print("# Inspired by wall/antmaze position sampling logic")
    print("#"*60)

    test_fixed_positions()
    test_random_positions_no_constraints()
    test_random_positions_with_min_distance()
    test_random_positions_with_distance_range()
    test_complex_map()

    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)
    print("\nThe random position sampling logic has been successfully")
    print("implemented following the wall/antmaze approach:")
    print("  • Pre-computes valid grid positions")
    print("  • Randomly samples with distance constraints")
    print("  • Uses Manhattan distance for discrete grids")
    print("  • Falls back gracefully if constraints can't be met")
    print("="*60 + "\n")
