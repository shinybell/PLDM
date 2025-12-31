#!/usr/bin/env python3
"""
Visualization script for LongHorizon Level 1-3 environments.

This script creates visualizations showing:
1. The maze structure for each level
2. Multiple episodes with different random start/goal positions
3. Agent trajectories (if running episodes)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle
import gymnasium as gym
import pldm_envs.minigrid


def visualize_maze_structure(env, level_name, ax=None):
    """Visualize the maze structure without agent/goal."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    unwrapped_env = env.unwrapped

    # Create a grid representation
    grid_array = np.zeros((unwrapped_env.height, unwrapped_env.width))

    for y in range(unwrapped_env.height):
        for x in range(unwrapped_env.width):
            cell = unwrapped_env.grid.get(x, y)
            if cell is not None and cell.type == 'wall':
                grid_array[y, x] = 1

    # Display the grid
    ax.imshow(grid_array, cmap='Greys', origin='upper')
    ax.set_title(f'{level_name} Maze Structure', fontsize=14, fontweight='bold')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.grid(True, alpha=0.3)

    return ax


def visualize_episode(env, level_name, episode_num, ax=None):
    """Visualize a single episode with agent and goal positions."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    unwrapped_env = env.unwrapped
    obs, info = env.reset()

    # Create a grid representation
    grid_array = np.zeros((unwrapped_env.height, unwrapped_env.width))

    for y in range(unwrapped_env.height):
        for x in range(unwrapped_env.width):
            cell = unwrapped_env.grid.get(x, y)
            if cell is not None and cell.type == 'wall':
                grid_array[y, x] = 1

    # Display the grid
    ax.imshow(grid_array, cmap='Greys', origin='upper', alpha=0.5)

    # Get agent and goal positions
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

    # Calculate Manhattan distance
    if goal_pos:
        manhattan_dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])

        # Plot agent (red circle)
        agent_circle = Circle((agent_pos[0], agent_pos[1]), 0.4,
                             color='red', alpha=0.8, zorder=10)
        ax.add_patch(agent_circle)

        # Plot goal (green square)
        goal_rect = Rectangle((goal_pos[0] - 0.4, goal_pos[1] - 0.4), 0.8, 0.8,
                              color='green', alpha=0.8, zorder=10)
        ax.add_patch(goal_rect)

        # Draw line connecting agent and goal
        ax.plot([agent_pos[0], goal_pos[0]], [agent_pos[1], goal_pos[1]],
               'b--', alpha=0.5, linewidth=2, zorder=5)

        ax.set_title(f'{level_name} - Episode {episode_num}\n'
                    f'Agent: {agent_pos}, Goal: {goal_pos}, Distance: {manhattan_dist}',
                    fontsize=12, fontweight='bold')
    else:
        ax.set_title(f'{level_name} - Episode {episode_num}',
                    fontsize=12, fontweight='bold')

    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.grid(True, alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.8, label='Agent Start'),
        Patch(facecolor='green', alpha=0.8, label='Goal'),
        Patch(facecolor='gray', alpha=0.5, label='Wall')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    return ax


def visualize_multiple_episodes(env_id, level_name, num_episodes=6):
    """Visualize multiple episodes to show position variation."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    env = gym.make(env_id)

    for i in range(num_episodes):
        visualize_episode(env, level_name, i + 1, ax=axes[i])

    plt.tight_layout()
    return fig


def visualize_all_levels():
    """Create comprehensive visualization for all three levels."""
    levels = [
        ("MiniGrid-LongHorizon-Level1-v0", "Level 1 (Simple)"),
        ("MiniGrid-LongHorizon-Level2-v0", "Level 2 (Medium)"),
        ("MiniGrid-LongHorizon-Level3-v0", "Level 3 (Complex)"),
    ]

    # 1. Maze structures
    print("Creating maze structure visualizations...")
    fig_structures, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (env_id, level_name) in enumerate(levels):
        env = gym.make(env_id)
        visualize_maze_structure(env, level_name, ax=axes[idx])
        env.close()

    plt.suptitle('Maze Structures for All Levels', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('pldm_envs/minigrid/visualizations/maze_structures.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: visualizations/maze_structures.png")

    # 2. Multiple episodes for each level
    for env_id, level_name in levels:
        print(f"\nCreating episode visualizations for {level_name}...")
        fig = visualize_multiple_episodes(env_id, level_name, num_episodes=6)

        filename = f"pldm_envs/minigrid/visualizations/{env_id.replace('MiniGrid-', '').replace('-v0', '')}_episodes.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close(fig)

    plt.close('all')


def create_distance_statistics_plot():
    """Create a plot showing distance statistics for each level."""
    levels = [
        ("MiniGrid-LongHorizon-Level1-v0", "Level 1", 10),
        ("MiniGrid-LongHorizon-Level2-v0", "Level 2", 15),
        ("MiniGrid-LongHorizon-Level3-v0", "Level 3", 20),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (env_id, level_name, min_dist) in enumerate(levels):
        env = gym.make(env_id)
        unwrapped_env = env.unwrapped

        # Collect distance data
        distances = []
        for _ in range(100):
            env.reset()
            agent_pos = unwrapped_env.agent_pos

            # Find goal
            goal_pos = None
            for x in range(unwrapped_env.width):
                for y in range(unwrapped_env.height):
                    cell = unwrapped_env.grid.get(x, y)
                    if cell is not None and cell.type == 'goal':
                        goal_pos = (x, y)
                        break
                if goal_pos:
                    break

            if goal_pos:
                dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
                distances.append(dist)

        # Plot histogram
        axes[idx].hist(distances, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
        axes[idx].axvline(min_dist, color='red', linestyle='--', linewidth=2,
                         label=f'Min Distance = {min_dist}')
        axes[idx].axvline(np.mean(distances), color='green', linestyle='--', linewidth=2,
                         label=f'Mean = {np.mean(distances):.1f}')

        axes[idx].set_xlabel('Manhattan Distance', fontsize=12)
        axes[idx].set_ylabel('Frequency', fontsize=12)
        axes[idx].set_title(f'{level_name}\n(Min: {np.min(distances)}, Max: {np.max(distances)}, '
                           f'Mean: {np.mean(distances):.1f})', fontsize=12, fontweight='bold')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

        env.close()

    plt.suptitle('Distance Distribution (100 episodes per level)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('pldm_envs/minigrid/visualizations/distance_statistics.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: visualizations/distance_statistics.png")
    plt.close()


def main():
    """Main visualization function."""
    import os

    # Create output directory
    os.makedirs('pldm_envs/minigrid/visualizations', exist_ok=True)

    print("\n" + "="*60)
    print("MiniGrid Level 1-3 Visualization")
    print("="*60)

    # Create all visualizations
    visualize_all_levels()
    create_distance_statistics_plot()

    print("\n" + "="*60)
    print("All visualizations completed!")
    print("="*60)
    print("\nGenerated files:")
    print("  1. visualizations/maze_structures.png")
    print("  2. visualizations/LongHorizon-Level1_episodes.png")
    print("  3. visualizations/LongHorizon-Level2_episodes.png")
    print("  4. visualizations/LongHorizon-Level3_episodes.png")
    print("  5. visualizations/distance_statistics.png")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
