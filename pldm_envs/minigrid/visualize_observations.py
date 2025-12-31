#!/usr/bin/env python3
"""
Visualization script for agent observations in LongHorizon Level 1-3 environments.

This script creates visualizations showing the RGB observations (top-down view)
for different episodes across all three levels.
"""

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import pldm_envs.minigrid


def visualize_observation_episode(env, level_name, episode_num, ax=None):
    """Visualize a single episode's observation (RGB image)."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    unwrapped_env = env.unwrapped

    # Reset environment
    obs, info = env.reset()

    # Render the environment to get RGB image
    rgb_img = env.render()

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
    else:
        manhattan_dist = 0

    # Display the RGB observation
    ax.imshow(rgb_img)
    ax.set_title(f'{level_name} - Episode {episode_num}\n'
                f'Agent: {agent_pos}, Goal: {goal_pos}, Distance: {manhattan_dist}',
                fontsize=11, fontweight='bold')
    ax.axis('off')

    return ax


def visualize_multiple_observations(env_id, level_name, num_episodes=6):
    """Visualize multiple episodes' observations."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Create environment with RGB rendering
    env = gym.make(env_id, render_mode='rgb_array')

    for i in range(num_episodes):
        visualize_observation_episode(env, level_name, i + 1, ax=axes[i])

    plt.tight_layout()
    env.close()
    return fig


def visualize_all_level_observations():
    """Create observation visualizations for all three levels."""
    import os
    os.makedirs('pldm_envs/minigrid/visualizations', exist_ok=True)

    levels = [
        ("MiniGrid-LongHorizon-Level1-v0", "Level 1 (Simple)"),
        ("MiniGrid-LongHorizon-Level2-v0", "Level 2 (Medium)"),
        ("MiniGrid-LongHorizon-Level3-v0", "Level 3 (Complex)"),
    ]

    print("\n" + "="*60)
    print("MiniGrid Observation Visualization")
    print("="*60)

    for env_id, level_name in levels:
        print(f"\nCreating observation visualizations for {level_name}...")
        fig = visualize_multiple_observations(env_id, level_name, num_episodes=6)

        filename = f"pldm_envs/minigrid/visualizations/{env_id.replace('MiniGrid-', '').replace('-v0', '')}_observations.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close(fig)

    print("\n" + "="*60)
    print("All observation visualizations completed!")
    print("="*60)
    print("\nGenerated files:")
    print("  1. visualizations/LongHorizon-Level1_observations.png")
    print("  2. visualizations/LongHorizon-Level2_observations.png")
    print("  3. visualizations/LongHorizon-Level3_observations.png")
    print("="*60 + "\n")


def create_observation_comparison():
    """Create a side-by-side comparison of observations across levels."""
    import os
    os.makedirs('pldm_envs/minigrid/visualizations', exist_ok=True)

    levels = [
        ("MiniGrid-LongHorizon-Level1-v0", "Level 1"),
        ("MiniGrid-LongHorizon-Level2-v0", "Level 2"),
        ("MiniGrid-LongHorizon-Level3-v0", "Level 3"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (env_id, level_name) in enumerate(levels):
        env = gym.make(env_id, render_mode='rgb_array')
        env.reset()
        rgb_img = env.render()

        axes[idx].imshow(rgb_img)
        axes[idx].set_title(level_name, fontsize=14, fontweight='bold')
        axes[idx].axis('off')

        env.close()

    plt.suptitle('Observation Comparison Across Levels', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('pldm_envs/minigrid/visualizations/observation_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: visualizations/observation_comparison.png")
    plt.close()


def main():
    """Main function."""
    visualize_all_level_observations()
    create_observation_comparison()


if __name__ == "__main__":
    main()
