"""
Test script for custom long-horizon MiniGrid environments.

This script tests all three custom environments to ensure they work correctly.
"""

import sys
import gymnasium as gym
import numpy as np
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper

# Register custom environments
import pldm_envs.minigrid


def test_environment(env_name: str, n_episodes: int = 3, verbose: bool = True):
    """
    Test a MiniGrid environment.

    Args:
        env_name: Name of the environment to test
        n_episodes: Number of episodes to run
        verbose: Whether to print detailed information
    """
    print("=" * 80)
    print(f"Testing: {env_name}")
    print("=" * 80)

    # Create environment
    env = gym.make(env_name, render_mode=None)
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

        if verbose:
            print(f"\nEpisode {ep_idx + 1}:")
            print(f"  Length: {step_count} steps")
            print(f"  Total reward: {total_reward:.4f}")
            print(f"  Terminated: {terminated}")
            print(f"  Observation shape: {obs.shape}")

    env.close()

    # Statistics
    print("\n" + "-" * 80)
    print("Statistics:")
    print("-" * 80)
    print(f"  Episodes: {n_episodes}")
    print(f"  Episode lengths:")
    print(f"    Mean: {np.mean(episode_lengths):.1f} steps")
    print(f"    Std:  {np.std(episode_lengths):.1f} steps")
    print(f"    Min:  {np.min(episode_lengths)} steps")
    print(f"    Max:  {np.max(episode_lengths)} steps")
    print(f"  Episode rewards:")
    print(f"    Mean: {np.mean(episode_rewards):.4f}")
    print(f"    Std:  {np.std(episode_rewards):.4f}")
    print(f"    Min:  {np.min(episode_rewards):.4f}")
    print(f"    Max:  {np.max(episode_rewards):.4f}")

    return {
        "env_name": env_name,
        "episode_lengths": episode_lengths,
        "episode_rewards": episode_rewards,
    }


def main():
    """Test all custom long-horizon environments."""
    print("\n" + "=" * 80)
    print("CUSTOM MINIGRID ENVIRONMENTS TEST")
    print("=" * 80 + "\n")

    environments = [
        "MiniGrid-LongHorizon-Level1-v0",
        "MiniGrid-LongHorizon-Level2-v0",
        "MiniGrid-LongHorizon-Level3-v0",
    ]

    results = []

    for env_name in environments:
        try:
            result = test_environment(env_name, n_episodes=5, verbose=False)
            results.append(result)
            print("\n✓ Test passed!\n")
        except Exception as e:
            print(f"\n✗ Test failed with error: {e}\n")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for result in results:
        print(f"\n{result['env_name']}:")
        print(f"  Average episode length: {np.mean(result['episode_lengths']):.1f} steps")
        print(f"  Average reward: {np.mean(result['episode_rewards']):.4f}")

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
