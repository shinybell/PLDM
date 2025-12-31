#!/usr/bin/env python3
"""
Test script for PLDM wrapper functionality.

This script demonstrates:
1. Creating 64x64 observations for PLDM
2. Different wrapper configurations (channel-first, normalization)
3. Visualizing the resized observations
"""

import numpy as np
import matplotlib.pyplot as plt
from pldm_envs.minigrid.wrappers import make_pldm_env, PLDMWrapper
import gymnasium as gym
import pldm_envs.minigrid


def test_basic_wrapper():
    """Test basic 64x64 resizing without normalization."""
    print("\n" + "="*60)
    print("Test 1: Basic 64x64 Wrapper (HWC, uint8)")
    print("="*60)

    env = make_pldm_env('MiniGrid-LongHorizon-Level1-v0')
    obs, info = env.reset()

    print(f"Observation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    print(f"Value range: [{obs.min()}, {obs.max()}]")
    print(f"Expected: shape=(64, 64, 3), dtype=uint8, range=[0, 255]")

    assert obs.shape == (64, 64, 3), f"Shape mismatch: {obs.shape}"
    assert obs.dtype == np.uint8, f"Dtype mismatch: {obs.dtype}"
    assert obs.min() >= 0 and obs.max() <= 255, f"Value range error"

    print("✓ Basic wrapper test passed!")
    env.close()
    return obs


def test_channel_first_wrapper():
    """Test channel-first format for PyTorch."""
    print("\n" + "="*60)
    print("Test 2: Channel-First Wrapper (CHW, uint8)")
    print("="*60)

    env = make_pldm_env(
        'MiniGrid-LongHorizon-Level1-v0',
        channel_first=True
    )
    obs, info = env.reset()

    print(f"Observation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    print(f"Value range: [{obs.min()}, {obs.max()}]")
    print(f"Expected: shape=(3, 64, 64), dtype=uint8, range=[0, 255]")

    assert obs.shape == (3, 64, 64), f"Shape mismatch: {obs.shape}"
    assert obs.dtype == np.uint8, f"Dtype mismatch: {obs.dtype}"

    print("✓ Channel-first wrapper test passed!")
    env.close()
    return obs


def test_normalized_wrapper():
    """Test normalized float32 format."""
    print("\n" + "="*60)
    print("Test 3: Normalized Wrapper (HWC, float32, [0, 1])")
    print("="*60)

    env = make_pldm_env(
        'MiniGrid-LongHorizon-Level1-v0',
        normalize=True
    )
    obs, info = env.reset()

    print(f"Observation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    print(f"Value range: [{obs.min():.3f}, {obs.max():.3f}]")
    print(f"Expected: shape=(64, 64, 3), dtype=float32, range=[0.0, 1.0]")

    assert obs.shape == (64, 64, 3), f"Shape mismatch: {obs.shape}"
    assert obs.dtype == np.float32, f"Dtype mismatch: {obs.dtype}"
    assert 0.0 <= obs.min() and obs.max() <= 1.0, f"Value range error"

    print("✓ Normalized wrapper test passed!")
    env.close()
    return obs


def test_full_pytorch_wrapper():
    """Test full PyTorch-compatible wrapper (CHW, float32, normalized)."""
    print("\n" + "="*60)
    print("Test 4: Full PyTorch Wrapper (CHW, float32, [0, 1])")
    print("="*60)

    env = make_pldm_env(
        'MiniGrid-LongHorizon-Level1-v0',
        channel_first=True,
        normalize=True
    )
    obs, info = env.reset()

    print(f"Observation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    print(f"Value range: [{obs.min():.3f}, {obs.max():.3f}]")
    print(f"Expected: shape=(3, 64, 64), dtype=float32, range=[0.0, 1.0]")

    assert obs.shape == (3, 64, 64), f"Shape mismatch: {obs.shape}"
    assert obs.dtype == np.float32, f"Dtype mismatch: {obs.dtype}"

    print("✓ Full PyTorch wrapper test passed!")
    env.close()
    return obs


def visualize_resized_observations():
    """Visualize 64x64 observations for all three levels."""
    print("\n" + "="*60)
    print("Visualizing 64x64 Observations")
    print("="*60)

    levels = [
        ("MiniGrid-LongHorizon-Level1-v0", "Level 1"),
        ("MiniGrid-LongHorizon-Level2-v0", "Level 2"),
        ("MiniGrid-LongHorizon-Level3-v0", "Level 3"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Original size observations
    for idx, (env_id, level_name) in enumerate(levels):
        env = gym.make(env_id, render_mode='rgb_array')
        env.reset()
        original_obs = env.render()

        axes[0, idx].imshow(original_obs)
        axes[0, idx].set_title(f'{level_name} - Original\n{original_obs.shape}')
        axes[0, idx].axis('off')
        env.close()

    # Row 2: 64x64 resized observations
    for idx, (env_id, level_name) in enumerate(levels):
        env = make_pldm_env(env_id)
        obs, _ = env.reset()

        axes[1, idx].imshow(obs)
        axes[1, idx].set_title(f'{level_name} - 64x64\n{obs.shape}')
        axes[1, idx].axis('off')
        env.close()

    plt.suptitle('Observation Size Comparison: Original vs 64x64',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    import os
    os.makedirs('pldm_envs/minigrid/visualizations', exist_ok=True)
    plt.savefig('pldm_envs/minigrid/visualizations/pldm_64x64_observations.png',
                dpi=150, bbox_inches='tight')
    print("✓ Saved: visualizations/pldm_64x64_observations.png")
    plt.close()


def test_multiple_episodes():
    """Test that wrapper works correctly across multiple episodes."""
    print("\n" + "="*60)
    print("Test 5: Multiple Episodes")
    print("="*60)

    env = make_pldm_env('MiniGrid-LongHorizon-Level1-v0')

    for i in range(5):
        obs, info = env.reset()
        print(f"Episode {i+1}: shape={obs.shape}, dtype={obs.dtype}, "
              f"range=[{obs.min()}, {obs.max()}]")

        assert obs.shape == (64, 64, 3)
        assert obs.dtype == np.uint8

    print("✓ Multiple episodes test passed!")
    env.close()


def create_all_level_64x64_samples():
    """Create sample 64x64 observations for all levels."""
    print("\n" + "="*60)
    print("Creating 64x64 Sample Observations for All Levels")
    print("="*60)

    levels = [
        ("MiniGrid-LongHorizon-Level1-v0", "Level 1"),
        ("MiniGrid-LongHorizon-Level2-v0", "Level 2"),
        ("MiniGrid-LongHorizon-Level3-v0", "Level 3"),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    for level_idx, (env_id, level_name) in enumerate(levels):
        env = make_pldm_env(env_id)

        for ep_idx in range(4):
            obs, _ = env.reset()

            axes[level_idx, ep_idx].imshow(obs)
            axes[level_idx, ep_idx].set_title(
                f'{level_name} - Episode {ep_idx+1}',
                fontsize=10
            )
            axes[level_idx, ep_idx].axis('off')

        env.close()

    plt.suptitle('PLDM 64x64 Observations - All Levels (4 Episodes Each)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    import os
    os.makedirs('pldm_envs/minigrid/visualizations', exist_ok=True)
    plt.savefig('pldm_envs/minigrid/visualizations/pldm_all_levels_64x64.png',
                dpi=150, bbox_inches='tight')
    print("✓ Saved: visualizations/pldm_all_levels_64x64.png")
    plt.close()


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# PLDM Wrapper Tests")
    print("#"*60)

    # Run basic tests
    test_basic_wrapper()
    test_channel_first_wrapper()
    test_normalized_wrapper()
    test_full_pytorch_wrapper()
    test_multiple_episodes()

    # Create visualizations
    visualize_resized_observations()
    create_all_level_64x64_samples()

    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)
    print("\nUsage examples:")
    print("  # Basic 64x64 uint8 (HWC format)")
    print("  env = make_pldm_env('MiniGrid-LongHorizon-Level1-v0')")
    print()
    print("  # PyTorch format (CHW, float32, normalized)")
    print("  env = make_pldm_env('MiniGrid-LongHorizon-Level1-v0',")
    print("                      channel_first=True, normalize=True)")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
