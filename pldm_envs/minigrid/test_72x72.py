#!/usr/bin/env python3
"""
Test script for 72x72 observations.

This script demonstrates that the PLDM wrapper can resize to different sizes,
including 72x72.
"""

import numpy as np
import matplotlib.pyplot as plt
from pldm_envs.minigrid.wrappers import make_pldm_env
import gymnasium as gym
import pldm_envs.minigrid


def test_72x72_wrapper():
    """Test 72x72 resizing."""
    print("\n" + "="*60)
    print("Test: 72x72 Wrapper (HWC, uint8)")
    print("="*60)

    env = make_pldm_env('MiniGrid-LongHorizon-Level1-v0', size=(72, 72))
    obs, info = env.reset()

    print(f"Observation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    print(f"Value range: [{obs.min()}, {obs.max()}]")
    print(f"Expected: shape=(72, 72, 3), dtype=uint8, range=[0, 255]")

    assert obs.shape == (72, 72, 3), f"Shape mismatch: {obs.shape}"
    assert obs.dtype == np.uint8, f"Dtype mismatch: {obs.dtype}"
    assert obs.min() >= 0 and obs.max() <= 255, f"Value range error"

    print("✓ 72x72 wrapper test passed!")
    env.close()
    return obs


def test_72x72_pytorch():
    """Test 72x72 with PyTorch format."""
    print("\n" + "="*60)
    print("Test: 72x72 PyTorch Wrapper (CHW, float32, [0, 1])")
    print("="*60)

    env = make_pldm_env(
        'MiniGrid-LongHorizon-Level1-v0',
        size=(72, 72),
        channel_first=True,
        normalize=True
    )
    obs, info = env.reset()

    print(f"Observation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    print(f"Value range: [{obs.min():.3f}, {obs.max():.3f}]")
    print(f"Expected: shape=(3, 72, 72), dtype=float32, range=[0.0, 1.0]")

    assert obs.shape == (3, 72, 72), f"Shape mismatch: {obs.shape}"
    assert obs.dtype == np.float32, f"Dtype mismatch: {obs.dtype}"

    print("✓ 72x72 PyTorch wrapper test passed!")
    env.close()
    return obs


def compare_sizes():
    """Compare 64x64 vs 72x72 observations visually."""
    print("\n" + "="*60)
    print("Comparing Different Observation Sizes")
    print("="*60)

    sizes = [
        (64, 64, "64x64"),
        (72, 72, "72x72"),
        (84, 84, "84x84"),
    ]

    levels = [
        ("MiniGrid-LongHorizon-Level1-v0", "Level 1"),
        ("MiniGrid-LongHorizon-Level2-v0", "Level 2"),
        ("MiniGrid-LongHorizon-Level3-v0", "Level 3"),
    ]

    fig, axes = plt.subplots(len(levels), len(sizes), figsize=(15, 15))

    for level_idx, (env_id, level_name) in enumerate(levels):
        for size_idx, (height, width, size_name) in enumerate(sizes):
            env = make_pldm_env(env_id, size=(height, width))
            obs, _ = env.reset()

            axes[level_idx, size_idx].imshow(obs)
            axes[level_idx, size_idx].set_title(
                f'{level_name} - {size_name}\n{obs.shape}',
                fontsize=10
            )
            axes[level_idx, size_idx].axis('off')
            env.close()

    plt.suptitle('Observation Size Comparison: 64x64 vs 72x72 vs 84x84',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    import os
    os.makedirs('pldm_envs/minigrid/visualizations', exist_ok=True)
    plt.savefig('pldm_envs/minigrid/visualizations/size_comparison.png',
                dpi=150, bbox_inches='tight')
    print("✓ Saved: visualizations/size_comparison.png")
    plt.close()


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# PLDM Wrapper Size Tests (64x64, 72x72, 84x84)")
    print("#"*60)

    # Run tests
    test_72x72_wrapper()
    test_72x72_pytorch()

    # Create visualizations
    compare_sizes()

    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)
    print("\nUsage examples:")
    print("  # 64x64 (default)")
    print("  env = make_pldm_env('MiniGrid-LongHorizon-Level1-v0')")
    print()
    print("  # 72x72")
    print("  env = make_pldm_env('MiniGrid-LongHorizon-Level1-v0', size=(72, 72))")
    print()
    print("  # 84x84 with PyTorch format")
    print("  env = make_pldm_env('MiniGrid-LongHorizon-Level1-v0',")
    print("                      size=(84, 84), channel_first=True, normalize=True)")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
