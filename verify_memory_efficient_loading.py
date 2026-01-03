
import os
import shutil
import numpy as np
import torch
from pathlib import Path
import sys

# Add project root to path
sys.path.append(os.getcwd())

from pldm_envs.minigrid.data.minigrid_dataset import MiniGridDataset, MiniGridDatasetConfig
from pldm_envs.minigrid.data_generation.merge_datasets import merge_datasets_memmap

def create_dummy_data(output_path, n_episodes=10, episode_length=20):
    """Create dummy .npz data for testing"""
    print(f"Creating dummy data at {output_path}")
    
    observations = np.random.randint(0, 255, (n_episodes, episode_length, 64, 64, 3), dtype=np.uint8)
    actions = np.random.randint(0, 7, (n_episodes, episode_length-1), dtype=np.int64)
    rewards = np.random.rand(n_episodes, episode_length-1).astype(np.float32)
    dones = np.zeros((n_episodes, episode_length-1), dtype=bool)
    positions = np.random.rand(n_episodes, episode_length, 2).astype(np.float32)
    
    np.savez_compressed(
        output_path,
        observations=observations,
        actions=actions,
        rewards=rewards,
        dones=dones,
        positions=positions
    )

def test_merge_and_load():
    # Setup paths
    base_dir = Path("data/test_memory_efficient")
    if base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    input_dir = base_dir / "input"
    input_dir.mkdir()
    
    output_dir = base_dir / "merged_dataset" # Directory output
    
    # 1. Create dummy input files
    input_files = []
    for i in range(3):
        p = input_dir / f"worker_{i}.npz"
        create_dummy_data(p, n_episodes=5)
        input_files.append(str(p))
        
    # 2. Merge into directory
    print("\nTesting merge_datasets_memmap (Directory Output)...")
    merge_datasets_memmap(input_files, str(output_dir), is_dir=True)
    
    # Verify directory content
    expected_files = ['observations.npy', 'actions.npy', 'rewards.npy', 'dones.npy', 'positions.npy']
    for f in expected_files:
        assert (output_dir / f).exists(), f"Missing {f} in output directory"
        
    print("Merge successful. Output directory contains all expected .npy files.")
    
    # 3. Load using MiniGridDataset
    print("\nTesting MiniGridDataset loading...")
    config = MiniGridDatasetConfig(
        data_path=str(output_dir),
        sample_length=10,
        img_size=64,
        include_rewards=True,
        include_dones=True
    )
    
    dataset = MiniGridDataset(config)
    
    # Check length
    # Total episodes = 3 workers * 5 episodes = 15
    # Episode length = 20
    # Sample length = 10
    # Slices per episode = 20 - 10 + 1 = 11
    # Total slices = 15 * 11 = 165
    print(f"Dataset length: {len(dataset)}")
    assert len(dataset) == 165, f"Expected 165 samples, got {len(dataset)}"
    
    # Check sample loading
    sample = dataset[0]
    print("Sample loaded successfully:")
    print(f"  States: {sample.states.shape}")
    print(f"  Actions: {sample.actions.shape}")
    print(f"  Locations: {sample.locations.shape}")
    
    assert sample.states.shape == (10, 3, 64, 64)
    assert sample.actions.shape == (9, 7) # One-hot
    
    print("\nVerification PASSED!")
    
    # Cleanup
    shutil.rmtree(base_dir)

if __name__ == "__main__":
    test_merge_and_load()
