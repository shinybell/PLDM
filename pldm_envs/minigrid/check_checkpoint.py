"""チェックポイントの内容を確認するスクリプト"""
import torch
import sys

if len(sys.argv) < 2:
    print("Usage: python check_checkpoint.py <checkpoint_path>")
    sys.exit(1)

checkpoint_path = sys.argv[1]
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("Checkpoint keys:")
for key in checkpoint.keys():
    print(f"  - {key}")
    if isinstance(checkpoint[key], dict):
        print(f"    (dict with {len(checkpoint[key])} items)")
    elif isinstance(checkpoint[key], torch.Tensor):
        print(f"    (tensor with shape {checkpoint[key].shape})")
    else:
        print(f"    ({type(checkpoint[key])})")
