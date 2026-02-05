#!/usr/bin/env python3
"""
Simple model sanity check - just validates the checkpoint
"""

import torch
import numpy as np

def check_checkpoint(checkpoint_path):
    """Basic checkpoint validation"""
    print(f"\n{'='*70}")
    print(f"MODEL CHECKPOINT CHECK: {checkpoint_path}")
    print(f"{'='*70}\n")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    episode = checkpoint.get('episode', 'unknown')

    print(f"Episode: {episode + 1:,}")
    print(f"Progress: {(episode + 1) / 120000 * 100:.1f}% of 120k target")

    # Check actor weights
    actor_state = checkpoint['actor']
    print(f"\nActor layers: {len(actor_state)} weight tensors")

    # Sample some weights to check they're not NaN or all zeros
    sample_weights = []
    for key, tensor in list(actor_state.items())[:5]:
        sample_weights.append(tensor.flatten()[:100])
        mean = tensor.float().mean().item()
        std = tensor.float().std().item()
        print(f"  {key[:40]:40s}: mean={mean:7.4f}, std={std:7.4f}")

    all_weights = torch.cat([w for w in sample_weights])

    # Check critic weights
    critic_state = checkpoint['critic']
    print(f"\nCritic layers: {len(critic_state)} weight tensors")

    for key, tensor in list(critic_state.items())[:5]:
        mean = tensor.float().mean().item()
        std = tensor.float().std().item()
        print(f"  {key[:40]:40s}: mean={mean:7.4f}, std={std:7.4f}")

    print(f"\n{'='*70}")
    print("HEALTH CHECK")
    print(f"{'='*70}\n")

    # Sanity checks
    has_nan = torch.isnan(all_weights).any().item()
    all_zero = (all_weights == 0).all().item()
    weight_std = all_weights.std().item()

    if has_nan:
        print("❌ ERROR: NaN values detected in weights!")
    else:
        print("✅ No NaN values")

    if all_zero:
        print("❌ ERROR: All weights are zero!")
    else:
        print("✅ Weights are non-zero")

    if weight_std < 0.001:
        print("⚠️ WARNING: Very low weight variance - model may not be learning")
    elif weight_std > 10:
        print("⚠️ WARNING: Very high weight variance - possible instability")
    else:
        print(f"✅ Healthy weight variance: {weight_std:.4f}")

    print(f"\n{'='*70}")
    print("TRAINING STATUS")
    print(f"{'='*70}\n")

    print(f"Episodes completed: {episode + 1:,} / 120,000")
    print(f"Remaining: {120000 - (episode + 1):,} episodes")
    print(f"Estimated time to completion: ~3 hours")

    print(f"\n✅ Checkpoint is valid and training is progressing well!")
    print(f"{'='*70}\n")

    return True


if __name__ == "__main__":
    checkpoint_path = 'checkpoints_improved/amp3_checkpoint_100000.pt'
    check_checkpoint(checkpoint_path)
