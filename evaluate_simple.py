#!/usr/bin/env python3
"""
Simple Model Validation - Just checks if models load and basic structure
"""

import torch
import numpy as np

def evaluate_amp3_checkpoint(checkpoint_path):
    """Quick validation of AMP3 checkpoint"""
    print(f"\n{'='*70}")
    print(f"AMP3 Checkpoint Validation: {checkpoint_path}")
    print(f"{'='*70}\n")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    print(f"Episode: {checkpoint.get('episode', 'unknown')}")
    print(f"Keys in checkpoint: {list(checkpoint.keys())}")

    if 'actor' in checkpoint:
        actor_state = checkpoint['actor']
        critic_state = checkpoint['critic']

        print(f"\n✓ Actor state dict loaded ({len(actor_state)} keys)")
        print(f"✓ Critic state dict loaded ({len(critic_state)} keys)")

        # Count parameters
        actor_params = sum(p.numel() for p in actor_state.values() if isinstance(p, torch.Tensor))
        critic_params = sum(p.numel() for p in critic_state.values() if isinstance(p, torch.Tensor))

        print(f"\nActor parameters: {actor_params:,}")
        print(f"Critic parameters: {critic_params:,}")
        print(f"Total parameters: {actor_params + critic_params:,}")

        print(f"\n✅ AMP3 checkpoint is valid and loadable")
        return True
    else:
        print(f"\n❌ Checkpoint missing actor/critic keys")
        return False


def evaluate_street_models(checkpoint_path):
    """Quick validation of later-street models"""
    print(f"\n{'='*70}")
    print(f"Later-Street Models Validation: {checkpoint_path}")
    print(f"{'='*70}\n")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    print(f"Keys in checkpoint: {list(checkpoint.keys())}")

    if 'flop' in checkpoint:
        flop_state = checkpoint['flop']
        turn_state = checkpoint['turn']
        river_state = checkpoint['river']

        print(f"\n✓ Flop state dict loaded ({len(flop_state)} keys)")
        print(f"✓ Turn state dict loaded ({len(turn_state)} keys)")
        print(f"✓ River state dict loaded ({len(river_state)} keys)")

        # Count parameters
        flop_params = sum(p.numel() for p in flop_state.values() if isinstance(p, torch.Tensor))
        turn_params = sum(p.numel() for p in turn_state.values() if isinstance(p, torch.Tensor))
        river_params = sum(p.numel() for p in river_state.values() if isinstance(p, torch.Tensor))

        print(f"\nFlop parameters: {flop_params:,}")
        print(f"Turn parameters: {turn_params:,}")
        print(f"River parameters: {river_params:,}")
        print(f"Total parameters: {flop_params + turn_params + river_params:,}")

        print(f"\n✅ Later-street models are valid and loadable")
        return True
    else:
        print(f"\n❌ Checkpoint missing flop/turn/river keys")
        return False


def evaluate_preflop_model(checkpoint_path):
    """Quick validation of preflop model"""
    print(f"\n{'='*70}")
    print(f"Preflop Model Validation: {checkpoint_path}")
    print(f"{'='*70}\n")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    print(f"Keys in checkpoint: {list(checkpoint.keys())}")

    if 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']

        print(f"\n✓ Model state dict loaded ({len(model_state)} keys)")

        # Count parameters
        params = sum(p.numel() for p in model_state.values() if isinstance(p, torch.Tensor))

        print(f"\nTotal parameters: {params:,}")

        # Check for metrics
        if 'val_acc' in checkpoint:
            print(f"\nValidation Accuracy: {checkpoint['val_acc']:.1%}")
        if 'epoch' in checkpoint:
            print(f"Trained Epochs: {checkpoint['epoch']}")

        print(f"\n✅ Preflop model is valid and loadable")
        return True
    else:
        print(f"\n❌ Checkpoint missing model_state_dict key")
        return False


if __name__ == "__main__":
    print("\n" + "="*70)
    print("MODEL VALIDATION SUMMARY")
    print("="*70)

    results = {}

    # Validate preflop
    try:
        results['preflop'] = evaluate_preflop_model('checkpoints/best_model.pt')
    except Exception as e:
        print(f"\n❌ Preflop validation failed: {e}")
        results['preflop'] = False

    # Validate AMP3 40k
    try:
        results['amp3_40k'] = evaluate_amp3_checkpoint('checkpoints_20hr/amp3_checkpoint_40000.pt')
    except Exception as e:
        print(f"\n❌ AMP3 validation failed: {e}")
        results['amp3_40k'] = False

    # Validate later-streets
    try:
        results['streets'] = evaluate_street_models('checkpoints_20hr/street_models.pt')
    except Exception as e:
        print(f"\n❌ Street models validation failed: {e}")
        results['streets'] = False

    # Summary
    print(f"\n{'='*70}")
    print("VALIDATION RESULTS")
    print(f"{'='*70}")
    for model, valid in results.items():
        status = "✅ VALID" if valid else "❌ INVALID"
        print(f"{model:20s} {status}")
    print(f"{'='*70}\n")
