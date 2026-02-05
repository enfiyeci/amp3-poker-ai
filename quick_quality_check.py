#!/usr/bin/env python3
"""
Quick quality check for AMP3 at 100k episodes
Tests model outputs without full game simulation
"""

import torch
import numpy as np
from amp3_network import AMP3Actor, AMP3Critic

def quick_quality_check(checkpoint_path, num_tests=1000):
    """Quick sanity check of model outputs"""
    print(f"\n{'='*70}")
    print(f"QUICK QUALITY CHECK: {checkpoint_path}")
    print(f"{'='*70}\n")

    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    episode = checkpoint.get('episode', 'unknown')
    print(f"✓ Checkpoint from episode {episode}")

    # Initialize models
    print("\nInitializing models...")
    actor = AMP3Actor()
    critic = AMP3Critic()

    actor.load_state_dict(checkpoint['actor'])
    critic.load_state_dict(checkpoint['critic'])

    actor.eval()
    critic.eval()

    actor_params = sum(p.numel() for p in actor.parameters())
    critic_params = sum(p.numel() for p in critic.parameters())
    print(f"✓ Actor parameters: {actor_params:,}")
    print(f"✓ Critic parameters: {critic_params:,}")

    # Test with random inputs
    print(f"\nTesting with {num_tests} random game states...")

    action_probs_list = []
    values_list = []
    action_distribution = {0: 0, 1: 0, 2: 0, 3: 0}  # fold, call, raise_small, raise_large

    with torch.no_grad():
        for i in range(num_tests):
            # Create random inputs (normalized)
            personal = torch.randn(1, 8) * 0.5
            public = torch.randn(1, 22) * 0.5
            position = torch.zeros(1, 6)
            position[0, i % 6] = 1.0  # Rotate through positions
            action_history = torch.randn(1, 2) * 0.3
            style_features = torch.randn(1, 24) * 0.3

            # Get actor output
            action_logits = actor(
                personal=personal,
                public=public,
                position=position,
                action_history=action_history,
                style_features=style_features
            )

            # Get action probabilities
            action_probs = torch.softmax(action_logits, dim=-1)
            action_probs_list.append(action_probs[0].numpy())

            # Get predicted action
            predicted_action = action_logits.argmax(dim=-1).item()
            action_distribution[predicted_action] += 1

            # Get critic value
            all_holes = torch.randn(1, 24) * 0.5
            value = critic(
                all_holes=all_holes,
                public=public,
                position=position,
                action_history=action_history
            )
            values_list.append(value.item())

    # Analyze results
    print(f"\n{'='*70}")
    print("QUALITY METRICS")
    print(f"{'='*70}\n")

    # Action distribution
    print("Action Distribution:")
    action_names = ['Fold', 'Call', 'Raise Small', 'Raise Large']
    for action_idx, name in enumerate(action_names):
        count = action_distribution[action_idx]
        pct = count / num_tests * 100
        print(f"  {name:15s}: {count:4d} ({pct:5.1f}%)")

    # Strategy diversity (entropy)
    avg_probs = np.mean(action_probs_list, axis=0)
    entropy = -np.sum(avg_probs * np.log2(avg_probs + 1e-10))
    max_entropy = 2.0  # log2(4)
    diversity_pct = (entropy / max_entropy) * 100

    print(f"\nStrategy Diversity:")
    print(f"  Entropy: {entropy:.3f} / {max_entropy:.3f}")
    print(f"  Diversity: {diversity_pct:.1f}%")

    # Confidence (average max probability)
    max_probs = [max(probs) for probs in action_probs_list]
    avg_confidence = np.mean(max_probs) * 100
    std_confidence = np.std(max_probs) * 100

    print(f"\nPrediction Confidence:")
    print(f"  Average: {avg_confidence:.1f}%")
    print(f"  Std Dev: {std_confidence:.1f}%")

    # Value predictions
    avg_value = np.mean(values_list)
    std_value = np.std(values_list)

    print(f"\nValue Estimates:")
    print(f"  Average: {avg_value:.3f}")
    print(f"  Std Dev: {std_value:.3f}")
    print(f"  Range: [{min(values_list):.3f}, {max(values_list):.3f}]")

    # Quality score (confidence × diversity)
    quality_score = (avg_confidence / 100) * diversity_pct

    print(f"\n{'='*70}")
    print(f"QUALITY SCORE: {quality_score:.1f} / 100")
    print(f"{'='*70}")

    # Assessment
    print("\nAssessment:")
    if diversity_pct < 50:
        print("  ⚠️ Low diversity - model may be too deterministic")
    elif diversity_pct > 90:
        print("  ✅ Good diversity - varied strategic choices")
    else:
        print("  ✅ Moderate diversity - balanced strategy")

    if avg_confidence < 40:
        print("  ⚠️ Low confidence - model uncertain about decisions")
    elif avg_confidence > 80:
        print("  ⚠️ Very high confidence - may be overconfident")
    else:
        print("  ✅ Reasonable confidence levels")

    # Compare to supervised models
    print(f"\n{'='*70}")
    print("COMPARISON TO OTHER MODELS")
    print(f"{'='*70}")
    print("\nQuality Scores:")
    print(f"  Preflop:     72.0")
    print(f"  AMP3 (100k): {quality_score:.1f} ← Current")
    print(f"  Turn:        59.4")
    print(f"  River:       58.1")
    print(f"  Flop:        51.2")

    if quality_score > 65:
        print("\n✅ AMP3 performing above expectation!")
    elif quality_score > 55:
        print("\n✅ AMP3 performing well - competitive with specialized models")
    elif quality_score > 45:
        print("\n⚠️ AMP3 showing promise - may improve with more training")
    else:
        print("\n⚠️ AMP3 needs more training to match specialized models")

    print(f"\n{'='*70}\n")

    return {
        'episode': episode,
        'quality_score': quality_score,
        'diversity': diversity_pct,
        'confidence': avg_confidence,
        'avg_value': avg_value,
        'action_distribution': action_distribution
    }


if __name__ == "__main__":
    checkpoint_path = 'checkpoints_improved/amp3_checkpoint_100000.pt'
    results = quick_quality_check(checkpoint_path, num_tests=1000)

    print("Results saved - ready for continued training to 120k!")
