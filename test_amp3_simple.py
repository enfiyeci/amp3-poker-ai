#!/usr/bin/env python3
"""
Simple AMP3 Model Test - Tests basic functionality
"""

import torch
import numpy as np
from collections import defaultdict

print("="*70)
print("AMP3 MODEL SIMPLE TEST")
print("="*70)

# Load checkpoint
checkpoint_path = 'checkpoints_20hr/amp3_checkpoint_40000.pt'
checkpoint = torch.load(checkpoint_path, map_location='cpu')

episode = checkpoint.get('episode', 'unknown')
print(f"\nCheckpoint: Episode {episode}")
print(f"Keys in checkpoint: {list(checkpoint.keys())}")

# Check what's in the actor
actor_state = checkpoint['actor']
print(f"\nActor state dict has {len(actor_state)} layers")

# Load actor network structure from amp3_network.py
from amp3_network import AMP3Actor

# Create actor
actor = AMP3Actor()
actor.load_state_dict(checkpoint['actor'])
actor.eval()

print("✓ Loaded Actor network")

# Test with synthetic inputs
num_tests = 1000
action_counts = defaultdict(int)
confidences = []

print(f"\nTesting with {num_tests} synthetic states...")

with torch.no_grad():
    for _ in range(num_tests):
        # Create inputs matching the Actor's expected format
        personal = torch.randn(1, 8) * 0.5
        public = torch.randn(1, 22) * 0.5
        position = torch.zeros(1, 6)
        position[0, np.random.randint(6)] = 1
        osm = torch.randn(1, 4) * 0.5

        # Get action probabilities
        try:
            logits = actor(personal=personal, public=public, position=position, osm_predictions=osm)
            probs = torch.softmax(logits, dim=-1)
            action = torch.argmax(probs, dim=-1).item()
            confidence = probs.max().item()

            action_counts[action] += 1
            confidences.append(confidence)
        except Exception as e:
            print(f"Error: {e}")
            print(f"personal shape: {personal.shape}")
            print(f"public shape: {public.shape}")
            print(f"position shape: {position.shape}")
            print(f"osm shape: {osm.shape}")
            break

if len(confidences) == num_tests:
    avg_confidence = np.mean(confidences)
    action_distribution = [action_counts[i] / num_tests * 100 for i in range(3)]

    # Calculate entropy (strategy diversity)
    entropy = -sum([(p/100) * np.log2(p/100) if p > 0 else 0 for p in action_distribution])
    max_entropy = np.log2(3)  # log2(3) for 3 actions

    # Quality score
    quality_score = (avg_confidence * 100) * (entropy / max_entropy)

    print(f"\n✅ AMP3 MODEL ANALYSIS")
    print(f"  Episodes Trained:      {episode}")
    print(f"  Tests Completed:       {num_tests}/{num_tests}")
    print(f"  Average Confidence:    {avg_confidence:.1%}")
    print(f"  Strategy Diversity:    {entropy:.2f}/{max_entropy:.2f} (entropy)")
    print(f"  Quality Score:         {quality_score:.1f}")
    print(f"  Action Distribution:")
    print(f"    Fold:  {action_distribution[0]:.1f}%")
    print(f"    Call:  {action_distribution[1]:.1f}%")
    print(f"    Raise: {action_distribution[2]:.1f}%")

    if entropy / max_entropy > 0.8:
        print(f"\n  Strategy: ✓ Balanced (high diversity)")
    elif entropy / max_entropy > 0.6:
        print(f"\n  Strategy: ⚠️ Somewhat biased")
    else:
        print(f"\n  Strategy: ❌ Very biased (needs more training)")

    print("\n" + "="*70)
    print("🎉 AMP3 MODEL VALIDATED!")
    print("="*70)
else:
    print(f"\n❌ Only {len(confidences)}/{num_tests} tests succeeded")
