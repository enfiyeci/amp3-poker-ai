#!/usr/bin/env python3
"""
Proper AMP3 Evaluation with Input/Output Visualization
Shows exactly what goes into and comes out of the model
"""

import torch
import numpy as np
from amp3_network import AMP3Actor, AMP3Critic

def visualize_io_structure():
    """Show the input/output structure of AMP3"""
    print(f"\n{'='*70}")
    print("AMP3 MODEL INPUT/OUTPUT STRUCTURE")
    print(f"{'='*70}\n")

    print("ACTOR NETWORK INPUTS:")
    print("  1. personal        (batch, 8)   - Hole cards + stack/pot/bet info")
    print("  2. public          (batch, 22)  - Community cards + all stacks/bets")
    print("  3. position        (batch, 6)   - One-hot position (0-5)")
    print("  4. action_history  (seq, batch, 2) - Past actions sequence")
    print("  5. style_features  (batch, 24)  - Opponent style profiles (6 players × 4)")
    print("\nACTOR NETWORK OUTPUT:")
    print("  • action_logits    (batch, 4)   - [FOLD, CALL, RAISE_SMALL, RAISE_LARGE]")
    print("  • action_probs     (batch, 4)   - Softmax probabilities")
    print("  • predicted_action (int)        - Argmax of logits")

    print(f"\n{'-'*70}\n")

    print("CRITIC NETWORK INPUTS:")
    print("  1. all_holes       (batch, 24)  - All players' hole cards")
    print("  2. public          (batch, 22)  - Community cards + stacks/bets")
    print("  3. position        (batch, 6)   - One-hot position")
    print("  4. action_history  (seq, batch, 2) - Past actions sequence")
    print("\nCRITIC NETWORK OUTPUT:")
    print("  • value            (batch, 1)   - Estimated state value")

    print(f"\n{'='*70}\n")


def evaluate_amp3_with_examples(checkpoint_path, num_examples=5):
    """Evaluate AMP3 and show detailed input/output examples"""
    print(f"\n{'='*70}")
    print(f"EVALUATING AMP3: {checkpoint_path}")
    print(f"{'='*70}\n")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    episode = checkpoint.get('episode', 0) + 1

    print(f"Checkpoint: Episode {episode:,}")
    print(f"Progress: {episode / 120000 * 100:.1f}%")

    # Initialize models
    actor = AMP3Actor()
    critic = AMP3Critic()

    actor.load_state_dict(checkpoint['actor'])
    critic.load_state_dict(checkpoint['critic'])

    actor.eval()
    critic.eval()

    actor_params = sum(p.numel() for p in actor.parameters())
    critic_params = sum(p.numel() for p in critic.parameters())

    print(f"\nActor: {actor_params:,} parameters")
    print(f"Critic: {critic_params:,} parameters")
    print(f"Total: {actor_params + critic_params:,} parameters")

    # Run test examples
    print(f"\n{'='*70}")
    print(f"TESTING {num_examples} EXAMPLE SCENARIOS")
    print(f"{'='*70}\n")

    action_names = ['FOLD', 'CALL', 'RAISE_SMALL', 'RAISE_LARGE']
    action_counts = {name: 0 for name in action_names}
    confidences = []
    values = []

    with torch.no_grad():
        for i in range(num_examples):
            print(f"\n{'─'*70}")
            print(f"EXAMPLE {i+1}: Random Game State")
            print(f"{'─'*70}")

            # Create example inputs
            personal = torch.randn(1, 8) * 0.5
            public = torch.randn(1, 22) * 0.5
            position = torch.zeros(1, 6)
            position[0, i % 6] = 1.0
            action_history = torch.randn(3, 1, 2) * 0.3  # sequence length 3
            style_features = torch.randn(1, 24) * 0.3
            all_holes = torch.randn(1, 24) * 0.5

            # Show inputs
            print("\nINPUTS:")
            print(f"  personal:       shape={tuple(personal.shape)}, mean={personal.mean():.3f}")
            print(f"  public:         shape={tuple(public.shape)}, mean={public.mean():.3f}")
            print(f"  position:       {position[0].tolist()} (player {i % 6})")
            print(f"  action_history: shape={tuple(action_history.shape)}")
            print(f"  style_features: shape={tuple(style_features.shape)}")

            # Actor forward pass
            action_logits = actor(
                personal=personal,
                public=public,
                position=position,
                action_history=action_history,
                style_features=style_features
            )

            action_probs = torch.softmax(action_logits, dim=-1)
            predicted_action = action_logits.argmax(dim=-1).item()
            confidence = action_probs[0, predicted_action].item()

            # Critic forward pass
            value = critic(
                all_holes=all_holes,
                public=public,
                position=position,
                action_history=action_history
            )

            # Show outputs
            print("\nOUTPUTS:")
            print(f"  action_logits:  {action_logits[0].tolist()}")
            print(f"  action_probs:   {[f'{p:.3f}' for p in action_probs[0].tolist()]}")
            print(f"  predicted:      {action_names[predicted_action]} (confidence: {confidence:.1%})")
            print(f"  value:          {value.item():.3f}")

            # Track statistics
            action_counts[action_names[predicted_action]] += 1
            confidences.append(confidence)
            values.append(value.item())

    # Summary statistics
    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}\n")

    print("Action Distribution:")
    for action, count in action_counts.items():
        pct = count / num_examples * 100
        bar = '█' * int(pct / 5)
        print(f"  {action:12s}: {count:2d} ({pct:5.1f}%) {bar}")

    avg_confidence = np.mean(confidences) * 100
    avg_value = np.mean(values)

    print(f"\nAverage Confidence: {avg_confidence:.1f}%")
    print(f"Average Value:      {avg_value:.3f}")
    print(f"Value Range:        [{min(values):.3f}, {max(values):.3f}]")

    print(f"\n{'='*70}\n")

    return {
        'episode': episode,
        'action_counts': action_counts,
        'avg_confidence': avg_confidence,
        'avg_value': avg_value
    }


def run_full_evaluation(checkpoint_path, num_tests=1000):
    """Run comprehensive evaluation with statistics"""
    print(f"\n{'='*70}")
    print(f"FULL EVALUATION: {num_tests} Tests")
    print(f"{'='*70}\n")

    # Load models
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    episode = checkpoint.get('episode', 0) + 1

    actor = AMP3Actor()
    actor.load_state_dict(checkpoint['actor'])
    actor.eval()

    print(f"Testing episode {episode:,} checkpoint...")

    action_distribution = {'FOLD': 0, 'CALL': 0, 'RAISE_SMALL': 0, 'RAISE_LARGE': 0}
    action_names = ['FOLD', 'CALL', 'RAISE_SMALL', 'RAISE_LARGE']
    confidences = []
    probs_list = []

    with torch.no_grad():
        for i in range(num_tests):
            # Random inputs
            personal = torch.randn(1, 8) * 0.5
            public = torch.randn(1, 22) * 0.5
            position = torch.zeros(1, 6)
            position[0, i % 6] = 1.0
            action_history = torch.randn(3, 1, 2) * 0.3
            style_features = torch.randn(1, 24) * 0.3

            # Forward pass
            action_logits = actor(
                personal=personal,
                public=public,
                position=position,
                action_history=action_history,
                style_features=style_features
            )

            action_probs = torch.softmax(action_logits, dim=-1)
            predicted_action = action_logits.argmax(dim=-1).item()
            confidence = action_probs[0, predicted_action].item()

            action_distribution[action_names[predicted_action]] += 1
            confidences.append(confidence)
            probs_list.append(action_probs[0].numpy())

            if (i + 1) % 200 == 0:
                print(f"  Tested {i+1}/{num_tests}...")

    # Calculate metrics
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}\n")

    print("Action Distribution:")
    for action, count in action_distribution.items():
        pct = count / num_tests * 100
        bar = '█' * int(pct / 2)
        print(f"  {action:12s}: {count:4d} ({pct:5.1f}%) {bar}")

    # Strategy diversity (entropy)
    avg_probs = np.mean(probs_list, axis=0)
    entropy = -np.sum(avg_probs * np.log2(avg_probs + 1e-10))
    max_entropy = 2.0
    diversity_pct = (entropy / max_entropy) * 100

    avg_confidence = np.mean(confidences) * 100

    # Quality score
    quality_score = (avg_confidence / 100) * diversity_pct

    print(f"\nMetrics:")
    print(f"  Average Confidence: {avg_confidence:.1f}%")
    print(f"  Strategy Diversity: {diversity_pct:.1f}% (entropy: {entropy:.3f}/{max_entropy:.1f})")
    print(f"  Quality Score:      {quality_score:.1f}/100")

    print(f"\n{'='*70}")
    print(f"QUALITY SCORE: {quality_score:.1f} / 100")
    print(f"{'='*70}")

    print(f"\nComparison:")
    print(f"  Preflop:        72.0")
    print(f"  AMP3 (100k):    {quality_score:.1f} ← Current")
    print(f"  Turn:           59.4")
    print(f"  River:          58.1")
    print(f"  Flop:           51.2")

    if quality_score >= 65:
        print(f"\n✅ EXCELLENT - AMP3 is performing at target level!")
    elif quality_score >= 60:
        print(f"\n✅ GOOD - AMP3 is competitive with specialized models!")
    elif quality_score >= 55:
        print(f"\n✅ DECENT - AMP3 showing solid performance!")
    else:
        print(f"\n⚠️  Needs more training to reach target performance")

    print(f"\n{'='*70}\n")

    return quality_score


if __name__ == "__main__":
    checkpoint_path = 'checkpoints_improved/amp3_checkpoint_100000.pt'

    # Show structure
    visualize_io_structure()

    # Run detailed examples
    evaluate_amp3_with_examples(checkpoint_path, num_examples=5)

    # Run full evaluation
    quality_score = run_full_evaluation(checkpoint_path, num_tests=1000)

    print(f"Evaluation complete! Quality score: {quality_score:.1f}")
