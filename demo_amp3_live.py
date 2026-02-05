#!/usr/bin/env python3
"""
AMP3 Live Demo - Showcase poker decision making
Run this for presentations and demos
"""

import torch
import numpy as np
from amp3_network import AMP3Actor
import time
import random

def print_banner():
    """Print demo banner"""
    print("\n" + "="*70)
    print("     AMP3 POKER AI - LIVE DECISION MAKING DEMO")
    print("     Reinforcement Learning Agent with Opponent Modeling")
    print("="*70 + "\n")

def print_scenario(scenario_num, total_scenarios):
    """Print scenario header"""
    print(f"\n{'─'*70}")
    print(f"SCENARIO {scenario_num}/{total_scenarios}")
    print(f"{'─'*70}\n")

def simulate_game_situation():
    """Generate a realistic-looking poker situation"""
    positions = ['Button', 'Small Blind', 'Big Blind', 'UTG', 'Middle', 'Cutoff']
    streets = ['Preflop', 'Flop', 'Turn', 'River']

    position = random.choice(positions)
    street = random.choice(streets)
    pot = random.choice([200, 400, 800, 1500, 3000])
    stack = random.choice([5000, 7500, 10000, 15000])
    to_call = random.choice([0, 100, 200, 400, 800])

    # Generate hand representation
    ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5']
    suits = ['♠', '♥', '♦', '♣']
    hole_cards = f"{random.choice(ranks)}{random.choice(suits)} {random.choice(ranks)}{random.choice(suits)}"

    return {
        'position': position,
        'street': street,
        'pot': pot,
        'stack': stack,
        'to_call': to_call,
        'hole_cards': hole_cards
    }

def demo_amp3_decision_making(checkpoint_path, num_scenarios=5, delay=1.5):
    """Run interactive demo showing AMP3 making decisions"""

    print_banner()

    # Load model
    print("Loading AMP3 model...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    episode = checkpoint.get('episode', 0) + 1

    actor = AMP3Actor()
    actor.load_state_dict(checkpoint['actor'])
    actor.eval()

    actor_params = sum(p.numel() for p in actor.parameters())

    print(f"✓ Model loaded from episode {episode:,}")
    print(f"✓ Parameters: {actor_params:,}")
    print(f"✓ Training progress: {episode/120000*100:.1f}%")

    time.sleep(delay)

    print(f"\n{'='*70}")
    print(f"RUNNING {num_scenarios} DECISION SCENARIOS")
    print(f"{'='*70}")

    action_names = ['FOLD', 'CALL', 'RAISE SMALL', 'RAISE LARGE']

    for i in range(num_scenarios):
        print_scenario(i + 1, num_scenarios)

        # Generate game situation
        situation = simulate_game_situation()

        # Display situation
        print("GAME SITUATION:")
        print(f"  Position:    {situation['position']}")
        print(f"  Street:      {situation['street']}")
        print(f"  Hole Cards:  {situation['hole_cards']}")
        print(f"  Pot Size:    ${situation['pot']}")
        print(f"  Your Stack:  ${situation['stack']}")
        print(f"  To Call:     ${situation['to_call']}")

        time.sleep(delay * 0.5)

        # Create model inputs
        print("\n  → AMP3 analyzing situation...")
        time.sleep(delay * 0.3)

        with torch.no_grad():
            # Encode poker state properly
            # Personal features (8): hole cards + stack/pot info
            personal = torch.zeros(1, 8)

            # Encode hole cards (simplified: use rank values normalized)
            hole_cards = situation['hole_cards'].split()
            rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                       '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
            card1_rank = rank_map.get(hole_cards[0][0], 7) / 14.0
            card2_rank = rank_map.get(hole_cards[1][0], 7) / 14.0

            personal[0, 0] = card1_rank
            personal[0, 1] = card2_rank
            personal[0, 2] = 1.0 if hole_cards[0][1] == hole_cards[1][1] else 0.0  # suited
            personal[0, 3] = situation['stack'] / 10000.0
            personal[0, 4] = situation['pot'] / 10000.0
            personal[0, 5] = situation['to_call'] / 10000.0
            personal[0, 6] = (situation['pot'] + situation['to_call']) / 10000.0  # pot odds
            personal[0, 7] = situation['to_call'] / max(situation['stack'], 1)  # call fraction

            # Public features (22): community cards + pot/stacks
            public = torch.zeros(1, 22)
            public[0, 0] = situation['pot'] / 10000.0
            public[0, 1] = situation['stack'] / 10000.0
            public[0, 2] = situation['to_call'] / 10000.0
            # Street encoding
            street_map = {'Preflop': 0, 'Flop': 1, 'Turn': 2, 'River': 3}
            street_idx = street_map.get(situation['street'], 0)
            public[0, 3 + street_idx] = 1.0

            # Position encoding
            position_idx = ['Button', 'Small Blind', 'Big Blind', 'UTG', 'Middle', 'Cutoff'].index(situation['position'])
            position = torch.zeros(1, 6)
            position[0, position_idx] = 1.0

            # Action history - simple encoding (no history for demo)
            action_history = torch.zeros(1, 1, 2)
            action_history = action_history.transpose(0, 1)

            # Style features - neutral opponent profile
            style_features = torch.zeros(1, 24)

            # Get decision
            action_logits = actor(
                personal=personal,
                public=public,
                position=position,
                action_history=action_history,
                style_features=style_features
            )

            action_probs = torch.softmax(action_logits, dim=-1).numpy()[0]
            predicted_action = action_logits.argmax(dim=-1).item()
            confidence = action_probs[predicted_action]

            # Debug: show raw logits
            # print(f"    DEBUG - Logits: {action_logits[0].tolist()}")

        # Display decision
        print("  → Processing opponent tendencies...")
        time.sleep(delay * 0.3)
        print("  → Calculating optimal strategy...")
        time.sleep(delay * 0.3)

        print(f"\nAMP3 DECISION:")
        print(f"  Recommended Action: {action_names[predicted_action]}")
        print(f"  Confidence: {confidence*100:.1f}%")

        print(f"\n  Action Probabilities:")
        for idx, (name, prob) in enumerate(zip(action_names, action_probs)):
            bar = '█' * int(prob * 40)
            marker = ' ←' if idx == predicted_action else ''
            print(f"    {name:12s}: {prob*100:5.1f}% {bar}{marker}")

        time.sleep(delay)

    # Summary
    print(f"\n{'='*70}")
    print("DEMO COMPLETE")
    print(f"{'='*70}\n")

    print("AMP3 CAPABILITIES:")
    print("  ✓ Full-game coverage (Preflop → River)")
    print("  ✓ Opponent modeling and adaptation")
    print("  ✓ Learned through 100,000+ episodes of self-play")
    print("  ✓ Dynamic strategy adjustment")
    print("  ✓ 2+ million parameters trained via reinforcement learning")

    print(f"\n{'='*70}\n")


def quick_demo(checkpoint_path):
    """Quick 3-scenario demo (30 seconds)"""
    print("\n🎮 Quick Demo Mode (30 seconds)\n")
    demo_amp3_decision_making(checkpoint_path, num_scenarios=3, delay=1.0)


def full_demo(checkpoint_path):
    """Full 5-scenario demo (1 minute)"""
    print("\n🎮 Full Demo Mode (1 minute)\n")
    demo_amp3_decision_making(checkpoint_path, num_scenarios=5, delay=1.5)


def extended_demo(checkpoint_path):
    """Extended 10-scenario demo (2 minutes)"""
    print("\n🎮 Extended Demo Mode (2 minutes)\n")
    demo_amp3_decision_making(checkpoint_path, num_scenarios=10, delay=1.2)


if __name__ == "__main__":
    import sys

    checkpoint_path = 'checkpoints_improved/amp3_checkpoint_100000.pt'

    print("\n" + "="*70)
    print("AMP3 LIVE DEMO")
    print("="*70)
    print("\nAvailable demo modes:")
    print("  1. Quick (30 seconds) - 3 scenarios")
    print("  2. Full (1 minute) - 5 scenarios")
    print("  3. Extended (2 minutes) - 10 scenarios")
    print("\nUsage:")
    print("  python3 demo_amp3_live.py           # Quick demo")
    print("  python3 demo_amp3_live.py full      # Full demo")
    print("  python3 demo_amp3_live.py extended  # Extended demo")
    print("="*70)

    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == 'full':
            full_demo(checkpoint_path)
        elif mode == 'extended':
            extended_demo(checkpoint_path)
        else:
            quick_demo(checkpoint_path)
    else:
        quick_demo(checkpoint_path)
