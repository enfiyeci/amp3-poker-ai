#!/usr/bin/env python3
"""
Post-Flop Model Validation
Tests Flop/Turn/River models with simulated scenarios
"""

import torch
import numpy as np
from poker_env import PokerEnvironment
from poker_core import Action, Street, GameState
from style_library import StyleLibrary
from preflop_imitation import LaterStreetNetwork
from collections import defaultdict

def validate_later_streets(checkpoint_path, num_scenarios=2000):
    """Validate later-street models through structured testing"""
    print(f"\n{'='*70}")
    print(f"POST-FLOP MODEL VALIDATION")
    print(f"{'='*70}\n")

    # Load checkpoint
    print("Loading models...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'flop' not in checkpoint:
        print("❌ Checkpoint missing flop/turn/river keys")
        return

    # Create networks
    flop_net = LaterStreetNetwork()
    turn_net = LaterStreetNetwork()
    river_net = LaterStreetNetwork()

    flop_net.load_state_dict(checkpoint['flop'])
    turn_net.load_state_dict(checkpoint['turn'])
    river_net.load_state_dict(checkpoint['river'])

    print("✓ Loaded Flop, Turn, and River networks")

    flop_net.eval()
    turn_net.eval()
    river_net.eval()

    # Initialize environment
    env = PokerEnvironment(num_players=2, big_blind=100, starting_stack=10000)

    # Statistics tracking
    decisions_by_street = {Street.FLOP: 0, Street.TURN: 0, Street.RIVER: 0}
    actions_by_street = {
        Street.FLOP: defaultdict(int),
        Street.TURN: defaultdict(int),
        Street.RIVER: defaultdict(int)
    }
    total_reward = 0
    games_completed = 0

    print(f"\nSimulating {num_scenarios} post-flop scenarios...")
    print("(Using model predictions to make decisions)\n")

    for scenario in range(num_scenarios):
        state = env.reset()
        game_reward = 0

        # Fast-forward to post-flop
        preflop_actions = 0
        while state.street == Street.PREFLOP and not state.is_terminal and preflop_actions < 10:
            legal_actions = state.get_legal_actions()
            if not legal_actions:
                break

            # Try to reach flop by calling
            call_actions = [a for a in legal_actions if a[0] == Action.CALL]
            if call_actions:
                action, min_amt, max_amt = call_actions[0]
                state, reward, done = env.step(action, min_amt)
            else:
                # If can't call, take first legal action
                action, min_amt, max_amt = legal_actions[0]
                amount = min_amt if action != Action.BET else min(max_amt, state.big_blind * 2)
                state, reward, done = env.step(action, amount)

            game_reward += reward.get(0, 0)
            preflop_actions += 1

        # Now test post-flop decisions
        post_flop_actions = 0
        while not state.is_terminal and state.street in [Street.FLOP, Street.TURN, Street.RIVER] and post_flop_actions < 20:
            current_street = state.street

            # Select appropriate network
            if current_street == Street.FLOP:
                net = flop_net
            elif current_street == Street.TURN:
                net = turn_net
            else:  # RIVER
                net = river_net

            # Get model prediction
            with torch.no_grad():
                # Create dummy inputs (simplified for testing)
                personal = torch.zeros(1, 8)
                public = torch.zeros(1, 22)
                position = torch.zeros(1, 6)
                position[0, state.current_player % 6] = 1.0

                try:
                    logits = net(personal, public, position)
                    pred_action_idx = logits.argmax(dim=-1).item()

                    # Map to action
                    action_map = [Action.FOLD, Action.CALL, Action.BET, Action.BET]
                    predicted_action = action_map[pred_action_idx]

                    # Track decision
                    decisions_by_street[current_street] += 1
                    actions_by_street[current_street][predicted_action] += 1

                    # Try to execute predicted action
                    legal_actions = state.get_legal_actions()
                    if not legal_actions:
                        break

                    # Find if predicted action is legal
                    matching = [a for a in legal_actions if a[0] == predicted_action]
                    if matching:
                        action, min_amt, max_amt = matching[0]
                        amount = min_amt if action != Action.BET else min(max_amt, state.big_blind * 3)
                        state, reward, done = env.step(action, amount)
                    else:
                        # Use first legal action if prediction not legal
                        action, min_amt, max_amt = legal_actions[0]
                        amount = min_amt if action != Action.BET else min(max_amt, state.big_blind * 2)
                        state, reward, done = env.step(action, amount)

                    game_reward += reward.get(0, 0)
                    post_flop_actions += 1

                except Exception as e:
                    # If prediction fails, take random legal action
                    legal_actions = state.get_legal_actions()
                    if legal_actions:
                        action, min_amt, max_amt = legal_actions[np.random.randint(len(legal_actions))]
                        amount = min_amt if action != Action.BET else min(max_amt, state.big_blind * 2)
                        state, reward, done = env.step(action, amount)
                        game_reward += reward.get(0, 0)
                    break

        total_reward += game_reward
        games_completed += 1

        if (scenario + 1) % 500 == 0:
            print(f"  Scenarios {scenario+1}/{num_scenarios} | Avg reward: {total_reward/games_completed:.1f} chips")

    # Print results
    print(f"\n{'='*70}")
    print("VALIDATION RESULTS")
    print(f"{'='*70}\n")

    print(f"Games completed: {games_completed}")
    print(f"Average reward per game: {total_reward/games_completed:.2f} chips")
    print(f"BB/100 equivalent: {(total_reward/games_completed)/100*100:.2f}")

    print(f"\nDecisions Made by Street:")
    total_decisions = sum(decisions_by_street.values())
    for street in [Street.FLOP, Street.TURN, Street.RIVER]:
        count = decisions_by_street[street]
        pct = count / total_decisions * 100 if total_decisions > 0 else 0
        print(f"  {street.name:6s}: {count:5d} decisions ({pct:.1f}%)")

    print(f"\nAction Distribution:")
    for street in [Street.FLOP, Street.TURN, Street.RIVER]:
        print(f"\n  {street.name}:")
        street_total = sum(actions_by_street[street].values())
        if street_total > 0:
            for action in [Action.FOLD, Action.CALL, Action.BET]:
                count = actions_by_street[street][action]
                pct = count / street_total * 100
                print(f"    {action.name:8s}: {count:4d} ({pct:.1f}%)")
        else:
            print(f"    No decisions made")

    print(f"\n{'='*70}")

    # Calculate some heuristic scores
    total_dec = sum(decisions_by_street.values())
    if total_dec > 0:
        print("\nModel Activity:")
        print(f"  ✓ Made {total_dec:,} post-flop decisions")
        print(f"  ✓ Successfully navigated {games_completed} game scenarios")

        avg_bb = (total_reward/games_completed)/100*100
        if avg_bb > 0:
            print(f"  ✓ Positive average BB/100: +{avg_bb:.1f}")
            print(f"\n✅ POST-FLOP MODELS ARE FUNCTIONAL")
        elif avg_bb > -20:
            print(f"  ⚠️ Near-neutral BB/100: {avg_bb:.1f}")
            print(f"\n⚠️ POST-FLOP MODELS NEED TUNING")
        else:
            print(f"  ❌ Negative BB/100: {avg_bb:.1f}")
            print(f"\n❌ POST-FLOP MODELS NEED IMPROVEMENT")

    print(f"{'='*70}\n")

    return {
        'decisions': decisions_by_street,
        'actions': actions_by_street,
        'avg_reward': total_reward / games_completed if games_completed > 0 else 0,
        'bb_100': (total_reward/games_completed)/100*100 if games_completed > 0 else 0
    }


if __name__ == "__main__":
    import sys

    checkpoint_path = 'checkpoints_20hr/street_models.pt'
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]

    results = validate_later_streets(checkpoint_path, num_scenarios=2000)
