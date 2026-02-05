#!/usr/bin/env python3
"""
Quick Evaluation Script for AMP3 and Later-Street Models
Evaluates models in head-to-head play without requiring full dataset
"""

import torch
import numpy as np
from poker_env import PokerEnvironment
from poker_core import Action, Street
from style_library import StyleLibrary
from amp3_network import AMP3Agent, encode_amp3_state, state_to_tensors
from preflop_imitation import LaterStreetNetwork

def evaluate_amp3_agent(checkpoint_path, num_games=1000):
    """Evaluate AMP3 agent through self-play"""
    print(f"\n{'='*70}")
    print(f"Evaluating AMP3 Agent: {checkpoint_path}")
    print(f"{'='*70}\n")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"Checkpoint loaded from episode: {checkpoint.get('episode', 'unknown')}")

    # Create agent
    agent = AMP3Agent(
        osm_model=None,
        actor=None,
        critic=None,
        actor_lr=0.0001,
        critic_lr=0.0001,
        gamma=0.99
    )

    # Load weights
    if 'actor' in checkpoint:
        agent.actor.load_state_dict(checkpoint['actor'])
        agent.critic.load_state_dict(checkpoint['critic'])
        print("✓ Loaded actor and critic weights")
    elif 'actor_state_dict' in checkpoint:
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        print("✓ Loaded actor and critic weights")
    else:
        print("✗ Warning: Checkpoint format not recognized")
        print(f"Available keys: {list(checkpoint.keys())}")
        return

    agent.actor.eval()
    agent.critic.eval()

    # Create environment
    env = PokerEnvironment(num_players=2, big_blind=100, starting_stack=10000)
    style_library = StyleLibrary()

    # Dummy OSM predictions
    opponent_styles = np.array([[0.25, 0.15, 0.5, 0.3] for _ in range(6)])

    # Run games
    wins = 0
    total_reward = 0
    actions_taken = {0: 0, 1: 0, 2: 0, 3: 0}  # FOLD, CALL, BET, ALL_IN

    print(f"\nSimulating {num_games} heads-up games...")

    for game in range(num_games):
        state = env.reset()
        game_reward = 0

        while not state.is_terminal:
            if state.current_player == 0:  # Agent's turn
                # Encode state
                amp3_state = encode_amp3_state(state, 0, opponent_styles)
                state_dict = state_to_tensors(amp3_state)

                # Get action (no gradient needed)
                with torch.no_grad():
                    action_enum, amount, log_prob = agent.get_action(
                        state, 0, opponent_styles
                    )

                # Track actions
                action_map = {Action.FOLD: 0, Action.CALL: 1, Action.BET: 2, Action.ALL_IN: 3}
                actions_taken[action_map.get(action_enum, 1)] += 1

                # Step environment
                next_state, reward, done = env.step(action_enum, amount)
                game_reward += reward
                state = next_state
            else:
                # Opponent random action
                legal_actions = state.get_legal_actions()
                if legal_actions:
                    # Pick a random legal action
                    action, min_amt, max_amt = legal_actions[np.random.randint(len(legal_actions))]
                    # Use appropriate amount
                    if action == Action.FOLD:
                        amount = 0
                    elif action == Action.CALL:
                        amount = min_amt
                    elif action == Action.BET:
                        amount = min(max_amt, state.big_blind * 3)  # 3BB bet
                    else:  # ALL_IN
                        amount = max_amt
                    state, _, _ = env.step(action, amount)
                else:
                    break

        # Check if won
        if 0 in state.winners:
            wins += 1

        total_reward += game_reward

        if (game + 1) % 200 == 0:
            print(f"  Games {game+1}/{num_games} | Win rate: {wins/(game+1):.1%} | Avg reward: {total_reward/(game+1):.1f}")

    # Results
    win_rate = wins / num_games
    avg_reward = total_reward / num_games

    print(f"\n{'='*70}")
    print("RESULTS:")
    print(f"{'='*70}")
    print(f"Win Rate:     {win_rate:.1%}")
    print(f"Avg Reward:   {avg_reward:.2f} chips/game")
    print(f"BB/100:       {(avg_reward / 100) * 100:.1f}")
    print(f"\nAction Distribution:")
    total_actions = sum(actions_taken.values())
    print(f"  FOLD:   {actions_taken[0]/total_actions:.1%} ({actions_taken[0]} times)")
    print(f"  CALL:   {actions_taken[1]/total_actions:.1%} ({actions_taken[1]} times)")
    print(f"  BET:    {actions_taken[2]/total_actions:.1%} ({actions_taken[2]} times)")
    print(f"  ALL_IN: {actions_taken[3]/total_actions:.1%} ({actions_taken[3]} times)")
    print(f"{'='*70}\n")

    return {
        'win_rate': win_rate,
        'avg_reward': avg_reward,
        'bb_100': (avg_reward / 100) * 100,
        'actions': actions_taken
    }


def evaluate_later_streets(checkpoint_path, num_games=500):
    """Evaluate later-street models"""
    print(f"\n{'='*70}")
    print(f"Evaluating Later-Street Models: {checkpoint_path}")
    print(f"{'='*70}\n")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'flop' not in checkpoint and 'flop_state_dict' not in checkpoint:
        print("✗ Checkpoint doesn't contain later-street models")
        print(f"Available keys: {list(checkpoint.keys())}")
        return

    # Create networks
    flop_net = LaterStreetNetwork()
    turn_net = LaterStreetNetwork()
    river_net = LaterStreetNetwork()

    # Load weights
    if 'flop' in checkpoint:
        flop_net.load_state_dict(checkpoint['flop'])
        turn_net.load_state_dict(checkpoint['turn'])
        river_net.load_state_dict(checkpoint['river'])
    else:
        flop_net.load_state_dict(checkpoint['flop_state_dict'])
        turn_net.load_state_dict(checkpoint['turn_state_dict'])
        river_net.load_state_dict(checkpoint['river_state_dict'])

    print("✓ Loaded Flop, Turn, and River networks")

    flop_net.eval()
    turn_net.eval()
    river_net.eval()

    # Simple evaluation: accuracy on random states
    env = PokerEnvironment(num_players=2, big_blind=100, starting_stack=10000)

    correct_by_street = {Street.FLOP: 0, Street.TURN: 0, Street.RIVER: 0}
    total_by_street = {Street.FLOP: 0, Street.TURN: 0, Street.RIVER: 0}

    print(f"Testing on {num_games} game scenarios...\n")

    for game in range(num_games):
        state = env.reset()

        # Fast-forward to post-flop by dealing cards
        while state.street == Street.PREFLOP and not state.is_terminal:
            legal_actions = state.get_legal_actions()
            if legal_actions:
                # Try to call to see flop
                call_action = [(a, min_a, max_a) for a, min_a, max_a in legal_actions if a == Action.CALL]
                if call_action:
                    action, min_amt, max_amt = call_action[0]
                    state, _, _ = env.step(action, min_amt)
                else:
                    action, min_amt, max_amt = legal_actions[0]
                    state, _, _ = env.step(action, min_amt if action != Action.BET else min(max_amt, state.big_blind * 2))
            else:
                break

        # Test decisions at each street
        while not state.is_terminal and state.street in [Street.FLOP, Street.TURN, Street.RIVER]:
            current_street = state.street

            # Get network prediction
            if current_street == Street.FLOP:
                net = flop_net
            elif current_street == Street.TURN:
                net = turn_net
            else:
                net = river_net

            # Encode state (simplified)
            with torch.no_grad():
                # Dummy features for now
                personal = torch.zeros(1, 8)
                public = torch.zeros(1, 22)
                position = torch.zeros(1, 6)
                position[0, state.current_player] = 1

                logits = net(personal, public, position)
                pred_action = logits.argmax(dim=-1).item()

            # Random "correct" action (since we don't have labels)
            # In real evaluation, you'd compare against expert play
            correct_action = np.random.randint(0, 4)

            if pred_action == correct_action:
                correct_by_street[current_street] += 1
            total_by_street[current_street] += 1

            # Take a random action to continue
            legal_actions = state.get_legal_actions()
            if legal_actions:
                action, min_amt, max_amt = legal_actions[np.random.randint(len(legal_actions))]
                amount = min_amt if action != Action.BET else min(max_amt, state.big_blind * 2)
                state, _, _ = env.step(action, amount)
            else:
                break

    print(f"{'='*70}")
    print("RESULTS (Note: Random baseline, not true accuracy):")
    print(f"{'='*70}")
    for street in [Street.FLOP, Street.TURN, Street.RIVER]:
        if total_by_street[street] > 0:
            acc = correct_by_street[street] / total_by_street[street]
            print(f"{street.name:6s} - {total_by_street[street]:4d} decisions, {acc:.1%} match rate")
        else:
            print(f"{street.name:6s} - No decisions made")
    print(f"{'='*70}\n")

    return total_by_street


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 evaluate_all_models.py amp3 <checkpoint_path>")
        print("  python3 evaluate_all_models.py streets <checkpoint_path>")
        sys.exit(1)

    model_type = sys.argv[1]
    checkpoint_path = sys.argv[2]

    if model_type == "amp3":
        evaluate_amp3_agent(checkpoint_path, num_games=1000)
    elif model_type == "streets":
        evaluate_later_streets(checkpoint_path, num_games=500)
    else:
        print(f"Unknown model type: {model_type}")
        print("Use 'amp3' or 'streets'")
