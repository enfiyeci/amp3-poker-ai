#!/usr/bin/env python3
"""
Evaluate AMP3 checkpoint at 100k episodes
Safe to run while training continues
"""

import torch
import numpy as np
from poker_env import PokerEnvironment
from poker_core import Action, Street
from amp3_network import AMP3Agent
from style_library import StyleLibrary

def evaluate_checkpoint(checkpoint_path, num_games=1000):
    """Evaluate AMP3 checkpoint through game simulation"""
    print(f"\n{'='*70}")
    print(f"EVALUATING AMP3 CHECKPOINT: {checkpoint_path}")
    print(f"{'='*70}\n")

    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    episode = checkpoint.get('episode', 'unknown')
    print(f"✓ Loaded checkpoint from episode {episode}")

    # Initialize agent
    print("Initializing AMP3 agent...")
    style_library = StyleLibrary()
    agent = AMP3Agent(style_library=style_library)

    # Load weights
    agent.actor.load_state_dict(checkpoint['actor'])
    agent.critic.load_state_dict(checkpoint['critic'])
    agent.actor.eval()
    agent.critic.eval()
    print(f"✓ Agent initialized with {agent.actor.count_parameters():,} actor parameters")

    # Initialize environment
    env = PokerEnvironment(num_players=2, big_blind=100, starting_stack=10000)

    # Statistics
    games_won = 0
    total_reward = 0
    action_counts = {Action.FOLD: 0, Action.CALL: 0, Action.BET: 0, Action.RAISE: 0}
    street_decisions = {Street.PREFLOP: 0, Street.FLOP: 0, Street.TURN: 0, Street.RIVER: 0}

    print(f"\nSimulating {num_games} games...")

    for game_num in range(num_games):
        state = env.reset()
        game_reward = 0
        done = False
        steps = 0
        max_steps = 100

        while not done and steps < max_steps:
            if state.current_player != 0:
                # Opponent's turn - simple baseline
                legal_actions = state.get_legal_actions()
                if not legal_actions:
                    break

                # Simple opponent strategy: 60% call, 40% fold
                if np.random.random() < 0.6:
                    call_actions = [a for a in legal_actions if a[0] == Action.CALL]
                    if call_actions:
                        action, min_amt, _ = call_actions[0]
                        state, rewards, done = env.step(action, min_amt)
                    else:
                        action, min_amt, _ = legal_actions[0]
                        state, rewards, done = env.step(action, min_amt)
                else:
                    fold_actions = [a for a in legal_actions if a[0] == Action.FOLD]
                    if fold_actions:
                        action, min_amt, _ = fold_actions[0]
                        state, rewards, done = env.step(action, min_amt)
                    else:
                        action, min_amt, _ = legal_actions[0]
                        state, rewards, done = env.step(action, min_amt)
            else:
                # AMP3's turn
                try:
                    # Track decisions by street
                    if state.street in street_decisions:
                        street_decisions[state.street] += 1

                    # Get action from agent
                    action_enum, amount, _ = agent.get_action(
                        state=state,
                        player_id=0,
                        training=False
                    )

                    # Track action
                    if action_enum in action_counts:
                        action_counts[action_enum] += 1

                    # Execute action
                    state, rewards, done = env.step(action_enum, amount)
                    game_reward += rewards.get(0, 0)

                except Exception as e:
                    print(f"\n⚠️ Error in game {game_num}: {e}")
                    break

            steps += 1

        if game_reward > 0:
            games_won += 1
        total_reward += game_reward

        if (game_num + 1) % 200 == 0:
            print(f"  Games {game_num+1}/{num_games} | Win rate: {games_won/(game_num+1)*100:.1f}% | Avg reward: {total_reward/(game_num+1):.1f}")

    # Results
    print(f"\n{'='*70}")
    print("EVALUATION RESULTS")
    print(f"{'='*70}\n")

    win_rate = games_won / num_games * 100
    avg_reward = total_reward / num_games
    bb_per_100 = (avg_reward / 100) * 100

    print(f"Games played: {num_games}")
    print(f"Games won: {games_won}")
    print(f"Win rate: {win_rate:.1f}%")
    print(f"Average reward: {avg_reward:.2f} chips")
    print(f"BB/100: {bb_per_100:.2f}")

    print(f"\nAction Distribution:")
    total_actions = sum(action_counts.values())
    for action, count in action_counts.items():
        pct = count / total_actions * 100 if total_actions > 0 else 0
        print(f"  {action.name:8s}: {count:5d} ({pct:.1f}%)")

    print(f"\nDecisions by Street:")
    total_decisions = sum(street_decisions.values())
    for street, count in street_decisions.items():
        pct = count / total_decisions * 100 if total_decisions > 0 else 0
        print(f"  {street.name:8s}: {count:5d} ({pct:.1f}%)")

    # Quality assessment
    print(f"\n{'='*70}")
    if win_rate >= 55:
        print("✅ STRONG PERFORMANCE - Above 55% win rate")
    elif win_rate >= 50:
        print("✅ GOOD PERFORMANCE - Above 50% win rate")
    elif win_rate >= 45:
        print("⚠️ DECENT PERFORMANCE - Competitive but room for improvement")
    else:
        print("⚠️ NEEDS IMPROVEMENT - Below 45% win rate")
    print(f"{'='*70}\n")

    return {
        'episode': episode,
        'win_rate': win_rate,
        'avg_reward': avg_reward,
        'bb_100': bb_per_100,
        'action_dist': action_counts,
        'street_decisions': street_decisions
    }


if __name__ == "__main__":
    checkpoint_path = 'checkpoints_improved/amp3_checkpoint_100000.pt'
    results = evaluate_checkpoint(checkpoint_path, num_games=1000)
