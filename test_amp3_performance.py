#!/usr/bin/env python3
"""
AMP3 Performance Test - Real Poker Games
Tests win rate, profitability, and decision quality
"""

import torch
import numpy as np
from poker_env import PokerEnvironment
from poker_core import Action, Street
from amp3_network import AMP3Actor
import sys

def test_amp3_performance(checkpoint_path, num_games=500):
    """Test AMP3 performance through real poker games"""
    print(f"\n{'='*70}")
    print(f"AMP3 PERFORMANCE TEST: {num_games} Games")
    print(f"{'='*70}\n")

    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    episode = checkpoint.get('episode', 0) + 1

    print(f"✓ Episode {episode:,}")
    print(f"✓ Progress: {episode / 120000 * 100:.1f}%")

    # Initialize actor
    actor = AMP3Actor()
    actor.load_state_dict(checkpoint['actor'])
    actor.eval()

    actor_params = sum(p.numel() for p in actor.parameters())
    print(f"✓ Actor parameters: {actor_params:,}")

    # Initialize environment
    env = PokerEnvironment(num_players=2, big_blind=100, starting_stack=10000)
    print(f"✓ Environment initialized")

    # Game statistics
    games_won = 0
    games_lost = 0
    games_tie = 0
    total_reward = 0

    action_counts = {
        Action.FOLD: 0,
        Action.CALL: 0,
        Action.BET: 0,
        Action.ALL_IN: 0
    }

    street_counts = {
        Street.PREFLOP: 0,
        Street.FLOP: 0,
        Street.TURN: 0,
        Street.RIVER: 0
    }

    errors = 0

    print(f"\nPlaying {num_games} games against baseline opponent...")
    print("(Baseline strategy: 70% call, 20% raise, 10% fold)\n")

    for game_num in range(num_games):
        state = env.reset()
        game_reward = 0
        steps = 0
        max_steps = 100

        while not state.is_terminal and steps < max_steps:
            current_player = state.current_player

            if current_player != 0:
                # Opponent turn - simple baseline strategy
                legal_actions = state.get_legal_actions()
                if not legal_actions:
                    break

                # Baseline: 70% call, 20% raise, 10% fold
                rand = np.random.random()
                if rand < 0.7:
                    # Try to call
                    call_actions = [a for a in legal_actions if a[0] == Action.CALL]
                    if call_actions:
                        action, min_amt, max_amt = call_actions[0]
                    else:
                        action, min_amt, max_amt = legal_actions[0]
                elif rand < 0.9:
                    # Try to raise/bet
                    raise_actions = [a for a in legal_actions if a[0] == Action.BET]
                    if raise_actions:
                        action, min_amt, max_amt = raise_actions[0]
                        min_amt = min(max_amt, min_amt * 2)  # Small raise
                    else:
                        action, min_amt, max_amt = legal_actions[0]
                else:
                    # Try to fold
                    fold_actions = [a for a in legal_actions if a[0] == Action.FOLD]
                    if fold_actions:
                        action, min_amt, max_amt = fold_actions[0]
                    else:
                        action, min_amt, max_amt = legal_actions[0]

                state, rewards, done = env.step(action, min_amt)

            else:
                # AMP3's turn
                try:
                    # Track decisions by street
                    if state.street in street_counts:
                        street_counts[state.street] += 1

                    # Get legal actions
                    legal_actions = state.get_legal_actions()
                    if not legal_actions:
                        break

                    # Create inputs for actor
                    # Personal features (8): simplified state encoding
                    personal = torch.zeros(1, 8)
                    personal[0, 0] = state.current_player / 6.0  # normalized position
                    personal[0, 1] = state.pot / 10000.0  # normalized pot
                    personal[0, 2] = state.stacks[0] / 10000.0  # normalized stack

                    # Public features (22): simplified
                    public = torch.zeros(1, 22)
                    public[0, 0] = state.pot / 10000.0

                    # Position encoding
                    position = torch.zeros(1, 6)
                    position[0, state.current_player % 6] = 1.0

                    # Action history - use simple encoding
                    action_history = torch.zeros(1, 1, 2)  # batch first for simpler handling
                    action_history = action_history.transpose(0, 1)  # -> (seq=1, batch=1, features=2)

                    # Style features
                    style_features = torch.zeros(1, 24)

                    # Get action from actor
                    with torch.no_grad():
                        action_logits = actor(
                            personal=personal,
                            public=public,
                            position=position,
                            action_history=action_history,
                            style_features=style_features
                        )

                        action_probs = torch.softmax(action_logits, dim=-1)
                        predicted_action_idx = action_logits.argmax(dim=-1).item()

                    # Map to poker action
                    action_map = [Action.FOLD, Action.CALL, Action.BET, Action.BET]
                    predicted_action = action_map[predicted_action_idx]

                    # Track action
                    if predicted_action in action_counts:
                        action_counts[predicted_action] += 1

                    # Find legal action that matches prediction
                    matching = [a for a in legal_actions if a[0] == predicted_action]
                    if matching:
                        action, min_amt, max_amt = matching[0]
                        amount = min_amt
                        if action in [Action.BET, Action.RAISE]:
                            amount = min(max_amt, min_amt * 2)  # 2x minimum
                    else:
                        # Fall back to first legal action
                        action, min_amt, max_amt = legal_actions[0]
                        amount = min_amt

                    # Execute action
                    state, rewards, done = env.step(action, amount)
                    game_reward += rewards.get(0, 0)

                except Exception as e:
                    errors += 1
                    if errors <= 3:  # Only print first few errors
                        print(f"\n⚠️ Error in game {game_num}: {e}")
                    # Fall back to random legal action
                    legal_actions = state.get_legal_actions()
                    if legal_actions:
                        action, min_amt, max_amt = legal_actions[np.random.randint(len(legal_actions))]
                        amount = min_amt if action != Action.BET else min(max_amt, state.big_blind * 2)
                        state, rewards, done = env.step(action, amount)
                    break

            steps += 1

        # Record game result
        if game_reward > 0:
            games_won += 1
        elif game_reward < 0:
            games_lost += 1
        else:
            games_tie += 1

        total_reward += game_reward

        # Progress update
        if (game_num + 1) % 100 == 0:
            current_wr = games_won / (game_num + 1) * 100
            avg_rew = total_reward / (game_num + 1)
            print(f"  Games {game_num+1:3d}/{num_games} | WR: {current_wr:5.1f}% | Avg: {avg_rew:+7.1f} chips")

    # Calculate final metrics
    print(f"\n{'='*70}")
    print("PERFORMANCE RESULTS")
    print(f"{'='*70}\n")

    win_rate = games_won / num_games * 100
    loss_rate = games_lost / num_games * 100
    tie_rate = games_tie / num_games * 100
    avg_reward = total_reward / num_games
    bb_per_100 = (avg_reward / 100) * 100

    print(f"Games Played:  {num_games}")
    print(f"  Won:         {games_won:3d} ({win_rate:5.1f}%)")
    print(f"  Lost:        {games_lost:3d} ({loss_rate:5.1f}%)")
    print(f"  Tie:         {games_tie:3d} ({tie_rate:5.1f}%)")

    print(f"\nProfitability:")
    print(f"  Total reward:    {total_reward:+,.0f} chips")
    print(f"  Average reward:  {avg_reward:+.2f} chips/game")
    print(f"  BB/100:          {bb_per_100:+.2f}")

    print(f"\nAction Distribution:")
    total_actions = sum(action_counts.values())
    for action in [Action.FOLD, Action.CALL, Action.BET, Action.ALL_IN]:
        if action in action_counts:
            count = action_counts[action]
            pct = count / total_actions * 100 if total_actions > 0 else 0
            bar = '█' * int(pct / 2)
            print(f"  {action.name:8s}: {count:5d} ({pct:5.1f}%) {bar}")

    print(f"\nDecisions by Street:")
    total_decisions = sum(street_counts.values())
    for street in [Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER]:
        if street in street_counts:
            count = street_counts[street]
            pct = count / total_decisions * 100 if total_decisions > 0 else 0
            print(f"  {street.name:8s}: {count:5d} ({pct:5.1f}%)")

    if errors > 0:
        print(f"\nErrors encountered: {errors} ({errors/num_games*100:.1f}%)")

    # Performance assessment
    print(f"\n{'='*70}")
    print("ASSESSMENT")
    print(f"{'='*70}\n")

    if win_rate >= 60:
        print("✅ EXCELLENT - Dominating baseline opponent!")
    elif win_rate >= 55:
        print("✅ STRONG - Significantly better than baseline!")
    elif win_rate >= 52:
        print("✅ GOOD - Consistently beating baseline!")
    elif win_rate >= 48:
        print("⚠️  COMPETITIVE - Close to baseline performance")
    else:
        print("⚠️  NEEDS IMPROVEMENT - Below baseline performance")

    if bb_per_100 > 10:
        print("✅ Highly profitable (>10 BB/100)")
    elif bb_per_100 > 5:
        print("✅ Very profitable (>5 BB/100)")
    elif bb_per_100 > 0:
        print("✅ Profitable")
    elif bb_per_100 > -5:
        print("⚠️ Slightly unprofitable")
    else:
        print("⚠️ Significantly unprofitable")

    print(f"\n{'='*70}\n")

    return {
        'episode': episode,
        'win_rate': win_rate,
        'avg_reward': avg_reward,
        'bb_100': bb_per_100,
        'action_counts': action_counts,
        'street_counts': street_counts,
        'errors': errors
    }


if __name__ == "__main__":
    checkpoint_path = 'checkpoints_improved/amp3_checkpoint_100000.pt'

    if len(sys.argv) > 1:
        num_games = int(sys.argv[1])
    else:
        num_games = 500

    results = test_amp3_performance(checkpoint_path, num_games=num_games)

    print(f"Test complete!")
    print(f"Win Rate: {results['win_rate']:.1f}%")
    print(f"BB/100: {results['bb_100']:+.2f}")
