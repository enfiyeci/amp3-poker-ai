#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for AMP3 Poker AI

Evaluates trained models using multiple metrics:
1. Classification metrics (accuracy, precision, recall, F1)
2. Poker-specific metrics (win rate, exploitability, expected value)
3. Head-to-head performance against baseline strategies
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import defaultdict
from typing import Dict, List, Tuple

# Import AMP3 components
from train_with_real_data import AMP3Policy, PokerDataset, load_data_from_csv
from poker_env import PokerEnvironment
from poker_core import Action, Street
from style_library import StyleLibrary
from preflop_imitation import PreflopImitationNetwork


# =============================================================================
# Classification Metrics
# =============================================================================

def evaluate_classification(model, val_loader, device='cpu'):
    """
    Evaluate classification performance on validation set.

    Returns:
        dict: Classification metrics including accuracy, precision, recall, F1
    """
    model.eval()
    all_true = []
    all_pred = []
    all_probs = []

    with torch.no_grad():
        for state, style, labels in val_loader:
            state = state.to(device)
            style = style.to(device)

            logits = model(state, style)
            probs = torch.softmax(logits, dim=-1)
            pred = logits.argmax(dim=-1)

            all_true.append(labels.numpy())
            all_pred.append(pred.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)
    all_probs = np.concatenate(all_probs)

    # Compute metrics
    action_names = ["FOLD", "CALL", "RAISE_SMALL", "RAISE_LARGE"]

    results = {
        'accuracy': accuracy_score(all_true, all_pred),
        'classification_report': classification_report(
            all_true, all_pred, target_names=action_names, output_dict=True
        ),
        'confusion_matrix': confusion_matrix(all_true, all_pred),
        'action_names': action_names,
    }

    # Per-action metrics
    results['per_action'] = {}
    for i, action in enumerate(action_names):
        mask = all_true == i
        if mask.sum() > 0:
            results['per_action'][action] = {
                'count': mask.sum(),
                'accuracy': (all_pred[mask] == i).mean(),
                'avg_confidence': all_probs[mask, i].mean(),
            }

    return results


# =============================================================================
# Expected Value (EV) Metrics
# =============================================================================

def calculate_ev_by_action(agent_model, num_scenarios=1000, device='cpu'):
    """
    Calculate expected value for each action type across various game states.

    Returns:
        dict: EV metrics for each action category
    """
    env = PokerEnvironment(num_players=2, big_blind=100, starting_stack=10000)
    agent_model.eval()

    action_evs = {
        'fold': [],
        'call': [],
        'raise_small': [],
        'raise_large': []
    }

    for _ in range(num_scenarios):
        state = env.reset(button_seat=np.random.randint(2))
        starting_stack = state.players[0].stack

        # Get model's action
        state_features = extract_state_features(state, 0)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_features['state']).unsqueeze(0).to(device)
            style_tensor = torch.FloatTensor(state_features['style']).unsqueeze(0).to(device)
            logits = agent_model(state_tensor, style_tensor)
            action_idx = logits.argmax(dim=-1).item()

        # Map action index to category
        action_names = ['fold', 'call', 'raise_small', 'raise_large']
        action_name = action_names[action_idx]

        # Simulate to end
        action, amount = decode_action(action_idx, state, 0)
        state, _, _ = env.step(action, amount)

        # Continue with random actions until terminal
        while not state.is_terminal:
            current_player = state.current_player
            valid_actions = [Action.FOLD, Action.CALL, Action.BET]
            random_action = np.random.choice(valid_actions)
            state, _, _ = env.step(random_action, 0)

        # Calculate EV
        final_stack = state.players[0].stack
        ev = final_stack - starting_stack
        action_evs[action_name].append(ev)

    # Calculate average EVs
    results = {}
    for action, evs in action_evs.items():
        if evs:
            results[action] = {
                'mean_ev': np.mean(evs),
                'std_ev': np.std(evs),
                'count': len(evs),
                'ev_bb': np.mean(evs) / 100  # in big blinds
            }
        else:
            results[action] = {
                'mean_ev': 0,
                'std_ev': 0,
                'count': 0,
                'ev_bb': 0
            }

    return results


def calculate_exploitability(agent_model, num_hands=5000, device='cpu'):
    """
    Estimate exploitability by measuring performance against counter-strategies.

    Lower exploitability means the strategy is harder to exploit.

    Returns:
        dict: Exploitability metrics
    """
    from style_library import StyleLibrary
    style_library = StyleLibrary()

    # Test against exploitative strategies
    exploiters = {
        'conservative': style_library.get_strategy('Sklansky_CONSERVATIVE'),
        'aggressive': style_library.get_strategy('Sklansky_AGGRESSIVE'),
        'bluffing': style_library.get_strategy('Sklansky_BLUFFING'),
    }

    exploiter_results = {}

    for name, strategy in exploiters.items():
        perf = simulate_hands(agent_model, strategy, num_hands=num_hands // len(exploiters), device=device)
        exploiter_results[name] = perf['bb_per_100']

    # Exploitability = worst-case performance against exploiters
    worst_bb100 = min(exploiter_results.values())
    avg_bb100 = np.mean(list(exploiter_results.values()))

    return {
        'worst_case_bb100': worst_bb100,
        'avg_bb100_vs_exploiters': avg_bb100,
        'exploiter_results': exploiter_results,
        'exploitability_score': max(0, -worst_bb100)  # Higher = more exploitable
    }


# =============================================================================
# Poker Performance Metrics
# =============================================================================

def simulate_hands(agent_model, opponent_strategy, num_hands=1000, device='cpu'):
    """
    Simulate poker hands between agent and opponent.

    Returns:
        dict: Performance metrics (chips won, win rate, showdown stats)
    """
    env = PokerEnvironment(num_players=2, big_blind=100, starting_stack=10000)
    agent_model.eval()

    results = {
        'hands_played': 0,
        'hands_won': 0,
        'chips_won': 0,
        'showdowns_reached': 0,
        'showdowns_won': 0,
        'vpip': 0,  # Voluntarily put $ in pot
        'pfr': 0,   # Pre-flop raise
        'aggression_factor': {'bets': 0, 'calls': 0},
    }

    agent_seat = 0  # Agent is always seat 0

    for hand in range(num_hands):
        state = env.reset(button_seat=hand % 2)
        starting_stack = state.players[agent_seat].stack

        preflop_voluntary_action = False

        while not state.is_terminal:
            current_player = state.current_player
            player = state.players[current_player]

            if current_player == agent_seat:
                # Agent's turn - use model
                state_features = extract_state_features(state, agent_seat)

                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state_features['state']).unsqueeze(0).to(device)
                    style_tensor = torch.FloatTensor(state_features['style']).unsqueeze(0).to(device)
                    logits = agent_model(state_tensor, style_tensor)
                    action_idx = logits.argmax(dim=-1).item()

                # Convert to Action
                action, amount = decode_action(action_idx, state, agent_seat)

                # Track statistics
                if state.street == Street.PREFLOP:
                    if action in [Action.CALL, Action.BET, Action.ALL_IN]:
                        preflop_voluntary_action = True
                    if action in [Action.BET, Action.ALL_IN]:
                        results['pfr'] += 1

                if action in [Action.BET, Action.ALL_IN]:
                    results['aggression_factor']['bets'] += 1
                elif action == Action.CALL:
                    results['aggression_factor']['calls'] += 1
            else:
                # Opponent's turn
                action, amount = opponent_strategy.get_action(
                    state, current_player, player.hole_cards, state.community_cards
                )

            state, _, _ = env.step(action, amount)

        # Hand finished
        results['hands_played'] += 1
        if preflop_voluntary_action:
            results['vpip'] += 1

        # Check if agent won
        final_stack = state.players[agent_seat].stack
        chips_change = final_stack - starting_stack
        results['chips_won'] += chips_change

        if chips_change > 0:
            results['hands_won'] += 1

        # Check for showdown
        if len(state.community_cards) == 5 and state.players[agent_seat].is_active:
            results['showdowns_reached'] += 1
            if chips_change > 0:
                results['showdowns_won'] += 1

    # Calculate rates
    results['win_rate'] = results['hands_won'] / results['hands_played']
    results['vpip_rate'] = results['vpip'] / results['hands_played']
    results['pfr_rate'] = results['pfr'] / results['hands_played']

    af_total = results['aggression_factor']['bets'] + results['aggression_factor']['calls']
    if af_total > 0:
        results['aggression_factor_value'] = (
            results['aggression_factor']['bets'] / af_total
        )
    else:
        results['aggression_factor_value'] = 0

    if results['showdowns_reached'] > 0:
        results['showdown_win_rate'] = (
            results['showdowns_won'] / results['showdowns_reached']
        )
    else:
        results['showdown_win_rate'] = 0

    # BB/100 (big blinds won per 100 hands)
    results['bb_per_100'] = (results['chips_won'] / 100) / (results['hands_played'] / 100)

    return results


def extract_state_features(state, player_idx):
    """Extract state and style features for model input."""
    player = state.players[player_idx]

    # State features (7)
    state_feats = np.array([
        player_idx,  # position
        state.current_bet - player.bet_this_street,  # to_call
        state.pot / 100,  # pot in BB
        player.stack / 100,  # stack in BB
        player.stack / (state.pot + 1e-6),  # SPR
        sum(1 for p in state.players if p.is_active),  # num_active
        sum(1 for a in state.action_history if a[1] == Action.BET),  # num_raises
    ], dtype=np.float32)

    # Style features (6) - placeholder values for evaluation
    style_feats = np.array([0.3, 0.2, 0.5, 0.3, 0.2, 0.5], dtype=np.float32)

    return {'state': state_feats, 'style': style_feats}


def decode_action(action_idx, state, player_idx):
    """Convert action index to Action enum and amount."""
    player = state.players[player_idx]
    to_call = state.current_bet - player.bet_this_street

    if action_idx == 0:  # Fold/Check
        if to_call > 0:
            return Action.FOLD, 0
        else:
            return Action.CALL, 0
    elif action_idx == 1:  # Call
        return Action.CALL, min(to_call, player.stack)
    elif action_idx == 2:  # Small raise
        amount = min(state.pot * 0.75 + to_call, player.stack)
        if amount > to_call:
            return Action.BET, amount
        else:
            return Action.CALL, to_call
    else:  # Large raise / All-in
        amount = min(state.pot * 1.5 + to_call, player.stack)
        if amount >= player.stack * 0.9:
            return Action.ALL_IN, player.stack
        elif amount > to_call:
            return Action.BET, amount
        else:
            return Action.CALL, to_call


# =============================================================================
# Head-to-Head Evaluation
# =============================================================================

def evaluate_vs_baselines(agent_model, num_hands=1000, device='cpu'):
    """
    Evaluate agent against baseline strategies from style library.

    Returns:
        dict: Results against each baseline
    """
    style_library = StyleLibrary()

    baselines = [
        'Sklansky_CONSERVATIVE',   # Tight/conservative play
        'Sklansky_AGGRESSIVE',      # Aggressive play
        'Sklansky_REGULAR',         # Balanced play
        'Chen_REGULAR',             # Chen formula regular
        'RuleBased_REGULAR',        # Rule-based regular
    ]

    results = {}

    for baseline_name in baselines:
        print(f"  Evaluating vs {baseline_name}...")
        baseline_strategy = style_library.get_strategy(baseline_name)

        perf = simulate_hands(agent_model, baseline_strategy, num_hands, device)
        results[baseline_name] = perf

    return results


# =============================================================================
# Main Evaluation
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate AMP3 Poker AI')
    parser.add_argument('--model_path', type=str,
                       default='checkpoints/best_model.pt',
                       help='Path to trained model')
    parser.add_argument('--data_path', type=str,
                       default='/Users/ardaenfiyeci/Downloads/preflop_demo_full.csv',
                       help='Path to validation data')
    parser.add_argument('--num_sim_hands', type=int, default=1000,
                       help='Number of hands to simulate for poker metrics')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu or cuda)')

    args = parser.parse_args()

    print("="*60)
    print("AMP3 Poker AI Evaluation")
    print("="*60)
    print()

    # Load model
    print("Loading model...")
    device = torch.device(args.device)
    model = AMP3Policy(state_dim=7, style_dim=6, hidden_dim=128, num_actions=4)

    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Model loaded from {args.model_path}")
    print()

    # =========================================================================
    # 1. Classification Metrics
    # =========================================================================

    print("="*60)
    print("1. CLASSIFICATION METRICS")
    print("="*60)
    print()

    print("Loading validation data...")
    X, y = load_data_from_csv(args.data_path)

    from sklearn.model_selection import train_test_split
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

    val_dataset = PokerDataset(X_val, y_val, state_dim=7)
    val_loader = DataLoader(val_dataset, batch_size=256)

    print(f"Validation set size: {len(val_dataset)}")
    print()

    class_results = evaluate_classification(model, val_loader, device)

    print(f"Overall Accuracy: {class_results['accuracy']:.3f}")
    print()

    print("Per-Action Performance:")
    print("-" * 60)
    for action, metrics in class_results['per_action'].items():
        print(f"{action:12} | Samples: {metrics['count']:6d} | "
              f"Accuracy: {metrics['accuracy']:.3f} | "
              f"Confidence: {metrics['avg_confidence']:.3f}")
    print()

    print("Classification Report:")
    print("-" * 60)
    report = class_results['classification_report']
    for action in class_results['action_names']:
        if action in report:
            print(f"{action:12} | Precision: {report[action]['precision']:.3f} | "
                  f"Recall: {report[action]['recall']:.3f} | "
                  f"F1: {report[action]['f1-score']:.3f}")
    print()

    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(class_results['confusion_matrix'])
    print()

    # =========================================================================
    # 2. Head-to-Head Performance
    # =========================================================================

    print("="*60)
    print("2. HEAD-TO-HEAD PERFORMANCE")
    print("="*60)
    print()

    print(f"Simulating {args.num_sim_hands} hands against baseline strategies...")
    print()

    h2h_results = evaluate_vs_baselines(model, args.num_sim_hands, device)

    print("Results:")
    print("-" * 80)
    print(f"{'Opponent':20} | {'Win Rate':>10} | {'BB/100':>10} | {'VPIP':>8} | {'AFq':>8}")
    print("-" * 80)

    for baseline, perf in h2h_results.items():
        print(f"{baseline:20} | {perf['win_rate']:>9.1%} | "
              f"{perf['bb_per_100']:>9.1f} | {perf['vpip_rate']:>7.1%} | "
              f"{perf['aggression_factor_value']:>7.2f}")

    print()

    # Average performance
    avg_win_rate = np.mean([p['win_rate'] for p in h2h_results.values()])
    avg_bb100 = np.mean([p['bb_per_100'] for p in h2h_results.values()])

    print(f"Average Performance:")
    print(f"  Win Rate: {avg_win_rate:.1%}")
    print(f"  BB/100:   {avg_bb100:.1f}")
    print()

    # =========================================================================
    # 3. Expected Value (EV) Analysis
    # =========================================================================

    print("="*60)
    print("3. EXPECTED VALUE (EV) ANALYSIS")
    print("="*60)
    print()

    print("Calculating EV by action type...")
    ev_results = calculate_ev_by_action(model, num_scenarios=1000, device=device)

    print("Expected Value by Action:")
    print("-" * 60)
    print(f"{'Action':15} | {'Mean EV (chips)':>15} | {'EV (BB)':>10} | {'Std Dev':>10} | {'Count':>6}")
    print("-" * 60)

    for action, metrics in ev_results.items():
        print(f"{action.upper():15} | {metrics['mean_ev']:>15.1f} | "
              f"{metrics['ev_bb']:>10.2f} | {metrics['std_ev']:>10.1f} | "
              f"{metrics['count']:>6d}")
    print()

    # =========================================================================
    # 4. Exploitability Analysis
    # =========================================================================

    print("="*60)
    print("4. EXPLOITABILITY ANALYSIS")
    print("="*60)
    print()

    print("Testing against counter-strategies...")
    exploit_results = calculate_exploitability(model, num_hands=3000, device=device)

    print("Performance vs Exploitative Opponents:")
    print("-" * 60)
    for exploiter, bb100 in exploit_results['exploiter_results'].items():
        print(f"  {exploiter:20} | BB/100: {bb100:>8.2f}")
    print()

    print(f"Worst-Case BB/100:        {exploit_results['worst_case_bb100']:>8.2f}")
    print(f"Avg vs Exploiters:        {exploit_results['avg_bb100_vs_exploiters']:>8.2f}")
    print(f"Exploitability Score:     {exploit_results['exploitability_score']:>8.2f}")
    print()

    if exploit_results['exploitability_score'] < 5:
        print("✅ Low exploitability - robust strategy")
    elif exploit_results['exploitability_score'] < 15:
        print("⚠️  Medium exploitability - some weaknesses")
    else:
        print("❌ High exploitability - vulnerable to counter-strategies")
    print()

    # =========================================================================
    # Summary
    # =========================================================================

    print("="*60)
    print("SUMMARY")
    print("="*60)
    print()
    print(f"✓ Classification Accuracy:  {class_results['accuracy']:.1%}")
    print(f"✓ Average Win Rate:         {avg_win_rate:.1%}")
    print(f"✓ Average BB/100:           {avg_bb100:+.1f}")
    print(f"✓ Exploitability Score:     {exploit_results['exploitability_score']:.2f}")
    print()

    # Overall performance rating
    accuracy_good = class_results['accuracy'] >= 0.75
    winrate_good = avg_win_rate >= 0.50
    bb100_good = avg_bb100 >= 0
    exploit_good = exploit_results['exploitability_score'] < 10

    excellent_count = sum([accuracy_good, winrate_good, bb100_good, exploit_good])

    if excellent_count >= 3:
        print("🎉 Model Performance: EXCELLENT")
    elif excellent_count >= 2:
        print("✅ Model Performance: GOOD")
    elif class_results['accuracy'] >= 0.55:
        print("⚠️  Model Performance: ACCEPTABLE")
    else:
        print("❌ Model Performance: NEEDS IMPROVEMENT")

    print()
    print("="*60)


if __name__ == '__main__':
    main()
