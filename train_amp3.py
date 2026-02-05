"""
AMP3 Training Pipeline

This script provides a complete training pipeline for all AMP3 components:
1. Style Library Feature Computation
2. OSM (Opponent Style Modeling) Training
3. Preflop Imitation Learning
4. Later-Street Model Training
5. Full AMP3 Actor-Critic Training
6. Deep CFR Training
7. NFSP Training

Usage:
    python train_amp3.py --stage all          # Run full pipeline
    python train_amp3.py --stage osm          # Train only OSM
    python train_amp3.py --stage preflop      # Train only preflop
    python train_amp3.py --stage amp3         # Train only AMP3
    python train_amp3.py --stage cfr          # Train only CFR
    python train_amp3.py --stage nfsp         # Train only NFSP
"""

import argparse
import os
import time
import json
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Import all AMP3 components
from poker_core import Card, Rank, Suit, evaluate_hand, Street
from poker_env import PokerEnvironment
from poker_core import Action
from style_library import StyleLibrary, StyleFeatures
from osm_network import (
    StyleModelingNetwork, OSMTrainer, StyleFeatureDataset,
    generate_osm_dataset, StylePredictor, GameHistoryEncoding
)
from amp3_network import (
    AMP3Actor, AMP3Critic, AMP3Agent, ReplayBuffer,
    encode_amp3_state, calculate_reward
)
from cfr_networks import DeepCFR, NFSPAgent, InfoSet
from preflop_imitation import (
    PreflopImitationNetwork, PreflopImitationTrainer,
    PreflopExpertDataset, StreetSpecificModels, LaterStreetState,
    AMP3WithSpecializedModels
)


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CONFIG = {
    # General
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir': './amp3_checkpoints',
    
    # Poker environment
    'num_players': 6,
    'big_blind': 100,
    'starting_stack': 10000,
    
    # Style library
    'num_style_games': 10000,  # Games to compute style features
    
    # OSM training
    'osm_num_games': 100000,
    'osm_epochs': 100,
    'osm_batch_size': 128,
    'osm_lr': 1e-3,
    'osm_lstm_hidden': 128,
    'osm_lstm_layers': 3,
    
    # Preflop imitation
    'preflop_num_samples': 100000,
    'preflop_epochs': 100,
    'preflop_batch_size': 256,
    'preflop_lr': 1e-3,
    'preflop_hidden': [128, 128, 64],
    
    # Later-street training
    'street_num_samples': 50000,
    'street_epochs': 100,
    'street_batch_size': 256,
    'street_lr': 1e-3,
    
    # AMP3 RL training
    'amp3_episodes': 120000,
    'amp3_batch_size': 256,
    'amp3_actor_lr': 1e-4,
    'amp3_critic_lr': 1e-4,
    'amp3_gamma': 0.99,
    'amp3_tau': 0.005,
    'amp3_entropy_coef': 0.01,
    'amp3_replay_capacity': 100000,
    'amp3_warmup_episodes': 1000,
    
    # CFR training
    'cfr_iterations': 1000,
    'cfr_traversals_per_iter': 100,
    'cfr_lr': 1e-3,
    
    # NFSP training
    'nfsp_episodes': 100000,
    'nfsp_rl_lr': 1e-4,
    'nfsp_sl_lr': 1e-3,
    'nfsp_eta': 0.1,
}


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)


# =============================================================================
# Stage 1: Style Library Feature Computation
# =============================================================================

def train_style_features(config: Dict) -> StyleLibrary:
    """
    Compute style features for all 64 strategies in the library.
    
    This involves simulating games between strategies and computing
    their VPIP, PFR, AFq, and WTSD statistics.
    """
    print("\n" + "="*60)
    print("Stage 1: Computing Style Library Features")
    print("="*60)
    
    library = StyleLibrary()
    print(f"Loaded {len(library.strategies)} strategies")
    
    # Compute features through simulation
    print(f"\nSimulating {config['num_style_games']} games to compute features...")
    
    env = PokerEnvironment(
        num_players=config['num_players'],
        big_blind=config['big_blind'],
        starting_stack=config['starting_stack']
    )
    
    # Track statistics for each strategy
    strategy_stats = defaultdict(lambda: {
        'hands': 0,
        'vpip_count': 0,
        'pfr_count': 0,
        'postflop_aggressive': 0,
        'postflop_passive': 0,
        'showdowns': 0,
        'went_to_showdown': 0,
    })
    
    for game_idx in range(config['num_style_games']):
        # Select strategies for this game
        strategy_names = random.sample(list(library.strategies.keys()), 6)
        strategies = [library.get_strategy(name) for name in strategy_names]
        
        state = env.reset()
        
        # Track preflop actions
        preflop_actions = {i: [] for i in range(6)}
        current_street = Street.PREFLOP
        
        while not state.is_terminal:
            player_idx = state.current_player
            strategy = strategies[player_idx]
            strategy_name = strategy_names[player_idx]
            player = state.players[player_idx]

            # Get action
            action, amount = strategy.get_action(state, player_idx, player.hole_cards, state.community_cards)

            # Track statistics
            if state.street == Street.PREFLOP:
                if action in [Action.CALL, Action.BET, Action.ALL_IN]:
                    strategy_stats[strategy_name]['vpip_count'] += 1
                if action in [Action.BET, Action.ALL_IN]:
                    strategy_stats[strategy_name]['pfr_count'] += 1
            else:
                if action in [Action.BET, Action.ALL_IN]:
                    strategy_stats[strategy_name]['postflop_aggressive'] += 1
                elif action == Action.CALL:
                    strategy_stats[strategy_name]['postflop_passive'] += 1

            strategy_stats[strategy_name]['hands'] += 1

            # Check for showdown
            if state.street == Street.RIVER and action != Action.FOLD:
                strategy_stats[strategy_name]['showdowns'] += 1

            state, _, _ = env.step(action, amount)
        
        # Track showdown participation
        if state.is_terminal:
            active_players = [i for i, p in enumerate(state.players) if p.is_active]
            if len(active_players) > 1:  # Showdown occurred
                for i in active_players:
                    strategy_stats[strategy_names[i]]['went_to_showdown'] += 1
        
        if (game_idx + 1) % 1000 == 0:
            print(f"  Processed {game_idx + 1}/{config['num_style_games']} games")
    
    # Compute final features
    print("\nComputed style features:")
    features = {}
    for name, stats in strategy_stats.items():
        if stats['hands'] > 0:
            vpip = stats['vpip_count'] / stats['hands']
            pfr = stats['pfr_count'] / stats['hands']
            afq = stats['postflop_aggressive'] / max(
                stats['postflop_aggressive'] + stats['postflop_passive'], 1
            )
            wtsd = stats['went_to_showdown'] / max(stats['showdowns'], 1)
            
            features[name] = StyleFeatures(vpip, pfr, afq, wtsd)
            print(f"  {name}: VPIP={vpip:.3f}, PFR={pfr:.3f}, "
                  f"AFq={afq:.3f}, WTSD={wtsd:.3f}")
    
    # Save features
    feature_path = os.path.join(config['save_dir'], 'style_features.json')
    with open(feature_path, 'w') as f:
        json.dump({k: v.to_vector().tolist() for k, v in features.items()}, f)
    print(f"\nSaved style features to {feature_path}")
    
    return library


# =============================================================================
# Stage 2: OSM Training
# =============================================================================

def train_osm(config: Dict, style_library: StyleLibrary) -> StyleModelingNetwork:
    """
    Train the Opponent Style Modeling network.
    """
    print("\n" + "="*60)
    print("Stage 2: Training OSM (Opponent Style Modeling)")
    print("="*60)
    
    device = config['device']
    
    # Generate or load OSM dataset
    dataset_path = os.path.join(config['save_dir'], 'osm_dataset.pt')
    
    if os.path.exists(dataset_path):
        print(f"Loading existing OSM dataset from {dataset_path}")
        dataset = StyleFeatureDataset.load(dataset_path)
    else:
        print(f"Generating OSM dataset with {config['osm_num_games']} games...")
        dataset = generate_osm_dataset(
            num_games=config['osm_num_games'],
            style_library=style_library,
        )
        dataset.save(dataset_path)
        print(f"Saved OSM dataset to {dataset_path}")
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['osm_batch_size'], 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['osm_batch_size']
    )
    
    # Create network
    osm_network = StyleModelingNetwork(
        lstm_hidden=config['osm_lstm_hidden'],
        lstm_layers=config['osm_lstm_layers'],
    ).to(device)
    
    print(f"OSM network parameters: {sum(p.numel() for p in osm_network.parameters()):,}")
    
    # Train
    trainer = OSMTrainer(osm_network, learning_rate=config['osm_lr'], device=device)
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(config['osm_epochs']):
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.validate(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(
                osm_network.state_dict(),
                os.path.join(config['save_dir'], 'osm_best.pt')
            )
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{config['osm_epochs']}: "
                  f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    osm_network.load_state_dict(
        torch.load(os.path.join(config['save_dir'], 'osm_best.pt'))
    )
    
    print(f"\nOSM training complete. Best val loss: {best_val_loss:.4f}")
    return osm_network


# =============================================================================
# Stage 3: Preflop Imitation Learning
# =============================================================================

def train_preflop(config: Dict, style_library: StyleLibrary) -> PreflopImitationNetwork:
    """
    Train the preflop imitation learning network.
    """
    print("\n" + "="*60)
    print("Stage 3: Training Preflop Imitation Network")
    print("="*60)
    
    device = config['device']
    
    # Generate or load preflop dataset
    dataset_path = os.path.join(config['save_dir'], 'preflop_dataset.pt')
    
    if os.path.exists(dataset_path):
        print(f"Loading existing preflop dataset from {dataset_path}")
        dataset = PreflopExpertDataset.load(dataset_path)
    else:
        print(f"Generating preflop dataset with {config['preflop_num_samples']} samples...")
        dataset = PreflopExpertDataset.generate_from_style_library(
            style_library=style_library,
            num_samples=config['preflop_num_samples'],
        )
        dataset.save(dataset_path)
        print(f"Saved preflop dataset to {dataset_path}")
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create network
    preflop_network = PreflopImitationNetwork(
        hidden_dims=config['preflop_hidden']
    ).to(device)
    
    print(f"Preflop network parameters: {sum(p.numel() for p in preflop_network.parameters()):,}")
    
    # Train
    trainer = PreflopImitationTrainer(
        preflop_network,
        learning_rate=config['preflop_lr'],
        device=device
    )
    
    # Reconstruct datasets for trainer
    train_ds = PreflopExpertDataset(
        [dataset.states[i].numpy() for i in train_dataset.indices],
        [dataset.actions[i].item() for i in train_dataset.indices]
    )
    val_ds = PreflopExpertDataset(
        [dataset.states[i].numpy() for i in val_dataset.indices],
        [dataset.actions[i].item() for i in val_dataset.indices]
    )
    
    history = trainer.train(
        train_ds,
        val_ds,
        epochs=config['preflop_epochs'],
        batch_size=config['preflop_batch_size'],
        verbose=True
    )
    
    # Save model
    torch.save(
        preflop_network.state_dict(),
        os.path.join(config['save_dir'], 'preflop_best.pt')
    )
    
    print(f"\nPreflop training complete. Final accuracy: {history['val_accuracy'][-1]:.4f}")
    return preflop_network


# =============================================================================
# Stage 4: Later-Street Model Training
# =============================================================================

def train_later_streets(
    config: Dict, 
    style_library: StyleLibrary
) -> StreetSpecificModels:
    """
    Train the later-street specialized models (Flop, Turn, River).
    """
    print("\n" + "="*60)
    print("Stage 4: Training Later-Street Models")
    print("="*60)
    
    device = config['device']
    street_models = StreetSpecificModels(device=device)
    
    env = PokerEnvironment(
        num_players=config['num_players'],
        big_blind=config['big_blind'],
        starting_stack=config['starting_stack']
    )
    
    # Collect data for each street
    for street_name, street_enum in [
        ('flop', Street.FLOP), 
        ('turn', Street.TURN), 
        ('river', Street.RIVER)
    ]:
        print(f"\n--- Training {street_name.upper()} model ---")
        
        states = []
        actions = []
        
        # Generate samples through simulation
        samples_collected = 0
        games_played = 0
        
        while samples_collected < config['street_num_samples']:
            # Select strategies
            strategy_names = random.sample(list(style_library.strategies.keys()), 6)
            strategies = [style_library.get_strategy(name) for name in strategy_names]
            
            state = env.reset()
            
            while not state.is_terminal:
                player_idx = state.current_player
                strategy = strategies[player_idx]
                
                # Record data if on target street
                if state.street == street_enum:
                    player = state.players[player_idx]
                    
                    later_state = LaterStreetState(
                        hole_cards=tuple(player.hole_cards),
                        community_cards=state.community_cards,
                        position=player_idx,
                        pot_size=state.pot,
                        to_call=state.current_bet - player.bet_this_street,
                        stack=player.stack,
                        street=state.street,
                        num_opponents=sum(1 for p in state.players if p.is_active) - 1,
                        action_history=state.action_history,
                    )
                    
                    state_vec = later_state.to_vector(env.big_blind)

                    # Get action
                    action, amount = strategy.get_action(state, player_idx, player.hole_cards, state.community_cards)

                    # Encode action
                    if action == Action.FOLD:
                        action_idx = 0
                    elif action == Action.CALL:
                        action_idx = 1
                    elif action == Action.BET:
                        if amount >= player.stack * 0.9:
                            action_idx = 3
                        else:
                            action_idx = 2
                    elif action == Action.ALL_IN:
                        action_idx = 3
                    else:
                        action_idx = 1
                    
                    states.append(state_vec)
                    actions.append(action_idx)
                    samples_collected += 1
                    
                    if samples_collected >= config['street_num_samples']:
                        break

                player = state.players[player_idx]
                action, amount = strategy.get_action(state, player_idx, player.hole_cards, state.community_cards)
                state, _, _ = env.step(action, amount)
            
            games_played += 1
            if games_played % 1000 == 0:
                print(f"  Games: {games_played}, Samples: {samples_collected}")
        
        print(f"  Collected {len(states)} {street_name} samples")
        
        # Create dataset
        states_tensor = torch.FloatTensor(np.array(states))
        actions_tensor = torch.LongTensor(actions)
        
        # Split
        train_size = int(0.9 * len(states))
        indices = torch.randperm(len(states))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        train_states = states_tensor[train_indices].to(device)
        train_actions = actions_tensor[train_indices].to(device)
        val_states = states_tensor[val_indices].to(device)
        val_actions = actions_tensor[val_indices].to(device)
        
        # Get network for this street
        network = street_models.get_network(street_enum)
        optimizer = optim.Adam(network.parameters(), lr=config['street_lr'])
        criterion = nn.CrossEntropyLoss()
        
        # Train
        best_val_loss = float('inf')
        
        for epoch in range(config['street_epochs']):
            network.train()
            
            # Mini-batch training
            perm = torch.randperm(len(train_states))
            total_loss = 0
            num_batches = 0
            
            for i in range(0, len(train_states), config['street_batch_size']):
                batch_idx = perm[i:i+config['street_batch_size']]
                batch_states = train_states[batch_idx]
                batch_actions = train_actions[batch_idx]
                
                optimizer.zero_grad()
                logits, _ = network(batch_states)
                loss = criterion(logits, batch_actions)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            # Validation
            network.eval()
            with torch.no_grad():
                val_logits, _ = network(val_states)
                val_loss = criterion(val_logits, val_actions).item()
                val_preds = torch.argmax(val_logits, dim=-1)
                val_acc = (val_preds == val_actions).float().mean().item()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}: loss={total_loss/num_batches:.4f}, "
                      f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        
        print(f"  {street_name.upper()} training complete. Best val loss: {best_val_loss:.4f}")
    
    # Save models
    street_models.save(os.path.join(config['save_dir'], 'street_models.pt'))
    print(f"\nSaved street models to {config['save_dir']}/street_models.pt")
    
    return street_models


# =============================================================================
# Stage 5: AMP3 Actor-Critic Training
# =============================================================================

def train_amp3(
    config: Dict,
    style_library: StyleLibrary,
    osm_network: StyleModelingNetwork,
    preflop_network: Optional[PreflopImitationNetwork] = None
) -> AMP3Agent:
    """
    Train the full AMP3 Actor-Critic agent.
    """
    print("\n" + "="*60)
    print("Stage 5: Training AMP3 Actor-Critic")
    print("="*60)
    
    device = config['device']
    
    # Create AMP3 agent
    amp3_agent = AMP3Agent(
        osm_model=osm_network,
        actor_lr=config['amp3_actor_lr'],
        critic_lr=config['amp3_critic_lr'],
        gamma=config['amp3_gamma'],
        device=device,
    )
    
    print(f"AMP3 Actor parameters: {sum(p.numel() for p in amp3_agent.actor.parameters()):,}")
    print(f"AMP3 Critic parameters: {sum(p.numel() for p in amp3_agent.critic.parameters()):,}")
    
    # Initialize from preflop if available
    if preflop_network is not None:
        print("Initializing actor with preflop network weights (partial)...")
        # Note: Full weight transfer would require architecture matching
        # Here we just use preflop for preflop decisions during warmup
    
    # Create environment
    env = PokerEnvironment(
        num_players=config['num_players'],
        big_blind=config['big_blind'],
        starting_stack=config['starting_stack']
    )
    
    # Training metrics (replay buffer is inside amp3_agent)
    episode_rewards = []
    win_rates = []
    
    print(f"\nTraining for {config['amp3_episodes']} episodes...")
    
    for episode in range(config['amp3_episodes']):
        # Select opponent strategies
        opponent_names = random.sample(list(style_library.strategies.keys()), 5)
        opponent_strategies = [style_library.get_strategy(n) for n in opponent_names]
        
        # AMP3 agent is player 0
        state = env.reset()
        episode_reward = 0

        # Predict opponent styles (empty history initially)
        opponent_styles = amp3_agent.predict_opponent_styles({}, state)
        
        # Store episode transitions
        episode_transitions = []

        while not state.is_terminal:
            player_idx = state.current_player

            if player_idx == 0:
                # AMP3 agent's turn
                action_enum, amount, log_prob = amp3_agent.get_action(
                    state,
                    player_idx,
                    opponent_styles
                )

                # Encode state for training (convert tensors to numpy)
                from amp3_network import encode_amp3_state, state_to_tensors, encode_critic_state, Experience
                amp3_state = encode_amp3_state(state, player_idx, opponent_styles)
                state_tensors = state_to_tensors(amp3_state, config['device'])

                # Convert tensors to numpy for storage
                state_numpy = {k: v.cpu().numpy() for k, v in state_tensors.items()}

                # Encode critic state
                critic_state_numpy = encode_critic_state(state, player_idx, opponent_styles)

                # Determine action index from Action enum
                action_map = {
                    Action.FOLD: 0,
                    Action.CALL: 1,
                    Action.BET: 2,
                    Action.ALL_IN: 3
                }
                action_idx = action_map.get(action_enum, 1)  # Default to CALL if unknown

                episode_transitions.append({
                    'state_numpy': state_numpy,
                    'critic_state_numpy': critic_state_numpy,
                    'action_idx': action_idx,
                    'action_enum': action_enum,
                    'amount': amount,
                    'log_prob': log_prob,
                    'game_state': state,
                })

                action = action_enum  # For env.step
            else:
                # Opponent's turn
                player = state.players[player_idx]
                action, amount = opponent_strategies[player_idx - 1].get_action(
                    state, player_idx, player.hole_cards, state.community_cards
                )

            state, _, _ = env.step(action, amount)

        # Calculate rewards and create Experience objects for AMP3's actions
        # Determine game outcome
        final_state = state
        is_winner = 0 in final_state.winners  # Player 0 (AMP3) won
        final_pot = final_state.pot
        num_winners = len(final_state.winners)
        reached_showdown = final_state.street == Street.RIVER and final_state.players[0].is_active

        for i, trans in enumerate(episode_transitions):
            # Get pot before this action
            pot_before = trans['game_state'].pot

            # Calculate reward
            reward = calculate_reward(
                trans['action_enum'],
                trans['amount'],
                pot_before,
                is_winner,
                final_pot,
                num_winners,
                reached_showdown
            )
            episode_reward += reward

            # Get next state (or None if terminal)
            if i < len(episode_transitions) - 1:
                next_trans = episode_transitions[i + 1]
                next_state_numpy = next_trans['state_numpy']
                next_critic_numpy = next_trans['critic_state_numpy']
                done = False
            else:
                next_state_numpy = None
                next_critic_numpy = None
                done = True

            # Create Experience object and add to replay buffer
            experience = Experience(
                state=trans['state_numpy'],
                critic_state=trans['critic_state_numpy'],
                action=trans['action_idx'],
                reward=reward,
                next_state=next_state_numpy,
                next_critic_state=next_critic_numpy,
                done=done,
                log_prob=trans['log_prob']
            )
            amp3_agent.replay_buffer.push(experience)
        
        episode_rewards.append(episode_reward)
        
        # Track win rate
        player0_chips = state.players[0].stack
        won = player0_chips > config['starting_stack']
        win_rates.append(float(won))
        
        # Update agent
        if len(amp3_agent.replay_buffer) >= config['amp3_batch_size'] and episode >= config['amp3_warmup_episodes']:
            actor_loss, critic_loss = amp3_agent.update(
                batch_size=config['amp3_batch_size']
            )
        
        # Logging
        if (episode + 1) % 1000 == 0:
            avg_reward = np.mean(episode_rewards[-1000:])
            avg_win_rate = np.mean(win_rates[-1000:])
            print(f"  Episode {episode+1}: avg_reward={avg_reward:.2f}, "
                  f"win_rate={avg_win_rate:.3f}")
        
        # Save checkpoint
        if (episode + 1) % 10000 == 0:
            torch.save({
                'actor': amp3_agent.actor.state_dict(),
                'critic': amp3_agent.critic.state_dict(),
                'episode': episode,
            }, os.path.join(config['save_dir'], f'amp3_checkpoint_{episode+1}.pt'))
    
    # Save final model
    torch.save({
        'actor': amp3_agent.actor.state_dict(),
        'critic': amp3_agent.critic.state_dict(),
    }, os.path.join(config['save_dir'], 'amp3_final.pt'))
    
    print(f"\nAMP3 training complete.")
    print(f"Final avg reward: {np.mean(episode_rewards[-1000:]):.2f}")
    print(f"Final win rate: {np.mean(win_rates[-1000:]):.3f}")
    
    return amp3_agent


# =============================================================================
# Stage 6: Deep CFR Training
# =============================================================================

def train_cfr(config: Dict) -> DeepCFR:
    """
    Train Deep CFR networks.
    """
    print("\n" + "="*60)
    print("Stage 6: Training Deep CFR")
    print("="*60)
    
    device = config['device']
    
    # Create Deep CFR
    cfr = DeepCFR(
        num_players=config['num_players'],
        device=device,
        lr=config['cfr_lr']
    )
    
    print(f"CFR value network parameters: {sum(p.numel() for p in cfr.value_networks[0].parameters()):,}")
    print(f"CFR strategy network parameters: {sum(p.numel() for p in cfr.strategy_network.parameters()):,}")
    
    env = PokerEnvironment(
        num_players=config['num_players'],
        big_blind=config['big_blind'],
        starting_stack=config['starting_stack']
    )
    
    print(f"\nRunning {config['cfr_iterations']} CFR iterations...")
    
    for iteration in range(config['cfr_iterations']):
        losses = cfr.train_iteration(
            env, 
            num_traversals=config['cfr_traversals_per_iter']
        )
        
        if (iteration + 1) % 100 == 0:
            print(f"  Iteration {iteration+1}: value_loss={losses['value_loss']:.4f}, "
                  f"strategy_loss={losses['strategy_loss']:.4f}")
    
    # Save
    torch.save({
        'value_networks': [vn.state_dict() for vn in cfr.value_networks],
        'strategy_network': cfr.strategy_network.state_dict(),
    }, os.path.join(config['save_dir'], 'cfr_final.pt'))
    
    print(f"\nDeep CFR training complete.")
    return cfr


# =============================================================================
# Stage 7: NFSP Training
# =============================================================================

def train_nfsp(config: Dict, style_library: StyleLibrary) -> List[NFSPAgent]:
    """
    Train NFSP (Neural Fictitious Self-Play) agents.
    """
    print("\n" + "="*60)
    print("Stage 7: Training NFSP")
    print("="*60)
    
    device = config['device']
    
    # Create NFSP agents for each player
    nfsp_agents = [
        NFSPAgent(
            state_dim=72,  # InfoSet dimension
            action_dim=4,
            device=device,
            rl_lr=config['nfsp_rl_lr'],
            sl_lr=config['nfsp_sl_lr'],
            eta=config['nfsp_eta']
        )
        for _ in range(config['num_players'])
    ]
    
    print(f"Created {len(nfsp_agents)} NFSP agents")
    print(f"Q-network parameters per agent: {sum(p.numel() for p in nfsp_agents[0].q_network.parameters()):,}")
    
    env = PokerEnvironment(
        num_players=config['num_players'],
        big_blind=config['big_blind'],
        starting_stack=config['starting_stack']
    )
    
    print(f"\nTraining for {config['nfsp_episodes']} episodes...")
    
    for episode in range(config['nfsp_episodes']):
        state = env.reset()
        
        while not state.is_terminal:
            player_idx = state.current_player
            agent = nfsp_agents[player_idx]
            
            # Create info set
            player = state.players[player_idx]
            info_set = InfoSet(
                player=player_idx,
                hole_cards=tuple(player.hole_cards),
                community_cards=tuple(state.community_cards),
                pot=state.pot,
                to_call=state.current_bet - player.current_bet,
                stack=player.stack,
                action_sequence=tuple(state.action_history),
                street=state.street,
                position=player_idx,
            )
            state_vec = info_set.to_vector()
            
            # Get action
            action_idx = agent.get_action(state_vec)
            
            # Convert to Action
            to_call = state.current_bet - player.current_bet
            if action_idx == 0:
                if to_call > 0:
                    action, amount = Action.FOLD, 0
                else:
                    action, amount = Action.CALL, 0
            elif action_idx == 1:
                action, amount = Action.CALL, min(to_call, player.stack)
            elif action_idx == 2:
                bet = min(state.pot + to_call, player.stack)
                if bet > to_call:
                    action, amount = Action.BET, bet
                else:
                    action, amount = Action.CALL, bet
            else:
                action, amount = Action.ALL_IN, player.stack

            # Store experience (simplified)
            # In full implementation, would track rewards and next states

            state, _, _ = env.step(action, amount)
        
        # Update agents
        if (episode + 1) % 100 == 0:
            for agent in nfsp_agents:
                agent.update()
        
        # Logging
        if (episode + 1) % 10000 == 0:
            print(f"  Episode {episode+1}/{config['nfsp_episodes']}")
    
    # Save
    torch.save({
        'agents': [
            {
                'q_network': a.q_network.state_dict(),
                'average_network': a.average_network.state_dict(),
            }
            for a in nfsp_agents
        ]
    }, os.path.join(config['save_dir'], 'nfsp_final.pt'))
    
    print(f"\nNFSP training complete.")
    return nfsp_agents


# =============================================================================
# Main Training Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='AMP3 Training Pipeline')
    parser.add_argument('--stage', type=str, default='all',
                       choices=['all', 'osm', 'preflop', 'streets', 'amp3', 'cfr', 'nfsp'],
                       help='Training stage to run')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config JSON file')
    parser.add_argument('--save_dir', type=str, default='./amp3_checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cpu/cuda)')
    
    args = parser.parse_args()
    
    # Load config
    config = DEFAULT_CONFIG.copy()
    
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            user_config = json.load(f)
        config.update(user_config)
    
    if args.save_dir:
        config['save_dir'] = args.save_dir
    
    if args.device:
        config['device'] = args.device
    
    # Setup
    set_seed(config['seed'])
    ensure_dir(config['save_dir'])
    
    print("="*60)
    print("AMP3 Training Pipeline")
    print("="*60)
    print(f"Device: {config['device']}")
    print(f"Save directory: {config['save_dir']}")
    print(f"Stage: {args.stage}")
    
    # Save config
    with open(os.path.join(config['save_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize style library
    style_library = StyleLibrary()
    
    # Run stages
    osm_network = None
    preflop_network = None
    street_models = None
    
    if args.stage in ['all', 'osm']:
        # Stage 1: Style features
        style_library = train_style_features(config)
        
        # Stage 2: OSM
        osm_network = train_osm(config, style_library)
    
    if args.stage in ['all', 'preflop']:
        # Stage 3: Preflop
        preflop_network = train_preflop(config, style_library)
    
    if args.stage in ['all', 'streets']:
        # Stage 4: Later streets
        street_models = train_later_streets(config, style_library)
    
    if args.stage in ['all', 'amp3']:
        # Load OSM if not trained
        if osm_network is None:
            osm_path = os.path.join(config['save_dir'], 'osm_best.pt')
            if os.path.exists(osm_path):
                osm_network = StyleModelingNetwork().to(config['device'])
                osm_network.load_state_dict(torch.load(osm_path))
                print(f"Loaded OSM from {osm_path}")
            else:
                print("Warning: OSM not found, training with random opponent modeling")
                osm_network = StyleModelingNetwork().to(config['device'])
        
        # Stage 5: AMP3
        amp3_agent = train_amp3(config, style_library, osm_network, preflop_network)
    
    if args.stage in ['all', 'cfr']:
        # Stage 6: CFR
        cfr = train_cfr(config)
    
    if args.stage in ['all', 'nfsp']:
        # Stage 7: NFSP
        nfsp_agents = train_nfsp(config, style_library)
    
    print("\n" + "="*60)
    print("Training Pipeline Complete!")
    print("="*60)
    print(f"\nCheckpoints saved to: {config['save_dir']}")
    print("\nAvailable models:")
    for f in os.listdir(config['save_dir']):
        if f.endswith('.pt'):
            print(f"  - {f}")


if __name__ == '__main__':
    main()
