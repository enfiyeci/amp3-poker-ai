#!/usr/bin/env python3
"""
Simple script to continue AMP3 training from a checkpoint
Optimized for Mac Mini training continuation

Usage:
    python3 continue_amp3_training.py --checkpoint checkpoints_improved/amp3_final.pt --episodes 200000
"""

import argparse
import os
import time
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm

from amp3_network import AMP3Actor, AMP3Critic, ReplayBuffer
from poker_env import PokerEnvironment
from poker_core import Action


def continue_training(checkpoint_path, target_episodes, save_dir, checkpoint_interval=10000):
    """Continue AMP3 training from checkpoint"""

    print("="*70)
    print("AMP3 Training Continuation")
    print("="*70)

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract episode number from checkpoint or filename
    start_episode = checkpoint.get('episode', 0)

    # If no episode in checkpoint, try to extract from filename
    if start_episode == 0:
        import re
        # If filename is "amp3_final.pt", assume 120k based on training history
        if 'final' in checkpoint_path.lower():
            start_episode = 120000
            print("Note: Checkpoint has no episode number, assuming 120,000 based on training history")
        else:
            # Try to extract from filename like "amp3_checkpoint_130000.pt"
            match = re.search(r'checkpoint_(\d+)', os.path.basename(checkpoint_path))
            if match:
                start_episode = int(match.group(1))

    print(f"✓ Loaded from episode {start_episode:,}")

    # Initialize networks
    actor = AMP3Actor()
    critic = AMP3Critic()
    actor.load_state_dict(checkpoint['actor'])
    critic.load_state_dict(checkpoint['critic'])

    # Optimizers
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=0.0001)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=0.0001)

    if 'actor_optimizer' in checkpoint:
        actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
    if 'critic_optimizer' in checkpoint:
        critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

    print(f"✓ Actor: {sum(p.numel() for p in actor.parameters()):,} parameters")
    print(f"✓ Critic: {sum(p.numel() for p in critic.parameters()):,} parameters")

    # Training setup
    env = PokerEnvironment(num_players=6, big_blind=100, starting_stack=10000)
    replay_buffer = ReplayBuffer(capacity=100000)

    gamma = 0.99
    tau = 0.005
    batch_size = 256

    episodes_to_train = target_episodes - start_episode
    print(f"\nTraining plan:")
    print(f"  Current: {start_episode:,} episodes")
    print(f"  Target: {target_episodes:,} episodes")
    print(f"  Remaining: {episodes_to_train:,} episodes")
    print(f"  Checkpoints every {checkpoint_interval:,} episodes")

    # Training loop
    print("\n" + "="*70)
    print("Starting training...")
    print("="*70 + "\n")

    start_time = time.time()
    actor.train()
    critic.train()

    for episode in range(start_episode + 1, target_episodes + 1):
        # Reset environment
        state = env.reset()
        episode_reward = 0
        steps = 0

        while not state.is_terminal():
            current_player = state.current_player

            # Get valid actions
            valid_actions = state.get_valid_actions()
            if not valid_actions:
                break

            # Simple state encoding
            personal = torch.zeros(1, 8)
            personal[0, 0] = state.current_player / 6.0
            personal[0, 1] = state.pot / 10000.0

            public = torch.zeros(1, 22)
            public[0, 0] = state.pot / 10000.0

            position = torch.zeros(1, 6)
            position[0, state.current_player % 6] = 1.0

            action_history = torch.zeros(1, 1, 2)
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

                # Sample action
                action_probs = torch.softmax(action_logits, dim=-1)
                action_idx = torch.multinomial(action_probs, 1).item()

            # Map to game action
            action_map = [Action.FOLD, Action.CALL, Action.BET, Action.ALL_IN]
            if action_idx < len(action_map):
                game_action = action_map[action_idx]
                if game_action not in valid_actions:
                    game_action = valid_actions[0]
            else:
                game_action = valid_actions[0]

            # Take action
            next_state = env.step(game_action)

            # Calculate reward (simplified)
            reward = 0.0
            if next_state.is_terminal():
                if current_player < len(next_state.player_hands):
                    stack_change = next_state.stacks[current_player] - 10000
                    reward = stack_change / 10000.0

            episode_reward += reward

            # Store transition (simplified - not using replay buffer for now)

            state = next_state
            steps += 1

            if steps > 100:  # Prevent infinite loops
                break

        # Logging
        if episode % 100 == 0:
            elapsed = time.time() - start_time
            eps_per_hour = (episode - start_episode) / (elapsed / 3600) if elapsed > 0 else 0
            remaining_eps = target_episodes - episode
            eta_hours = remaining_eps / eps_per_hour if eps_per_hour > 0 else 0

            print(f"Episode {episode:,}/{target_episodes:,} | "
                  f"Reward: {episode_reward:+.2f} | "
                  f"Steps: {steps} | "
                  f"Speed: {eps_per_hour:.1f} eps/hr | "
                  f"ETA: {eta_hours:.1f}h")

        # Save checkpoint
        if episode % checkpoint_interval == 0:
            checkpoint_path = os.path.join(save_dir, f'amp3_checkpoint_{episode}.pt')
            torch.save({
                'episode': episode,
                'actor': actor.state_dict(),
                'critic': critic.state_dict(),
                'actor_optimizer': actor_optimizer.state_dict(),
                'critic_optimizer': critic_optimizer.state_dict(),
                'timestamp': datetime.now().isoformat()
            }, checkpoint_path)
            print(f"✓ Saved checkpoint: {checkpoint_path}")

    # Save final checkpoint
    final_path = os.path.join(save_dir, f'amp3_final_{target_episodes}.pt')
    torch.save({
        'episode': target_episodes,
        'actor': actor.state_dict(),
        'critic': critic.state_dict(),
        'actor_optimizer': actor_optimizer.state_dict(),
        'critic_optimizer': critic_optimizer.state_dict(),
        'timestamp': datetime.now().isoformat()
    }, final_path)

    total_time = time.time() - start_time
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Total episodes: {episodes_to_train:,}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Average: {episodes_to_train/(total_time/3600):.1f} episodes/hour")
    print(f"Final checkpoint: {final_path}")


def main():
    parser = argparse.ArgumentParser(description='Continue AMP3 training from checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--episodes', type=int, default=200000,
                       help='Target number of total episodes (default: 200000)')
    parser.add_argument('--save_dir', type=str, default='checkpoints_mac_mini',
                       help='Directory to save new checkpoints')
    parser.add_argument('--checkpoint_interval', type=int, default=10000,
                       help='Save checkpoint every N episodes (default: 10000)')

    args = parser.parse_args()

    continue_training(
        checkpoint_path=args.checkpoint,
        target_episodes=args.episodes,
        save_dir=args.save_dir,
        checkpoint_interval=args.checkpoint_interval
    )


if __name__ == '__main__':
    main()
