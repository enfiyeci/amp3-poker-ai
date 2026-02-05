# AMP3: Adaptive Multi-player Poker Policy

A complete implementation of the AMP3 learning method for 6-player No-Limit Texas Hold'em based on the paper "AMP3: An Adaptive Multi-player Poker Policy Learning Method Based on Opponent Style Modeling" (Neural Computing and Applications, 2025).

## Installation

```bash
pip install torch numpy
```

## File Structure

```
amp3_full/
├── poker_core.py        # Card/hand evaluation, Sklansky groups, Chen formula
├── poker_env.py         # 6-player NLHE game environment
├── style_library.py     # 64 player strategies (Random, Sklansky, Chen, RuleBased)
├── osm_network.py       # LSTM-based Opponent Style Modeling network
├── amp3_network.py      # Actor-Critic RL with style adaptation
├── cfr_networks.py      # Deep CFR and NFSP implementations
├── preflop_imitation.py # Preflop imitation + Later-street models
├── train_amp3.py        # Complete training pipeline
└── README.md            # This file
```

## Quick Start

### Train All Models (Full Pipeline)

```bash
python train_amp3.py --stage all --save_dir ./checkpoints
```

### Train Individual Stages

```bash
# Stage 1-2: Style features + Opponent Style Modeling
python train_amp3.py --stage osm --save_dir ./checkpoints

# Stage 3: Preflop Imitation Learning
python train_amp3.py --stage preflop --save_dir ./checkpoints

# Stage 4: Later-Street Models (Flop/Turn/River)
python train_amp3.py --stage streets --save_dir ./checkpoints

# Stage 5: AMP3 Actor-Critic (main RL training)
python train_amp3.py --stage amp3 --save_dir ./checkpoints

# Stage 6: Deep CFR
python train_amp3.py --stage cfr --save_dir ./checkpoints

# Stage 7: Neural Fictitious Self-Play
python train_amp3.py --stage nfsp --save_dir ./checkpoints
```

## Training Pipeline

The training follows this order:

1. **Style Library Feature Computation** - Simulate games to compute VPIP, PFR, AFq, WTSD for all 64 strategies
2. **OSM Training** - Train LSTM network to predict opponent style from action history
3. **Preflop Imitation** - Supervised learning on expert preflop decisions
4. **Later-Street Models** - Train specialized networks for Flop, Turn, River
5. **AMP3 Actor-Critic** - Reinforcement learning with opponent style adaptation
6. **Deep CFR** (optional) - Counterfactual regret minimization
7. **NFSP** (optional) - Neural Fictitious Self-Play

## Configuration

Create a `config.json` file to customize training:

```json
{
    "seed": 42,
    "device": "cuda",
    "num_players": 6,
    "big_blind": 100,
    "starting_stack": 10000,
    
    "osm_num_games": 100000,
    "osm_epochs": 100,
    "osm_batch_size": 128,
    "osm_lr": 0.001,
    
    "preflop_num_samples": 100000,
    "preflop_epochs": 100,
    
    "amp3_episodes": 120000,
    "amp3_batch_size": 256,
    "amp3_actor_lr": 0.0001,
    "amp3_critic_lr": 0.0001,
    "amp3_gamma": 0.99
}
```

Then run:
```bash
python train_amp3.py --config config.json --stage all
```

## Key Components

### Style Library (64 Strategies)

| Strategy Type | Player Types | Total |
|--------------|--------------|-------|
| Random | 7 probability configs | 7 |
| Sklansky-based | Conservative, Regular, Aggressive, Bluffing, Deceptive | 5 |
| Chen-based | Conservative, Regular, Aggressive, Bluffing, Deceptive | 5 |
| Rule-based | Conservative, Regular, Aggressive, Bluffing, Deceptive | 5 |

Each is combined: 7 + (3 methods × 5 types) × 4 variations = 64+ strategies

### Style Features (VPIP, PFR, AFq, WTSD)

- **VPIP**: Voluntarily Put money In Pot (preflop participation rate)
- **PFR**: Pre-Flop Raise percentage
- **AFq**: Aggression Frequency (post-flop)
- **WTSD**: Went To ShowDown percentage

### Network Architectures

**OSM Network:**
- Public cards encoder: FC(10→64→32)
- Hole cards encoder: FC(4→64→32)
- Action LSTM: 3-layer bidirectional LSTM(2→128)
- Output: 4 style features with Sigmoid

**AMP3 Actor:**
- Personal encoder: FC(8→128→64)
- Public encoder: FC(22→128→64)
- Position encoder: FC(6→32)
- Style encoder: FC(24→128→64)
- Action LSTM: 3-layer bidirectional LSTM
- Policy head: FC(→128→64→4)

**AMP3 Critic:**
- Uses global information (all hole cards visible)
- Outputs single Q-value

## Usage Examples

### Play a Game with Trained Agent

```python
from poker_env import PokerEnvironment, Action, ActionType
from amp3_network import AMP3Agent
from osm_network import StyleModelingNetwork

# Load trained models
osm = StyleModelingNetwork()
osm.load_state_dict(torch.load('checkpoints/osm_best.pt'))

agent = AMP3Agent(osm_network=osm)
agent.actor.load_state_dict(torch.load('checkpoints/amp3_final.pt')['actor'])

# Create environment
env = PokerEnvironment(num_players=6, big_blind=100, starting_stack=10000)
state = env.reset()

# Get action for player 0
action = agent.get_action(state, player_idx=0, opponent_styles=None)
print(f"Agent action: {action}")
```

### Evaluate Against Style Library

```python
from style_library import StyleLibrary

library = StyleLibrary()

# Test AMP3 vs random opponents
wins = 0
for game in range(1000):
    opponents = [library.get_random_strategy() for _ in range(5)]
    # ... run game ...
    
print(f"Win rate: {wins/1000:.2%}")
```

## Paper Reference

Based on: "AMP3: An Adaptive Multi-player Poker Policy Learning Method Based on Opponent Style Modeling"
- DOI: 10.1007/s00521-025-11262-x
- Neural Computing and Applications, 2025

## Hardware Requirements

- **Minimum**: CPU with 8GB RAM (slow training)
- **Recommended**: GPU with 8GB+ VRAM for faster training
- **Full training time**: ~24-48 hours on GPU for complete pipeline

## License

This implementation is for research and educational purposes.
