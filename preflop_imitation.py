"""
Preflop Imitation Learning Module for AMP3

This module implements:
1. Preflop expert data generation from style library
2. Preflop MLP network for imitation learning
3. Later-street specialized models (Flop, Turn, River)

Based on the paper's approach of using imitation learning to bootstrap
the preflop policy before RL fine-tuning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from enum import IntEnum
import random
from collections import defaultdict

from poker_core import (
    Card, HandRank, evaluate_hand, get_sklansky_group, 
    get_chen_score, get_hand_strength_monte_carlo,
    Suit, Rank
)
from poker_env import PokerEnvironment
from poker_core import Action, Street


# =============================================================================
# Preflop State Encoding
# =============================================================================

@dataclass
class PreflopState:
    """Encoded preflop state for imitation learning."""
    hole_cards: Tuple[Card, Card]
    position: int  # 0-5 (BTN, SB, BB, UTG, HJ, CO)
    pot_size: float
    to_call: float
    stack: float
    num_raises: int
    num_callers: int
    
    def to_vector(self, big_blind: float = 100.0) -> np.ndarray:
        """
        Encode preflop state as feature vector.
        
        Features (29 total):
        - Hole card 1: rank (1), suit (1), normalized (2)
        - Hole card 2: rank (1), suit (1), normalized (2)
        - Hand features: sklansky group (1), chen score (1), 
                        is_pair (1), is_suited (1), gap (1)
        - Position: one-hot (6)
        - Betting: pot/BB (1), to_call/BB (1), stack/BB (1),
                  num_raises (1), num_callers (1), pot_odds (1),
                  stack_to_pot (1), is_raised (1), can_raise (1)
        """
        c1, c2 = self.hole_cards
        
        # Ensure c1 has higher rank
        if c2.rank.value > c1.rank.value:
            c1, c2 = c2, c1
        
        # Card features
        card_features = [
            c1.rank.value / 14.0,
            c1.suit.value / 4.0,
            c2.rank.value / 14.0,
            c2.suit.value / 4.0,
        ]
        
        # Hand features
        sklansky = get_sklansky_group(c1, c2) / 9.0
        chen = get_chen_score(c1, c2) / 20.0
        is_pair = float(c1.rank == c2.rank)
        is_suited = float(c1.suit == c2.suit)
        gap = abs(c1.rank.value - c2.rank.value) / 12.0
        
        hand_features = [sklansky, chen, is_pair, is_suited, gap]
        
        # Position one-hot
        position_onehot = [0.0] * 6
        position_onehot[self.position] = 1.0
        
        # Betting features
        pot_norm = self.pot_size / big_blind
        to_call_norm = self.to_call / big_blind
        stack_norm = self.stack / big_blind
        
        pot_odds = self.to_call / (self.pot_size + self.to_call + 1e-6)
        stack_to_pot = self.stack / (self.pot_size + 1e-6)
        is_raised = float(self.num_raises > 0)
        can_raise = float(self.stack > self.to_call)
        
        betting_features = [
            min(pot_norm / 10.0, 1.0),
            min(to_call_norm / 10.0, 1.0),
            min(stack_norm / 200.0, 1.0),
            min(self.num_raises / 4.0, 1.0),
            min(self.num_callers / 5.0, 1.0),
            pot_odds,
            min(stack_to_pot / 50.0, 1.0),
            is_raised,
            can_raise,
        ]
        
        return np.array(
            card_features + hand_features + position_onehot + betting_features,
            dtype=np.float32
        )


# =============================================================================
# Preflop Expert Data Generation
# =============================================================================

class PreflopExpertDataset(Dataset):
    """Dataset of expert preflop decisions from style library."""
    
    def __init__(self, states: List[np.ndarray], actions: List[int]):
        self.states = torch.FloatTensor(np.array(states))
        self.actions = torch.LongTensor(actions)
    
    def __len__(self):
        return len(self.actions)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]
    
    @classmethod
    def generate_from_style_library(
        cls,
        style_library,
        num_samples: int = 100000,
        expert_styles: Optional[List[str]] = None
    ) -> 'PreflopExpertDataset':
        """
        Generate expert preflop data by simulating games with style library.
        
        Args:
            style_library: StyleLibrary instance
            num_samples: Number of preflop decisions to collect
            expert_styles: List of style names to use as experts.
                          If None, uses aggressive and skilled players.
        """
        from style_library import StyleLibrary
        
        if expert_styles is None:
            # Use skilled/aggressive players as experts
            expert_styles = [
                'sklansky_aggressive',
                'sklansky_regular', 
                'chen_aggressive',
                'chen_regular',
                'rule_based_aggressive',
                'rule_based_regular',
            ]
        
        states = []
        actions = []
        
        # Create environment
        env = PokerEnvironment(num_players=6, big_blind=100, starting_stack=10000)
        
        samples_collected = 0
        
        while samples_collected < num_samples:
            # Randomly select strategies for this game
            strategies = []
            expert_positions = []
            
            for i in range(6):
                if random.random() < 0.5:
                    # Use expert strategy
                    style_name = random.choice(expert_styles)
                    strategies.append(style_library.get_strategy(style_name))
                    expert_positions.append(i)
                else:
                    # Use random strategy from library
                    strategies.append(style_library.get_random_strategy())
            
            # Play game and collect preflop decisions from experts
            state = env.reset()
            
            while not state.is_terminal and state.street == Street.PREFLOP:
                current_player = state.current_player
                strategy = strategies[current_player]
                
                # Get action from strategy
                action = strategy.get_action(state, current_player)
                
                # If this is an expert, record the decision
                if current_player in expert_positions:
                    player_state = state.players[current_player]
                    
                    preflop_state = PreflopState(
                        hole_cards=tuple(player_state.hole_cards),
                        position=current_player,
                        pot_size=state.pot,
                        to_call=state.current_bet - player_state.current_bet,
                        stack=player_state.stack,
                        num_raises=sum(1 for a in state.action_history 
                                      if a[1] == Action.BET),
                        num_callers=sum(1 for a in state.action_history 
                                       if a[1] == Action.CALL),
                    )
                    
                    # Encode state
                    state_vec = preflop_state.to_vector(env.big_blind)
                    
                    # Encode action (0=fold, 1=call/check, 2=raise, 3=all-in)
                    if action.action_type == Action.FOLD:
                        action_idx = 0
                    elif action.action_type == Action.CALL:
                        action_idx = 1
                    elif action.action_type == Action.BET:
                        if action.amount >= player_state.stack * 0.9:
                            action_idx = 3  # All-in
                        else:
                            action_idx = 2  # Raise
                    elif action.action_type == Action.ALL_IN:
                        action_idx = 3
                    else:
                        action_idx = 1
                    
                    states.append(state_vec)
                    actions.append(action_idx)
                    samples_collected += 1
                    
                    if samples_collected >= num_samples:
                        break
                
                # Execute action
                state = env.step(action)
        
        return cls(states, actions)
    
    def save(self, path: str):
        """Save dataset to file."""
        torch.save({
            'states': self.states,
            'actions': self.actions,
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'PreflopExpertDataset':
        """Load dataset from file."""
        data = torch.load(path)
        dataset = cls.__new__(cls)
        dataset.states = data['states']
        dataset.actions = data['actions']
        return dataset


# =============================================================================
# Preflop Imitation Network
# =============================================================================

class PreflopImitationNetwork(nn.Module):
    """
    MLP network for preflop decision making via imitation learning.
    
    Architecture follows the paper's preflop network structure.
    """
    
    def __init__(
        self,
        input_dim: int = 29,
        hidden_dims: List[int] = [128, 128, 64],
        num_actions: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_actions))
        
        self.network = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns action logits."""
        return self.network(x)
    
    def get_action_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Returns action probabilities."""
        logits = self.forward(x)
        return self.softmax(logits)
    
    def get_action(self, state: PreflopState, big_blind: float = 100.0) -> int:
        """Get action for a single state."""
        self.eval()
        with torch.no_grad():
            state_vec = torch.FloatTensor(state.to_vector(big_blind)).unsqueeze(0)
            probs = self.get_action_probs(state_vec)
            return torch.argmax(probs, dim=-1).item()


class PreflopImitationTrainer:
    """Trainer for preflop imitation learning."""
    
    def __init__(
        self,
        network: PreflopImitationNetwork,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = 'cpu'
    ):
        self.network = network.to(device)
        self.device = device
        self.optimizer = optim.Adam(
            network.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch. Returns average loss."""
        self.network.train()
        total_loss = 0.0
        num_batches = 0
        
        for states, actions in dataloader:
            states = states.to(self.device)
            actions = actions.to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.network(states)
            loss = self.criterion(logits, actions)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Evaluate on dataset. Returns (loss, accuracy)."""
        self.network.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for states, actions in dataloader:
                states = states.to(self.device)
                actions = actions.to(self.device)
                
                logits = self.network(states)
                loss = self.criterion(logits, actions)
                
                total_loss += loss.item() * len(actions)
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == actions).sum().item()
                total += len(actions)
        
        return total_loss / max(total, 1), correct / max(total, 1)
    
    def train(
        self,
        train_dataset: PreflopExpertDataset,
        val_dataset: Optional[PreflopExpertDataset] = None,
        epochs: int = 100,
        batch_size: int = 256,
        early_stopping_patience: int = 10,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Full training loop with optional validation and early stopping.
        """
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False
            )
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            
            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_acc)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = self.network.state_dict().copy()
                else:
                    patience_counter += 1
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}: "
                          f"train_loss={train_loss:.4f}, "
                          f"val_loss={val_loss:.4f}, "
                          f"val_acc={val_acc:.4f}")
                
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}")
        
        # Restore best model
        if best_state is not None:
            self.network.load_state_dict(best_state)
        
        return history


# =============================================================================
# Later-Street Specialized Models
# =============================================================================

class LaterStreetState:
    """State encoding for post-flop streets."""
    
    def __init__(
        self,
        hole_cards: Tuple[Card, Card],
        community_cards: List[Card],
        position: int,
        pot_size: float,
        to_call: float,
        stack: float,
        street: Street,
        num_opponents: int,
        action_history: List[Tuple[int, Action, float]],
    ):
        self.hole_cards = hole_cards
        self.community_cards = community_cards
        self.position = position
        self.pot_size = pot_size
        self.to_call = to_call
        self.stack = stack
        self.street = street
        self.num_opponents = num_opponents
        self.action_history = action_history
    
    def to_vector(self, big_blind: float = 100.0) -> np.ndarray:
        """
        Encode later-street state as feature vector.
        
        Features (67 total):
        - Hole cards (4): rank/suit normalized
        - Community cards (10): up to 5 cards × 2 features
        - Hand strength features (6): hand rank, kicker, draws, etc.
        - Position (6): one-hot
        - Street (3): one-hot for flop/turn/river
        - Betting (10): pot, to_call, stack, pot_odds, etc.
        - Action history summary (8): aggression by street, etc.
        - Opponent info (4): num opponents, avg stack, etc.
        - Board texture (16): paired, suited, connected, etc.
        """
        features = []
        
        # Hole cards (4)
        for card in self.hole_cards:
            rank_val = card.rank if isinstance(card.rank, int) else card.rank.value
            suit_val = card.suit if isinstance(card.suit, int) else card.suit.value
            features.extend([rank_val / 14.0, suit_val / 4.0])

        # Community cards (10) - pad with zeros if fewer than 5
        for i in range(5):
            if i < len(self.community_cards):
                card = self.community_cards[i]
                rank_val = card.rank if isinstance(card.rank, int) else card.rank.value
                suit_val = card.suit if isinstance(card.suit, int) else card.suit.value
                features.extend([rank_val / 14.0, suit_val / 4.0])
            else:
                features.extend([0.0, 0.0])
        
        # Hand strength features (6)
        all_cards = list(self.hole_cards) + self.community_cards
        if len(all_cards) >= 5:
            hand_rank, kickers = evaluate_hand(all_cards)
            hand_strength = get_hand_strength_monte_carlo(
                list(self.hole_cards), self.community_cards, 
                self.num_opponents, num_simulations=100
            )
        else:
            hand_rank = HandRank.HIGH_CARD
            kickers = [0]
            hand_strength = 0.5
        
        features.extend([
            hand_rank.value / 9.0,
            kickers[0] / 14.0 if kickers else 0.0,
            hand_strength,
            self._count_flush_draw() / 4.0,
            self._count_straight_draw() / 4.0,
            self._count_overcards() / 2.0,
        ])
        
        # Position one-hot (6)
        position_onehot = [0.0] * 6
        position_onehot[self.position] = 1.0
        features.extend(position_onehot)
        
        # Street one-hot (3)
        street_onehot = [0.0, 0.0, 0.0]
        if self.street == Street.FLOP:
            street_onehot[0] = 1.0
        elif self.street == Street.TURN:
            street_onehot[1] = 1.0
        elif self.street == Street.RIVER:
            street_onehot[2] = 1.0
        features.extend(street_onehot)
        
        # Betting features (10)
        pot_norm = self.pot_size / big_blind
        to_call_norm = self.to_call / big_blind
        stack_norm = self.stack / big_blind
        pot_odds = self.to_call / (self.pot_size + self.to_call + 1e-6)
        stack_to_pot = self.stack / (self.pot_size + 1e-6)
        
        features.extend([
            min(pot_norm / 50.0, 1.0),
            min(to_call_norm / 20.0, 1.0),
            min(stack_norm / 200.0, 1.0),
            pot_odds,
            min(stack_to_pot / 20.0, 1.0),
            float(self.to_call > 0),  # facing bet
            min(self.to_call / (self.pot_size + 1e-6), 1.0),  # bet to pot ratio
            float(self.stack <= self.pot_size),  # short stacked
            float(self.to_call >= self.stack * 0.5),  # facing big bet
            float(self.to_call == 0),  # can check
        ])
        
        # Action history summary (8)
        preflop_raises = sum(1 for a in self.action_history 
                           if a[1] == Action.BET and len(self.community_cards) == 0)
        postflop_raises = sum(1 for a in self.action_history 
                            if a[1] == Action.BET and len(self.community_cards) > 0)
        total_calls = sum(1 for a in self.action_history if a[1] == Action.CALL)
        total_folds = sum(1 for a in self.action_history if a[1] == Action.FOLD)
        
        features.extend([
            min(preflop_raises / 3.0, 1.0),
            min(postflop_raises / 3.0, 1.0),
            min(total_calls / 6.0, 1.0),
            min(total_folds / 5.0, 1.0),
            min(len(self.action_history) / 20.0, 1.0),
            float(any(a[1] == Action.ALL_IN for a in self.action_history)),
            0.0,  # placeholder
            0.0,  # placeholder
        ])
        
        # Opponent info (4)
        features.extend([
            self.num_opponents / 5.0,
            0.5,  # avg opponent stack (placeholder)
            0.5,  # opponent aggression (placeholder)
            0.5,  # opponent tightness (placeholder)
        ])
        
        # Board texture (16)
        features.extend(self._get_board_texture())
        
        return np.array(features, dtype=np.float32)
    
    def _count_flush_draw(self) -> int:
        """Count cards to flush draw."""
        if len(self.community_cards) < 3:
            return 0
        
        suit_counts = defaultdict(int)
        for card in list(self.hole_cards) + self.community_cards:
            suit_counts[card.suit] += 1
        
        max_suited = max(suit_counts.values())
        if max_suited >= 5:
            return 0  # Already have flush
        elif max_suited == 4:
            return 1  # Flush draw
        elif max_suited == 3 and len(self.community_cards) == 3:
            return 2  # Backdoor flush draw
        return 0
    
    def _count_straight_draw(self) -> int:
        """Count straight draw outs (simplified)."""
        if len(self.community_cards) < 3:
            return 0

        all_cards = list(self.hole_cards) + self.community_cards
        ranks = sorted(set(c.rank if isinstance(c.rank, int) else c.rank.value for c in all_cards))

        # Check for 4 consecutive
        for i in range(len(ranks) - 3):
            if ranks[i+3] - ranks[i] == 3:
                return 2  # Open-ended
            elif ranks[i+3] - ranks[i] == 4:
                return 1  # Gutshot

        return 0

    def _count_overcards(self) -> int:
        """Count hole cards higher than board."""
        if len(self.community_cards) == 0:
            return 0

        board_ranks = [c.rank if isinstance(c.rank, int) else c.rank.value for c in self.community_cards]
        board_high = max(board_ranks)
        hole_ranks = [c.rank if isinstance(c.rank, int) else c.rank.value for c in self.hole_cards]
        return sum(1 for r in hole_ranks if r > board_high)
    
    def _get_board_texture(self) -> List[float]:
        """Analyze board texture (16 features)."""
        texture = [0.0] * 16
        
        if len(self.community_cards) < 3:
            return texture
        
        board = self.community_cards
        ranks = [c.rank if isinstance(c.rank, int) else c.rank.value for c in board]
        suits = [c.suit if isinstance(c.suit, int) else c.suit.value for c in board]
        
        # Paired board
        texture[0] = float(len(set(ranks)) < len(ranks))
        
        # Trips on board
        rank_counts = defaultdict(int)
        for r in ranks:
            rank_counts[r] += 1
        texture[1] = float(max(rank_counts.values()) >= 3)
        
        # Monotone (all same suit)
        texture[2] = float(len(set(suits)) == 1)
        
        # Two-tone (two suits)
        texture[3] = float(len(set(suits)) == 2)
        
        # Rainbow (all different suits, for flop)
        texture[4] = float(len(set(suits)) == len(suits) and len(suits) >= 3)
        
        # Connected (cards within 2 ranks)
        sorted_ranks = sorted(ranks)
        texture[5] = float(sorted_ranks[-1] - sorted_ranks[0] <= 4)
        
        # High board (highest card >= J)
        texture[6] = float(max(ranks) >= 11)
        
        # Low board (highest card <= 8)
        texture[7] = float(max(ranks) <= 8)
        
        # Broadway potential (multiple 10+)
        texture[8] = float(sum(1 for r in ranks if r >= 10) >= 2)
        
        # Wheel potential (A-5 cards)
        wheel_cards = [14, 2, 3, 4, 5]
        texture[9] = float(sum(1 for r in ranks if r in wheel_cards) >= 2)
        
        # Flush possible (3+ same suit)
        suit_counts = defaultdict(int)
        for s in suits:
            suit_counts[s] += 1
        texture[10] = float(max(suit_counts.values()) >= 3)
        
        # Straight possible
        texture[11] = float(sorted_ranks[-1] - sorted_ranks[0] <= 4)
        
        # Dry board (not connected, not suited)
        texture[12] = float(texture[5] == 0 and texture[2] == 0 and texture[3] == 0)
        
        # Wet board (connected and/or suited)
        texture[13] = float(texture[5] == 1 or texture[2] == 1)
        
        # Ace high board
        texture[14] = float(14 in ranks)
        
        # Face card heavy (2+ face cards)
        texture[15] = float(sum(1 for r in ranks if r >= 11) >= 2)
        
        return texture


class LaterStreetNetwork(nn.Module):
    """
    Network for post-flop decision making.
    
    Can be specialized for a specific street or used across all streets.
    """
    
    def __init__(
        self,
        input_dim: int = 67,
        hidden_dims: List[int] = [256, 256, 128],
        num_actions: int = 4,
        dropout: float = 0.2,
        use_lstm: bool = True,
        lstm_hidden: int = 64
    ):
        super().__init__()
        
        self.use_lstm = use_lstm
        
        # Main MLP encoder
        mlp_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            mlp_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        self.mlp_encoder = nn.Sequential(*mlp_layers)
        
        # Optional LSTM for action sequence
        if use_lstm:
            self.action_lstm = nn.LSTM(
                input_size=3,  # action type, amount, player
                hidden_size=lstm_hidden,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
            )
            # Create separate heads for with/without LSTM
            self.policy_head_with_lstm = nn.Sequential(
                nn.Linear(prev_dim + lstm_hidden * 2, 64),
                nn.ReLU(),
                nn.Linear(64, num_actions),
            )
            self.policy_head_no_lstm = nn.Sequential(
                nn.Linear(prev_dim, 64),
                nn.ReLU(),
                nn.Linear(64, num_actions),
            )
            self.value_head_with_lstm = nn.Sequential(
                nn.Linear(prev_dim + lstm_hidden * 2, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
            self.value_head_no_lstm = nn.Sequential(
                nn.Linear(prev_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
        else:
            self.action_lstm = None
            # Single head when no LSTM
            self.policy_head = nn.Sequential(
                nn.Linear(prev_dim, 64),
                nn.ReLU(),
                nn.Linear(64, num_actions),
            )
            self.value_head = nn.Sequential(
                nn.Linear(prev_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(
        self, 
        state: torch.Tensor,
        action_sequence: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            state: State features [batch, input_dim]
            action_sequence: Optional action history [batch, seq_len, 3]
        
        Returns:
            policy_logits: [batch, num_actions]
            value: [batch, 1]
        """
        mlp_out = self.mlp_encoder(state)

        if self.use_lstm and action_sequence is not None:
            lstm_out, _ = self.action_lstm(action_sequence)
            lstm_final = lstm_out[:, -1, :]  # Take last hidden state
            combined = torch.cat([mlp_out, lstm_final], dim=-1)
            policy_logits = self.policy_head_with_lstm(combined)
            value = self.value_head_with_lstm(combined)
        else:
            if self.use_lstm:
                # LSTM enabled but no action sequence provided
                policy_logits = self.policy_head_no_lstm(mlp_out)
                value = self.value_head_no_lstm(mlp_out)
            else:
                # No LSTM at all
                policy_logits = self.policy_head(mlp_out)
                value = self.value_head(mlp_out)

        return policy_logits, value
    
    def get_action_probs(
        self,
        state: torch.Tensor,
        action_sequence: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get action probabilities."""
        logits, _ = self.forward(state, action_sequence)
        return self.softmax(logits)


class StreetSpecificModels:
    """
    Container for street-specific models (Flop, Turn, River).
    
    Each street can have its own specialized network.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        
        # Create street-specific networks
        self.flop_network = LaterStreetNetwork(
            input_dim=67,
            hidden_dims=[256, 256, 128],
        ).to(device)
        
        self.turn_network = LaterStreetNetwork(
            input_dim=67,
            hidden_dims=[256, 256, 128],
        ).to(device)
        
        self.river_network = LaterStreetNetwork(
            input_dim=67,
            hidden_dims=[256, 256, 128],
        ).to(device)
        
        self.networks = {
            Street.FLOP: self.flop_network,
            Street.TURN: self.turn_network,
            Street.RIVER: self.river_network,
        }
    
    def get_network(self, street: Street) -> LaterStreetNetwork:
        """Get network for specific street."""
        return self.networks.get(street, self.flop_network)
    
    def get_action(
        self,
        state: LaterStreetState,
        big_blind: float = 100.0
    ) -> Tuple[int, float]:
        """
        Get action for a later-street state.
        
        Returns:
            action_idx: 0=fold/check, 1=call, 2=bet/raise, 3=all-in
            value: Estimated state value
        """
        network = self.get_network(state.street)
        network.eval()
        
        with torch.no_grad():
            state_vec = torch.FloatTensor(
                state.to_vector(big_blind)
            ).unsqueeze(0).to(self.device)
            
            logits, value = network(state_vec)
            probs = torch.softmax(logits, dim=-1)
            action = torch.argmax(probs, dim=-1).item()
            
            return action, value.item()
    
    def save(self, path: str):
        """Save all street models."""
        torch.save({
            'flop': self.flop_network.state_dict(),
            'turn': self.turn_network.state_dict(),
            'river': self.river_network.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load all street models."""
        data = torch.load(path)
        self.flop_network.load_state_dict(data['flop'])
        self.turn_network.load_state_dict(data['turn'])
        self.river_network.load_state_dict(data['river'])


# =============================================================================
# Combined AMP3 with Preflop Imitation and Later-Street Models
# =============================================================================

class AMP3WithSpecializedModels:
    """
    Full AMP3 agent combining:
    1. Preflop imitation learning network
    2. Later-street specialized models
    3. OSM for opponent modeling
    4. Actor-Critic for RL fine-tuning
    """
    
    def __init__(
        self,
        preflop_network: PreflopImitationNetwork,
        street_models: StreetSpecificModels,
        osm_network=None,  # StyleModelingNetwork from osm_network.py
        amp3_actor=None,   # AMP3Actor from amp3_network.py
        device: str = 'cpu'
    ):
        self.preflop_network = preflop_network.to(device)
        self.street_models = street_models
        self.osm_network = osm_network
        self.amp3_actor = amp3_actor
        self.device = device
        
        # Mode: 'imitation' uses preflop + street models
        #       'amp3' uses full AMP3 actor
        self.mode = 'imitation'
    
    def set_mode(self, mode: str):
        """Set decision mode ('imitation' or 'amp3')."""
        assert mode in ['imitation', 'amp3']
        self.mode = mode
    
    def get_action(
        self,
        game_state,  # GameState from poker_core.py
        player_idx: int,
        opponent_styles: Optional[np.ndarray] = None
    ) -> Action:
        """
        Get action for current game state.
        
        Args:
            game_state: Current game state
            player_idx: Index of acting player
            opponent_styles: Optional pre-computed opponent styles [6, 4]
        
        Returns:
            Action to take
        """
        player = game_state.players[player_idx]
        
        if self.mode == 'amp3' and self.amp3_actor is not None:
            # Use full AMP3 actor
            return self._get_amp3_action(game_state, player_idx, opponent_styles)
        
        # Use specialized models
        if game_state.street == Street.PREFLOP:
            return self._get_preflop_action(game_state, player_idx)
        else:
            return self._get_later_street_action(game_state, player_idx)
    
    def _get_preflop_action(self, game_state, player_idx: int) -> Action:
        """Get preflop action using imitation network."""
        player = game_state.players[player_idx]
        
        preflop_state = PreflopState(
            hole_cards=tuple(player.hole_cards),
            position=player_idx,
            pot_size=game_state.pot,
            to_call=game_state.current_bet - player.current_bet,
            stack=player.stack,
            num_raises=sum(1 for a in game_state.action_history 
                          if a[1] == Action.BET),
            num_callers=sum(1 for a in game_state.action_history 
                           if a[1] == Action.CALL),
        )
        
        action_idx = self.preflop_network.get_action(
            preflop_state, 
            big_blind=100.0  # Assuming standard BB
        )
        
        return self._decode_action(action_idx, game_state, player_idx)
    
    def _get_later_street_action(self, game_state, player_idx: int) -> Action:
        """Get post-flop action using street-specific model."""
        player = game_state.players[player_idx]
        
        later_state = LaterStreetState(
            hole_cards=tuple(player.hole_cards),
            community_cards=game_state.community_cards,
            position=player_idx,
            pot_size=game_state.pot,
            to_call=game_state.current_bet - player.current_bet,
            stack=player.stack,
            street=game_state.street,
            num_opponents=sum(1 for p in game_state.players if p.is_active) - 1,
            action_history=game_state.action_history,
        )
        
        action_idx, _ = self.street_models.get_action(later_state)
        return self._decode_action(action_idx, game_state, player_idx)
    
    def _get_amp3_action(
        self, 
        game_state, 
        player_idx: int,
        opponent_styles: Optional[np.ndarray]
    ) -> Action:
        """Get action using full AMP3 actor."""
        # Import here to avoid circular dependency
        from amp3_network import encode_amp3_state
        
        if opponent_styles is None and self.osm_network is not None:
            # Predict opponent styles
            opponent_styles = np.zeros((6, 4))
            # Would need game history to predict properly
            # Using default values for now
            opponent_styles[:, 0] = 0.3  # VPIP
            opponent_styles[:, 1] = 0.2  # PFR
            opponent_styles[:, 2] = 0.5  # AFq
            opponent_styles[:, 3] = 0.3  # WTSD
        
        state = encode_amp3_state(game_state, player_idx, opponent_styles)
        state_vec = state.to_vector()
        
        self.amp3_actor.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
            logits = self.amp3_actor(state_tensor)
            probs = torch.softmax(logits, dim=-1)
            action_idx = torch.argmax(probs, dim=-1).item()
        
        return self._decode_action(action_idx, game_state, player_idx)
    
    def _decode_action(
        self, 
        action_idx: int, 
        game_state, 
        player_idx: int
    ) -> Action:
        """Convert action index to Action object."""
        player = game_state.players[player_idx]
        to_call = game_state.current_bet - player.current_bet
        
        if action_idx == 0:  # Fold/Check
            if to_call == 0:
                return Action(Action.CALL, 0)  # Check
            else:
                return Action(Action.FOLD, 0)
        elif action_idx == 1:  # Call
            return Action(Action.CALL, min(to_call, player.stack))
        elif action_idx == 2:  # Bet/Raise (pot-sized)
            bet_amount = game_state.pot + to_call
            bet_amount = min(bet_amount, player.stack)
            if bet_amount <= to_call:
                return Action(Action.CALL, to_call)
            return Action(Action.BET, bet_amount)
        else:  # All-in
            return Action(Action.ALL_IN, player.stack)
    
    def save(self, path: str):
        """Save all models."""
        torch.save({
            'preflop': self.preflop_network.state_dict(),
            'amp3_actor': self.amp3_actor.state_dict() if self.amp3_actor else None,
        }, path + '_networks.pt')
        self.street_models.save(path + '_streets.pt')
    
    def load(self, path: str):
        """Load all models."""
        data = torch.load(path + '_networks.pt')
        self.preflop_network.load_state_dict(data['preflop'])
        if data['amp3_actor'] is not None and self.amp3_actor is not None:
            self.amp3_actor.load_state_dict(data['amp3_actor'])
        self.street_models.load(path + '_streets.pt')


if __name__ == '__main__':
    # Quick test
    print("Testing Preflop Imitation Module...")
    
    # Create preflop network
    preflop_net = PreflopImitationNetwork()
    print(f"Preflop network parameters: {sum(p.numel() for p in preflop_net.parameters())}")
    
    # Create later-street models
    street_models = StreetSpecificModels()
    print(f"Flop network parameters: {sum(p.numel() for p in street_models.flop_network.parameters())}")
    
    # Test preflop state encoding
    c1 = Card(Rank.ACE, Suit.SPADES)
    c2 = Card(Rank.KING, Suit.SPADES)
    
    preflop_state = PreflopState(
        hole_cards=(c1, c2),
        position=0,
        pot_size=150,
        to_call=100,
        stack=9900,
        num_raises=1,
        num_callers=0,
    )
    
    vec = preflop_state.to_vector()
    print(f"Preflop state vector shape: {vec.shape}")
    
    # Test network forward
    with torch.no_grad():
        x = torch.FloatTensor(vec).unsqueeze(0)
        logits = preflop_net(x)
        probs = torch.softmax(logits, dim=-1)
        print(f"Action probabilities: {probs.numpy()}")
    
    print("\nPreflop Imitation Module tests passed!")
