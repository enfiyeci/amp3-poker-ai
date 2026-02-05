"""
amp3_network.py - Adaptive Multi-player Poker Policy (AMP3) Network
Implements Actor-Critic RL framework for adaptive poker decision making.
Integrates opponent style predictions from OSM into the policy network.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Categorical
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from collections import deque
import random

from poker_core import Card, Action, Street, GameState
from osm_network import StyleModelingNetwork


# =============================================================================
# State Encoding for AMP3
# =============================================================================

@dataclass  
class AMP3State:
    """
    Complete state representation for AMP3 policy network.
    Combines personal info, public info, action history, and opponent styles.
    """
    
    # Personal information (4 values)
    hole_cards: np.ndarray      # shape (4,) - 2 cards * 2 features
    
    # Public information
    community_cards: np.ndarray # shape (10,) - 5 cards * 2 features
    pot: float
    current_bet: float
    to_call: float
    stack: float
    position_idx: int           # 0-5 for position encoding
    
    # All players' stacks and bets (12 values)
    player_stacks: np.ndarray   # shape (6,)
    player_bets: np.ndarray     # shape (6,)
    
    # Action history encoded (sequence)
    action_history: np.ndarray  # shape (max_actions, 2)
    
    # Opponent style predictions (24 values = 5 opponents * 4 features + 4 for self)
    style_features: np.ndarray  # shape (24,)


def encode_amp3_state(
    state: GameState,
    seat: int,
    style_predictions: np.ndarray,  # shape (6, 4) - all players' style features
    max_actions: int = 48
) -> AMP3State:
    """
    Encode full game state for AMP3 network.
    """
    from osm_network import encode_cards, encode_action
    
    player = state.players[seat]
    
    # Encode hole cards
    hole_enc = encode_cards(player.hole_cards, 2)  # (4,)
    
    # Encode community cards
    community_enc = encode_cards(state.community_cards, 5)  # (10,)
    
    # Encode player stacks and bets
    stacks = np.array([p.stack for p in state.players], dtype=np.float32)
    bets = np.array([p.bet_this_street for p in state.players], dtype=np.float32)
    
    # Normalize by big blind
    bb = state.big_blind
    stacks_norm = stacks / (100 * bb)  # Assume ~100bb starting
    bets_norm = bets / (10 * bb)
    pot_norm = state.pot / (100 * bb)
    current_bet_norm = state.current_bet / (10 * bb)
    to_call_norm = state.get_to_call(seat) / (10 * bb)
    stack_norm = player.stack / (100 * bb)
    
    # Encode action history
    action_enc = np.zeros((max_actions, 2), dtype=np.float32)
    for i, (player_idx, action, _) in enumerate(state.action_history[:max_actions]):
        is_hero = 1.0 if player_idx == seat else 0.0
        action_norm = action.value / 3.0
        action_enc[i] = [is_hero, action_norm]
    
    # Flatten style features
    style_flat = style_predictions.flatten()  # (24,)
    
    # Position encoding (relative to button)
    position_idx = (seat - state.button_seat) % state.num_players
    
    return AMP3State(
        hole_cards=hole_enc,
        community_cards=community_enc,
        pot=pot_norm,
        current_bet=current_bet_norm,
        to_call=to_call_norm,
        stack=stack_norm,
        position_idx=position_idx,
        player_stacks=stacks_norm,
        player_bets=bets_norm,
        action_history=action_enc,
        style_features=style_flat
    )


def encode_critic_state(
    state: GameState,
    seat: int,
    style_predictions: np.ndarray,
    max_actions: int = 48
) -> Dict[str, np.ndarray]:
    """
    Encode state for critic (global view with all hole cards).
    For training, critic sees all players' cards (omniscient for learning).
    """
    from osm_network import encode_cards

    # Encode all players' hole cards (not just hero's)
    all_holes = []
    for player in state.players:
        if player.hole_cards:
            hole_enc = encode_cards(player.hole_cards, 2)
        else:
            hole_enc = np.zeros(4, dtype=np.float32)
        all_holes.append(hole_enc)
    all_holes = np.concatenate(all_holes)  # (24,) = 6 players * 4 features

    # Public info (same as actor)
    from osm_network import encode_cards
    community_enc = encode_cards(state.community_cards, 5)

    stacks = np.array([p.stack for p in state.players], dtype=np.float32)
    bets = np.array([p.bet_this_street for p in state.players], dtype=np.float32)

    bb = state.big_blind
    stacks_norm = stacks / (100 * bb)
    bets_norm = bets / (10 * bb)

    public = np.concatenate([community_enc, stacks_norm, bets_norm])  # (22,)

    # Position
    position = np.zeros(6, dtype=np.float32)
    position_idx = (seat - state.button_seat) % state.num_players
    position[position_idx] = 1.0

    # Action history
    action_enc = np.zeros((max_actions, 2), dtype=np.float32)
    for i, (player_idx, action, _) in enumerate(state.action_history[:max_actions]):
        is_hero = 1.0 if player_idx == seat else 0.0
        action_norm = action.value / 3.0
        action_enc[i] = [is_hero, action_norm]

    return {
        'all_holes': all_holes,
        'public': public,
        'position': position,
        'action_history': action_enc
    }


def state_to_tensors(amp3_state: AMP3State, device: str = 'cpu') -> Dict[str, torch.Tensor]:
    """Convert AMP3State to tensors for network input"""
    
    # Personal info (4 + 4 scalar values = 8)
    personal = np.concatenate([
        amp3_state.hole_cards,  # 4
        np.array([amp3_state.pot, amp3_state.current_bet, 
                  amp3_state.to_call, amp3_state.stack], dtype=np.float32)  # 4
    ])
    
    # Public info (10 + 12 = 22)
    public = np.concatenate([
        amp3_state.community_cards,  # 10
        amp3_state.player_stacks,    # 6
        amp3_state.player_bets       # 6
    ])
    
    # Position one-hot (6)
    position = np.zeros(6, dtype=np.float32)
    position[amp3_state.position_idx] = 1.0
    
    return {
        'personal': torch.from_numpy(personal).to(device),
        'public': torch.from_numpy(public).to(device),
        'position': torch.from_numpy(position).to(device),
        'action_history': torch.from_numpy(amp3_state.action_history).to(device),
        'style_features': torch.from_numpy(amp3_state.style_features).to(device)
    }


# =============================================================================
# Actor Network
# =============================================================================

class AMP3Actor(nn.Module):
    """
    Actor network for AMP3.
    Takes game state + opponent style predictions and outputs action probabilities.
    
    Architecture:
    - FC layers for personal/public info
    - Position encoding
    - LSTM for action history
    - Style features integration
    - Combined policy head
    """
    
    def __init__(self,
                 personal_dim: int = 8,      # hole cards + pot/bet/call/stack
                 public_dim: int = 22,       # community cards + all stacks/bets
                 position_dim: int = 6,      # one-hot position
                 action_dim: int = 2,        # action history features
                 style_dim: int = 24,        # 6 players * 4 style features
                 hidden_dim: int = 128,
                 lstm_hidden: int = 128,
                 lstm_layers: int = 3,
                 num_actions: int = 4,       # fold, call, bet/raise, all-in
                 dropout: float = 0.1):
        
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Personal info encoder
        self.personal_encoder = nn.Sequential(
            nn.Linear(personal_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1)
        )
        
        # Public info encoder
        self.public_encoder = nn.Sequential(
            nn.Linear(public_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1)
        )
        
        # Position encoder
        self.position_encoder = nn.Sequential(
            nn.Linear(position_dim, hidden_dim // 4),
            nn.LeakyReLU(0.1)
        )
        
        # Style features encoder
        self.style_encoder = nn.Sequential(
            nn.Linear(style_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1)
        )
        
        # Action history LSTM (Bidirectional)
        self.action_lstm = nn.LSTM(
            input_size=action_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Combined feature dimension
        # personal(64) + public(64) + position(32) + style(64) + lstm(256) = 480
        combined_dim = hidden_dim // 2 * 3 + hidden_dim // 4 + lstm_hidden * 2
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim // 2, num_actions)
        )
    
    def forward(self,
                personal: torch.Tensor,
                public: torch.Tensor,
                position: torch.Tensor,
                action_history: torch.Tensor,
                style_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            personal: (batch, 8)
            public: (batch, 22)
            position: (batch, 6)
            action_history: (batch, seq_len, 2)
            style_features: (batch, 24)
        
        Returns:
            action_logits: (batch, num_actions)
        """
        # Encode individual components
        personal_enc = self.personal_encoder(personal)
        public_enc = self.public_encoder(public)
        position_enc = self.position_encoder(position)
        style_enc = self.style_encoder(style_features)
        
        # Encode action history with bidirectional LSTM
        lstm_out, (h_n, _) = self.action_lstm(action_history)
        # Concatenate forward and backward final hidden states
        action_enc = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        
        # Combine all features
        combined = torch.cat([
            personal_enc, public_enc, position_enc, style_enc, action_enc
        ], dim=-1)
        
        # Output action logits
        logits = self.policy_head(combined)
        return logits
    
    def get_action_probs(self, **kwargs) -> torch.Tensor:
        """Get action probabilities"""
        logits = self.forward(**kwargs)
        return F.softmax(logits, dim=-1)
    
    def get_action(self, **kwargs) -> Tuple[int, torch.Tensor]:
        """Sample action from policy"""
        probs = self.get_action_probs(**kwargs)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob


# =============================================================================
# Critic Network
# =============================================================================

class AMP3Critic(nn.Module):
    """
    Critic network for AMP3.
    Takes GLOBAL game state (all hole cards visible) and outputs Q-value.
    Does NOT use opponent style predictions (evaluates true state value).
    """
    
    def __init__(self,
                 all_holes_dim: int = 24,    # 6 players * 4 (2 cards * 2 features)
                 public_dim: int = 22,       # community cards + all stacks/bets
                 position_dim: int = 6,
                 action_dim: int = 2,
                 hidden_dim: int = 128,
                 lstm_hidden: int = 128,
                 lstm_layers: int = 3,
                 dropout: float = 0.1):
        
        super().__init__()
        
        # All hole cards encoder (global info)
        self.holes_encoder = nn.Sequential(
            nn.Linear(all_holes_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1)
        )
        
        # Public info encoder
        self.public_encoder = nn.Sequential(
            nn.Linear(public_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1)
        )
        
        # Position encoder
        self.position_encoder = nn.Sequential(
            nn.Linear(position_dim, hidden_dim // 4),
            nn.LeakyReLU(0.1)
        )
        
        # Action history LSTM
        self.action_lstm = nn.LSTM(
            input_size=action_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Combined dimension
        combined_dim = hidden_dim // 2 * 2 + hidden_dim // 4 + lstm_hidden * 2
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self,
                all_hole_cards: torch.Tensor,
                public: torch.Tensor,
                position: torch.Tensor,
                action_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            all_hole_cards: (batch, 24) - all players' hole cards
            public: (batch, 22)
            position: (batch, 6)
            action_history: (batch, seq_len, 2)
        
        Returns:
            value: (batch, 1)
        """
        holes_enc = self.holes_encoder(all_hole_cards)
        public_enc = self.public_encoder(public)
        position_enc = self.position_encoder(position)
        
        lstm_out, (h_n, _) = self.action_lstm(action_history)
        action_enc = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        
        combined = torch.cat([holes_enc, public_enc, position_enc, action_enc], dim=-1)
        value = self.value_head(combined)
        
        return value


# =============================================================================
# Experience Replay Buffer
# =============================================================================

@dataclass
class Experience:
    """Single experience tuple for replay buffer"""
    state: Dict[str, np.ndarray]      # Actor input state
    critic_state: Dict[str, np.ndarray]  # Critic input state (global)
    action: int
    reward: float
    next_state: Optional[Dict[str, np.ndarray]]
    next_critic_state: Optional[Dict[str, np.ndarray]]
    done: bool
    log_prob: float


class ReplayBuffer:
    """Experience replay buffer for AMP3 training"""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience: Experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()


# =============================================================================
# AMP3 Agent
# =============================================================================

class AMP3Agent:
    """
    Complete AMP3 agent combining OSM + Actor-Critic.
    Predicts opponent styles and makes adaptive decisions.
    """
    
    def __init__(self,
                 osm_model: Optional[StyleModelingNetwork] = None,
                 actor: Optional[AMP3Actor] = None,
                 critic: Optional[AMP3Critic] = None,
                 actor_lr: float = 1e-4,
                 critic_lr: float = 1e-4,
                 gamma: float = 0.99,
                 device: str = 'cpu'):
        
        self.device = device
        self.gamma = gamma
        
        # Initialize networks
        if osm_model is None:
            osm_model = StyleModelingNetwork()
        if actor is None:
            actor = AMP3Actor()
        if critic is None:
            critic = AMP3Critic()
        
        self.osm_model = osm_model.to(device)
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        
        # Target network for stable learning
        self.critic_target = AMP3Critic().to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer()
        
        # Training stats
        self.actor_losses = []
        self.critic_losses = []
        
        # Style prediction cache
        self._style_cache: Dict[int, np.ndarray] = {}
    
    def predict_opponent_styles(self, 
                                 game_histories: Dict[int, List[Dict]],
                                 state: GameState) -> np.ndarray:
        """
        Predict style features for all players.
        game_histories: Dict mapping player seat to list of their past games
        Returns: (6, 4) array of style features
        """
        from osm_network import StylePredictor
        
        self.osm_model.eval()
        predictor = StylePredictor(self.osm_model, self.device)
        
        styles = np.zeros((6, 4), dtype=np.float32)
        
        for seat in range(6):
            if seat in game_histories and len(game_histories[seat]) > 0:
                mean_style, _ = predictor.predict_aggregate(
                    game_histories[seat], seat
                )
                styles[seat] = mean_style
            else:
                # Default style (average player)
                styles[seat] = np.array([0.25, 0.15, 0.4, 0.3], dtype=np.float32)
        
        return styles
    
    def get_action(self, 
                   state: GameState, 
                   seat: int,
                   style_predictions: np.ndarray) -> Tuple[Action, float, float]:
        """
        Get action for the current state.
        Returns (action, bet_amount, log_prob)
        """
        self.actor.eval()
        
        # Encode state
        amp3_state = encode_amp3_state(state, seat, style_predictions)
        tensors = state_to_tensors(amp3_state, self.device)
        
        # Batch dimension
        for k in tensors:
            tensors[k] = tensors[k].unsqueeze(0)
        
        with torch.no_grad():
            action_idx, log_prob = self.actor.get_action(
                personal=tensors['personal'],
                public=tensors['public'],
                position=tensors['position'],
                action_history=tensors['action_history'],
                style_features=tensors['style_features']
            )
        
        # Convert action index to Action enum and bet amount
        action, amount = self._decode_action(action_idx, state, seat)
        
        return action, amount, log_prob.item()
    
    def _decode_action(self, 
                       action_idx: int, 
                       state: GameState, 
                       seat: int) -> Tuple[Action, float]:
        """Convert action index to Action enum and amount"""
        player = state.players[seat]
        to_call = state.get_to_call(seat)
        
        if action_idx == 0:  # Fold
            if to_call == 0:
                return Action.CALL, 0  # Check instead
            return Action.FOLD, 0
        
        elif action_idx == 1:  # Call/Check
            return Action.CALL, min(to_call, player.stack)
        
        elif action_idx == 2:  # Small bet/raise (pot-size)
            bet_amount = state.pot * 0.75 + to_call
            return Action.BET, min(bet_amount, player.stack)
        
        else:  # action_idx == 3: Large bet/raise or all-in
            if player.stack <= state.pot * 2:
                return Action.ALL_IN, player.stack
            bet_amount = state.pot * 1.5 + to_call
            return Action.BET, min(bet_amount, player.stack)
    
    def update(self, batch_size: int = 256) -> Tuple[float, float]:
        """
        Update actor and critic networks from replay buffer.
        Returns (actor_loss, critic_loss)
        """
        if len(self.replay_buffer) < batch_size:
            return 0.0, 0.0
        
        experiences = self.replay_buffer.sample(batch_size)
        
        # Prepare batched tensors
        states = self._batch_states([e.state for e in experiences])
        critic_states = self._batch_critic_states([e.critic_state for e in experiences])
        actions = torch.tensor([e.action for e in experiences], device=self.device)
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32, device=self.device)
        dones = torch.tensor([e.done for e in experiences], dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor([e.log_prob for e in experiences], dtype=torch.float32, device=self.device)
        
        # Filter non-terminal experiences for next state values
        non_terminal_mask = ~dones.bool()
        non_terminal_next_critic = [e.next_critic_state for e, done in zip(experiences, dones) if not done]
        
        # Critic update
        self.critic.train()
        current_q = self.critic(
            critic_states['all_holes'],
            critic_states['public'],
            critic_states['position'],
            critic_states['action_history']
        ).squeeze(-1)
        
        # Compute target Q values
        next_q = torch.zeros(batch_size, device=self.device)
        if len(non_terminal_next_critic) > 0:
            next_critic_states = self._batch_critic_states(non_terminal_next_critic)
            with torch.no_grad():
                next_q[non_terminal_mask] = self.critic_target(
                    next_critic_states['all_holes'],
                    next_critic_states['public'],
                    next_critic_states['position'],
                    next_critic_states['action_history']
                ).squeeze(-1)
        
        target_q = rewards + self.gamma * next_q * (1 - dones)
        
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Actor update (policy gradient)
        self.actor.train()
        
        logits = self.actor(
            states['personal'],
            states['public'],
            states['position'],
            states['action_history'],
            states['style_features']
        )
        
        # Compute advantages
        with torch.no_grad():
            advantages = target_q - current_q.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy loss with entropy bonus
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        
        actor_loss = -(action_log_probs * advantages).mean() - 0.01 * entropy
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Update target network (soft update)
        tau = 0.005
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        
        return actor_loss.item(), critic_loss.item()
    
    def _batch_states(self, states: List[Dict]) -> Dict[str, torch.Tensor]:
        """Batch actor states"""
        return {
            'personal': torch.stack([torch.from_numpy(s['personal']) for s in states]).to(self.device),
            'public': torch.stack([torch.from_numpy(s['public']) for s in states]).to(self.device),
            'position': torch.stack([torch.from_numpy(s['position']) for s in states]).to(self.device),
            'action_history': torch.stack([torch.from_numpy(s['action_history']) for s in states]).to(self.device),
            'style_features': torch.stack([torch.from_numpy(s['style_features']) for s in states]).to(self.device),
        }
    
    def _batch_critic_states(self, states: List[Dict]) -> Dict[str, torch.Tensor]:
        """Batch critic states"""
        return {
            'all_holes': torch.stack([torch.from_numpy(s['all_holes']) for s in states]).to(self.device),
            'public': torch.stack([torch.from_numpy(s['public']) for s in states]).to(self.device),
            'position': torch.stack([torch.from_numpy(s['position']) for s in states]).to(self.device),
            'action_history': torch.stack([torch.from_numpy(s['action_history']) for s in states]).to(self.device),
        }
    
    def save(self, path: str):
        """Save all model weights"""
        torch.save({
            'osm_state_dict': self.osm_model.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load all model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.osm_model.load_state_dict(checkpoint['osm_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])


# =============================================================================
# Reward Function
# =============================================================================

def calculate_reward(
    action: Action,
    amount: float,
    pot_before: float,
    is_winner: bool,
    final_pot: float,
    num_winners: int,
    reached_showdown: bool
) -> float:
    """
    Calculate reward for an action as defined in the paper.
    
    reward = 
        0                    if fold
        -chips_added         if call/bet
        (pot/winners)^2      if win
        -pot^2               if lose at showdown
    """
    if action == Action.FOLD:
        return 0.0
    
    # Cost of calling/betting
    reward = -amount
    
    # Terminal rewards
    if is_winner:
        share = final_pot / max(1, num_winners)
        reward += share ** 0.5  # Using sqrt instead of square for stability
    elif reached_showdown:
        reward -= final_pot ** 0.5  # Penalty for losing at showdown
    
    return reward


if __name__ == "__main__":
    # Test AMP3 networks
    print("Testing AMP3 Networks...")
    
    # Test Actor
    actor = AMP3Actor()
    batch_size = 32
    
    personal = torch.randn(batch_size, 8)
    public = torch.randn(batch_size, 22)
    position = torch.randn(batch_size, 6)
    action_history = torch.randn(batch_size, 48, 2)
    style_features = torch.randn(batch_size, 24)
    
    logits = actor(personal, public, position, action_history, style_features)
    print(f"Actor output shape: {logits.shape}")  # Should be (32, 4)
    
    # Test Critic
    critic = AMP3Critic()
    all_holes = torch.randn(batch_size, 24)
    
    value = critic(all_holes, public, position, action_history)
    print(f"Critic output shape: {value.shape}")  # Should be (32, 1)
    
    # Test AMP3Agent
    print("\nTesting AMP3Agent...")
    agent = AMP3Agent()
    print("Agent initialized successfully")
