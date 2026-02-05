"""
cfr_networks.py - Neural Network CFR Approximation
Implements Deep CFR and Neural Fictitious Self-Play for poker strategy computation.
Based on techniques from Libratus and Pluribus.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict
from collections import defaultdict
from dataclasses import dataclass
import random

from poker_core import Card, Action, Street, GameState, evaluate_hand


# =============================================================================
# Information Set Encoding
# =============================================================================

@dataclass
class InfoSet:
    """
    Information set representation for poker.
    Contains all information available to a player at a decision point.
    """
    player: int
    hole_cards: Tuple[int, int]  # Card indices
    community_cards: Tuple[int, ...]  # Card indices
    pot: float
    to_call: float
    stack: float
    action_sequence: Tuple[Tuple[int, int], ...]  # (player, action) tuples
    street: Street
    position: int
    
    def __hash__(self):
        return hash((
            self.player,
            self.hole_cards,
            self.community_cards,
            round(self.pot, 2),
            round(self.to_call, 2),
            self.action_sequence,
            self.street,
            self.position
        ))
    
    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for neural network"""
        from osm_network import encode_cards
        
        # Encode hole cards (4 features)
        hole_cards_list = [Card.from_int(c) for c in self.hole_cards]
        hole_enc = encode_cards(hole_cards_list, 2)
        
        # Encode community cards (10 features)
        comm_cards_list = [Card.from_int(c) for c in self.community_cards]
        comm_enc = encode_cards(comm_cards_list, 5)
        
        # Betting info (4 features)
        betting = np.array([
            self.pot / 100.0,
            self.to_call / 100.0,
            self.stack / 100.0,
            self.street.value / 3.0
        ], dtype=np.float32)
        
        # Position one-hot (6 features)
        position = np.zeros(6, dtype=np.float32)
        position[self.position % 6] = 1.0
        
        # Action sequence encoding (48 features for 24 max actions)
        action_enc = np.zeros(48, dtype=np.float32)
        for i, (player, action) in enumerate(self.action_sequence[:24]):
            action_enc[i * 2] = player / 5.0
            action_enc[i * 2 + 1] = action / 3.0
        
        return np.concatenate([hole_enc, comm_enc, betting, position, action_enc])


# =============================================================================
# CFR Value Network
# =============================================================================

class CFRValueNetwork(nn.Module):
    """
    Neural network to approximate counterfactual values.
    Used in Deep CFR to generalize across similar information sets.
    """
    
    def __init__(self,
                 input_dim: int = 72,  # InfoSet vector size
                 hidden_dim: int = 256,
                 num_actions: int = 4,
                 dropout: float = 0.1):
        
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) - information set vectors
        Returns:
            values: (batch, num_actions) - counterfactual values for each action
        """
        return self.network(x)


class CFRStrategyNetwork(nn.Module):
    """
    Neural network to approximate average strategy.
    Outputs action probabilities for each information set.
    """
    
    def __init__(self,
                 input_dim: int = 72,
                 hidden_dim: int = 256,
                 num_actions: int = 4,
                 dropout: float = 0.1):
        
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) - information set vectors
        Returns:
            strategy: (batch, num_actions) - action probabilities
        """
        return self.network(x)


# =============================================================================
# Deep CFR Implementation
# =============================================================================

class DeepCFR:
    """
    Deep Counterfactual Regret Minimization.
    Uses neural networks to approximate counterfactual values and strategies.
    """
    
    def __init__(self,
                 num_players: int = 6,
                 num_actions: int = 4,
                 hidden_dim: int = 256,
                 buffer_size: int = 1000000,
                 batch_size: int = 256,
                 learning_rate: float = 1e-3,
                 device: str = 'cpu'):
        
        self.num_players = num_players
        self.num_actions = num_actions
        self.device = device
        self.batch_size = batch_size
        
        # Value networks (one per player)
        self.value_networks = [
            CFRValueNetwork(hidden_dim=hidden_dim, num_actions=num_actions).to(device)
            for _ in range(num_players)
        ]
        
        # Strategy network (shared)
        self.strategy_network = CFRStrategyNetwork(
            hidden_dim=hidden_dim, num_actions=num_actions
        ).to(device)
        
        # Optimizers
        self.value_optimizers = [
            torch.optim.Adam(net.parameters(), lr=learning_rate)
            for net in self.value_networks
        ]
        self.strategy_optimizer = torch.optim.Adam(
            self.strategy_network.parameters(), lr=learning_rate
        )
        
        # Advantage memory (regrets)
        self.advantage_memory = [[] for _ in range(num_players)]
        
        # Strategy memory
        self.strategy_memory = []
        
        # Iteration counter
        self.iteration = 0
    
    def get_regret_matched_strategy(self, 
                                     advantages: np.ndarray,
                                     valid_actions: Optional[List[bool]] = None) -> np.ndarray:
        """
        Compute strategy from advantages using regret matching.
        """
        if valid_actions is None:
            valid_actions = [True] * len(advantages)
        
        # Apply action masking
        masked_advantages = np.where(valid_actions, advantages, -np.inf)
        
        # Regret matching
        positive_regrets = np.maximum(masked_advantages, 0)
        regret_sum = positive_regrets.sum()
        
        if regret_sum > 0:
            strategy = positive_regrets / regret_sum
        else:
            # Uniform over valid actions
            strategy = np.array(valid_actions, dtype=np.float32)
            strategy = strategy / strategy.sum()
        
        return strategy
    
    def traverse(self,
                 state: GameState,
                 player: int,
                 reach_probs: np.ndarray,
                 traverser: int) -> float:
        """
        Traverse game tree and compute counterfactual values.
        Returns expected value for the traverser.
        """
        if state.is_terminal:
            # Terminal payoff
            rewards = {p.seat: p.stack - 100.0 for p in state.players}  # Profit/loss
            return rewards.get(traverser, 0.0)
        
        current_player = state.current_player
        
        if current_player != traverser:
            # Opponent's turn - sample action
            infoset = self._state_to_infoset(state, current_player)
            infoset_vec = torch.from_numpy(infoset.to_vector()).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                advantages = self.value_networks[current_player](infoset_vec).cpu().numpy()[0]
            
            strategy = self.get_regret_matched_strategy(advantages)
            
            # Sample action
            action_idx = np.random.choice(self.num_actions, p=strategy)
            action, amount = self._idx_to_action(action_idx, state, current_player)
            
            # Update reach probability
            new_reach = reach_probs.copy()
            new_reach[current_player] *= strategy[action_idx]
            
            # Continue traversal
            next_state, _, done = self._step_state(state.copy(), action, amount)
            return self.traverse(next_state, player, new_reach, traverser)
        
        else:
            # Traverser's turn - compute counterfactual values
            infoset = self._state_to_infoset(state, current_player)
            infoset_vec = infoset.to_vector()
            
            action_values = np.zeros(self.num_actions)
            
            for action_idx in range(self.num_actions):
                action, amount = self._idx_to_action(action_idx, state, current_player)
                
                # Check if action is valid
                if not self._is_valid_action(action, state, current_player):
                    action_values[action_idx] = -1e9
                    continue
                
                next_state, _, done = self._step_state(state.copy(), action, amount)
                action_values[action_idx] = self.traverse(
                    next_state, current_player, reach_probs.copy(), traverser
                )
            
            # Compute strategy and expected value
            with torch.no_grad():
                advantages = self.value_networks[current_player](
                    torch.from_numpy(infoset_vec).unsqueeze(0).to(self.device)
                ).cpu().numpy()[0]
            
            strategy = self.get_regret_matched_strategy(advantages)
            expected_value = (strategy * action_values).sum()
            
            # Compute counterfactual regrets
            cf_reach = reach_probs.copy()
            cf_reach[current_player] = 1.0
            cf_prob = cf_reach.prod()
            
            regrets = cf_prob * (action_values - expected_value)
            
            # Store in advantage memory
            self.advantage_memory[current_player].append({
                'infoset': infoset_vec,
                'regrets': regrets,
                'iteration': self.iteration
            })
            
            # Store in strategy memory
            self.strategy_memory.append({
                'infoset': infoset_vec,
                'strategy': strategy,
                'reach': reach_probs[current_player],
                'iteration': self.iteration
            })
            
            return expected_value
    
    def train_iteration(self, num_traversals: int = 1000):
        """Run one CFR iteration"""
        from poker_env import PokerEnvironment
        
        self.iteration += 1
        env = PokerEnvironment(num_players=self.num_players)
        
        # Traverse game trees
        for _ in range(num_traversals):
            state = env.reset()
            reach_probs = np.ones(self.num_players)
            
            # Traverse for each player
            for traverser in range(self.num_players):
                self.traverse(state.copy(), traverser, reach_probs.copy(), traverser)
        
        # Train value networks
        for player in range(self.num_players):
            self._train_value_network(player)
        
        # Train strategy network
        self._train_strategy_network()
        
        # Clear old memories (reservoir sampling)
        self._prune_memories()
    
    def _train_value_network(self, player: int):
        """Train value network for a player"""
        memory = self.advantage_memory[player]
        if len(memory) < self.batch_size:
            return
        
        # Sample batch
        batch = random.sample(memory, self.batch_size)
        
        infosets = torch.stack([
            torch.from_numpy(b['infoset']) for b in batch
        ]).to(self.device)
        
        regrets = torch.stack([
            torch.from_numpy(b['regrets']) for b in batch
        ]).to(self.device)
        
        # Train
        self.value_networks[player].train()
        self.value_optimizers[player].zero_grad()
        
        predictions = self.value_networks[player](infosets)
        loss = F.mse_loss(predictions, regrets)
        
        loss.backward()
        self.value_optimizers[player].step()
    
    def _train_strategy_network(self):
        """Train strategy network"""
        if len(self.strategy_memory) < self.batch_size:
            return
        
        batch = random.sample(self.strategy_memory, self.batch_size)
        
        infosets = torch.stack([
            torch.from_numpy(b['infoset']) for b in batch
        ]).to(self.device)
        
        strategies = torch.stack([
            torch.from_numpy(b['strategy']) for b in batch
        ]).to(self.device)
        
        reaches = torch.tensor([b['reach'] for b in batch], device=self.device)
        
        # Weighted cross-entropy loss
        self.strategy_network.train()
        self.strategy_optimizer.zero_grad()
        
        predictions = self.strategy_network(infosets)
        log_predictions = torch.log(predictions + 1e-8)
        loss = -(reaches.unsqueeze(1) * strategies * log_predictions).sum(dim=-1).mean()
        
        loss.backward()
        self.strategy_optimizer.step()
    
    def _prune_memories(self, max_size: int = 100000):
        """Prune memories to max size"""
        for player in range(self.num_players):
            if len(self.advantage_memory[player]) > max_size:
                # Keep more recent samples (higher iteration weight)
                self.advantage_memory[player] = random.sample(
                    self.advantage_memory[player], max_size
                )
        
        if len(self.strategy_memory) > max_size:
            self.strategy_memory = random.sample(self.strategy_memory, max_size)
    
    def _state_to_infoset(self, state: GameState, player: int) -> InfoSet:
        """Convert game state to information set"""
        p = state.players[player]
        
        hole_cards = tuple(c.to_int() for c in p.hole_cards)
        community = tuple(c.to_int() for c in state.community_cards)
        
        action_seq = tuple(
            (a[0], a[1].value) for a in state.action_history
        )
        
        position = (player - state.button_seat) % state.num_players
        
        return InfoSet(
            player=player,
            hole_cards=hole_cards,
            community_cards=community,
            pot=state.pot,
            to_call=state.get_to_call(player),
            stack=p.stack,
            action_sequence=action_seq,
            street=state.street,
            position=position
        )
    
    def _idx_to_action(self, idx: int, state: GameState, player: int) -> Tuple[Action, float]:
        """Convert action index to Action and amount"""
        p = state.players[player]
        to_call = state.get_to_call(player)
        
        if idx == 0:  # Fold/Check
            if to_call == 0:
                return Action.CALL, 0
            return Action.FOLD, 0
        elif idx == 1:  # Call
            return Action.CALL, min(to_call, p.stack)
        elif idx == 2:  # Small raise
            amount = to_call + state.pot * 0.5
            return Action.BET, min(amount, p.stack)
        else:  # Large raise / All-in
            amount = to_call + state.pot
            return Action.BET, min(amount, p.stack)
    
    def _is_valid_action(self, action: Action, state: GameState, player: int) -> bool:
        """Check if action is valid"""
        p = state.players[player]
        to_call = state.get_to_call(player)
        
        if action == Action.FOLD:
            return to_call > 0  # Can't fold when no bet to call
        
        return True
    
    def _step_state(self, state: GameState, action: Action, amount: float):
        """Apply action to state"""
        from poker_env import PokerEnvironment
        
        # Create a temporary environment to step
        env = PokerEnvironment(num_players=state.num_players)
        env.state = state
        return env.step(action, amount)
    
    def get_strategy(self, infoset: InfoSet) -> np.ndarray:
        """Get strategy for an information set"""
        self.strategy_network.eval()
        
        with torch.no_grad():
            infoset_vec = torch.from_numpy(infoset.to_vector()).unsqueeze(0).to(self.device)
            strategy = self.strategy_network(infoset_vec).cpu().numpy()[0]
        
        return strategy
    
    def save(self, path: str):
        """Save all models"""
        torch.save({
            'value_networks': [net.state_dict() for net in self.value_networks],
            'strategy_network': self.strategy_network.state_dict(),
            'iteration': self.iteration
        }, path)
    
    def load(self, path: str):
        """Load all models"""
        checkpoint = torch.load(path, map_location=self.device)
        for i, state_dict in enumerate(checkpoint['value_networks']):
            self.value_networks[i].load_state_dict(state_dict)
        self.strategy_network.load_state_dict(checkpoint['strategy_network'])
        self.iteration = checkpoint['iteration']


# =============================================================================
# Neural Fictitious Self-Play (NFSP)
# =============================================================================

class NFSPAgent:
    """
    Neural Fictitious Self-Play agent.
    Combines reinforcement learning (best response) with supervised learning (average strategy).
    """
    
    def __init__(self,
                 input_dim: int = 72,
                 num_actions: int = 4,
                 hidden_dim: int = 256,
                 rl_buffer_size: int = 200000,
                 sl_buffer_size: int = 2000000,
                 batch_size: int = 256,
                 anticipatory_param: float = 0.1,  # eta in NFSP paper
                 rl_learning_rate: float = 1e-3,
                 sl_learning_rate: float = 1e-3,
                 device: str = 'cpu'):
        
        self.device = device
        self.batch_size = batch_size
        self.eta = anticipatory_param
        
        # Best response network (DQN)
        self.q_network = CFRValueNetwork(
            input_dim=input_dim, hidden_dim=hidden_dim, num_actions=num_actions
        ).to(device)
        self.q_target = CFRValueNetwork(
            input_dim=input_dim, hidden_dim=hidden_dim, num_actions=num_actions
        ).to(device)
        self.q_target.load_state_dict(self.q_network.state_dict())
        
        # Average strategy network
        self.avg_network = CFRStrategyNetwork(
            input_dim=input_dim, hidden_dim=hidden_dim, num_actions=num_actions
        ).to(device)
        
        # Optimizers
        self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=rl_learning_rate)
        self.avg_optimizer = torch.optim.Adam(self.avg_network.parameters(), lr=sl_learning_rate)
        
        # Experience buffers
        self.rl_buffer = []  # (state, action, reward, next_state, done)
        self.sl_buffer = []  # (state, action)
        self.rl_buffer_size = rl_buffer_size
        self.sl_buffer_size = sl_buffer_size
        
        # Training stats
        self.num_updates = 0
    
    def get_action(self, infoset: InfoSet, use_average: bool = False) -> int:
        """
        Get action for an information set.
        use_average: If True, use average strategy. If False, use best response.
        """
        infoset_vec = torch.from_numpy(infoset.to_vector()).unsqueeze(0).to(self.device)
        
        if use_average:
            self.avg_network.eval()
            with torch.no_grad():
                probs = self.avg_network(infoset_vec).cpu().numpy()[0]
            return np.random.choice(len(probs), p=probs)
        else:
            # Epsilon-greedy for exploration
            epsilon = max(0.1, 1.0 - self.num_updates / 100000)
            
            if random.random() < epsilon:
                return random.randint(0, 3)
            
            self.q_network.eval()
            with torch.no_grad():
                q_values = self.q_network(infoset_vec).cpu().numpy()[0]
            return int(np.argmax(q_values))
    
    def store_rl_transition(self, state: np.ndarray, action: int, 
                            reward: float, next_state: np.ndarray, done: bool):
        """Store transition in RL buffer"""
        if len(self.rl_buffer) >= self.rl_buffer_size:
            self.rl_buffer.pop(0)
        self.rl_buffer.append((state, action, reward, next_state, done))
    
    def store_sl_transition(self, state: np.ndarray, action: int):
        """Store transition in supervised learning buffer"""
        if len(self.sl_buffer) >= self.sl_buffer_size:
            self.sl_buffer.pop(0)
        self.sl_buffer.append((state, action))
    
    def update(self, gamma: float = 0.99):
        """Update both networks"""
        self.num_updates += 1
        
        # Update Q-network (DQN)
        if len(self.rl_buffer) >= self.batch_size:
            self._update_q_network(gamma)
        
        # Update average strategy network
        if len(self.sl_buffer) >= self.batch_size:
            self._update_avg_network()
        
        # Update target network
        if self.num_updates % 1000 == 0:
            self.q_target.load_state_dict(self.q_network.state_dict())
    
    def _update_q_network(self, gamma: float):
        """DQN update"""
        batch = random.sample(self.rl_buffer, self.batch_size)
        
        states = torch.tensor([b[0] for b in batch], dtype=torch.float32, device=self.device)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor([b[3] for b in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=self.device)
        
        self.q_network.train()
        
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q = self.q_target(next_states).max(dim=1)[0]
            target_q = rewards + gamma * next_q * (1 - dones)
        
        loss = F.mse_loss(current_q, target_q)
        
        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()
    
    def _update_avg_network(self):
        """Supervised learning update for average strategy"""
        batch = random.sample(self.sl_buffer, self.batch_size)
        
        states = torch.tensor([b[0] for b in batch], dtype=torch.float32, device=self.device)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long, device=self.device)
        
        self.avg_network.train()
        
        probs = self.avg_network(states)
        log_probs = torch.log(probs + 1e-8)
        
        loss = F.nll_loss(log_probs, actions)
        
        self.avg_optimizer.zero_grad()
        loss.backward()
        self.avg_optimizer.step()
    
    def get_strategy(self, infoset: InfoSet) -> np.ndarray:
        """Get current average strategy"""
        self.avg_network.eval()
        
        with torch.no_grad():
            infoset_vec = torch.from_numpy(infoset.to_vector()).unsqueeze(0).to(self.device)
            probs = self.avg_network(infoset_vec).cpu().numpy()[0]
        
        return probs
    
    def save(self, path: str):
        """Save models"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'q_target': self.q_target.state_dict(),
            'avg_network': self.avg_network.state_dict(),
            'num_updates': self.num_updates
        }, path)
    
    def load(self, path: str):
        """Load models"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.q_target.load_state_dict(checkpoint['q_target'])
        self.avg_network.load_state_dict(checkpoint['avg_network'])
        self.num_updates = checkpoint['num_updates']


if __name__ == "__main__":
    print("Testing CFR Networks...")
    
    # Test value network
    value_net = CFRValueNetwork()
    strategy_net = CFRStrategyNetwork()
    
    batch = torch.randn(32, 72)
    
    values = value_net(batch)
    strategy = strategy_net(batch)
    
    print(f"Value network output shape: {values.shape}")  # (32, 4)
    print(f"Strategy network output shape: {strategy.shape}")  # (32, 4)
    print(f"Strategy sums to 1: {strategy.sum(dim=-1).mean():.4f}")
    
    # Test NFSP agent
    print("\nTesting NFSP Agent...")
    nfsp = NFSPAgent()
    
    # Create dummy infoset
    infoset = InfoSet(
        player=0,
        hole_cards=(0, 1),
        community_cards=(),
        pot=10.0,
        to_call=5.0,
        stack=95.0,
        action_sequence=(),
        street=Street.PREFLOP,
        position=0
    )
    
    action = nfsp.get_action(infoset, use_average=False)
    print(f"NFSP action (best response): {action}")
    
    action = nfsp.get_action(infoset, use_average=True)
    print(f"NFSP action (average): {action}")
