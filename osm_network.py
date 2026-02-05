"""
osm_network.py - Opponent Style Modeling Network
Deep learning network to predict opponent style features from game history.
Uses LSTM for action sequence encoding as described in the paper.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import json
from pathlib import Path

from poker_core import Card, Action, Street


# =============================================================================
# Data Encoding
# =============================================================================

@dataclass
class GameHistoryEncoding:
    """Encoded representation of a single game for OSM"""
    
    # Public cards encoding (5 cards * 2 features = 10)
    # Each card: [rank/14, suit/4] or [0, 0] if not dealt
    public_cards: np.ndarray  # shape (10,)
    
    # Hole cards encoding (2 cards * 2 features = 4)  
    hole_cards: np.ndarray    # shape (4,)
    
    # Action history (variable length sequence)
    # Each action: [is_target_player, action_type/3]
    action_history: np.ndarray  # shape (max_seq_len, 2)
    
    # Target style features
    style_features: np.ndarray  # shape (4,) - vpip, pfr, afq, wtsd


def encode_card(card: Optional[Card]) -> np.ndarray:
    """Encode a card as [rank_normalized, suit_normalized]"""
    if card is None:
        return np.array([0.0, 0.0], dtype=np.float32)
    return np.array([card.rank / 14.0, card.suit / 3.0], dtype=np.float32)


def encode_cards(cards: List[Card], max_cards: int) -> np.ndarray:
    """Encode a list of cards, padding to max_cards"""
    encoded = []
    for i in range(max_cards):
        if i < len(cards):
            encoded.extend(encode_card(cards[i]))
        else:
            encoded.extend([0.0, 0.0])
    return np.array(encoded, dtype=np.float32)


def encode_action(player_idx: int, action: Action, target_player: int) -> np.ndarray:
    """Encode a single action"""
    is_target = 1.0 if player_idx == target_player else 0.0
    action_normalized = action.value / 3.0  # Normalize to [0, 1]
    return np.array([is_target, action_normalized], dtype=np.float32)


def encode_game_history(
    public_cards: List[Card],
    hole_cards: List[Card],
    action_history: List[Tuple[int, Action, float]],
    target_player: int,
    style_features: np.ndarray,
    max_actions: int = 48
) -> GameHistoryEncoding:
    """
    Encode a complete game history for a target player.
    """
    # Encode public cards (5 cards max)
    public_enc = encode_cards(public_cards, 5)  # shape (10,)
    
    # Encode hole cards (2 cards)
    hole_enc = encode_cards(hole_cards, 2)  # shape (4,)
    
    # Encode action history
    action_enc = np.zeros((max_actions, 2), dtype=np.float32)
    for i, (player_idx, action, _) in enumerate(action_history[:max_actions]):
        action_enc[i] = encode_action(player_idx, action, target_player)
    
    return GameHistoryEncoding(
        public_cards=public_enc,
        hole_cards=hole_enc,
        action_history=action_enc,
        style_features=style_features
    )


# =============================================================================
# Dataset
# =============================================================================

class StyleFeatureDataset(Dataset):
    """
    Dataset of game histories with style feature labels.
    Each sample contains history info for predicting a player's style.
    """
    
    def __init__(self, 
                 data_path: Optional[str] = None,
                 data_list: Optional[List[GameHistoryEncoding]] = None):
        
        if data_list is not None:
            self.data = data_list
        elif data_path is not None:
            self.data = self._load_data(data_path)
        else:
            self.data = []
    
    def _load_data(self, path: str) -> List[GameHistoryEncoding]:
        """Load data from file"""
        data = np.load(path, allow_pickle=True)
        return [GameHistoryEncoding(**d) for d in data]
    
    def save(self, path: str):
        """Save data to file"""
        data_dicts = [
            {
                'public_cards': d.public_cards,
                'hole_cards': d.hole_cards,
                'action_history': d.action_history,
                'style_features': d.style_features
            }
            for d in self.data
        ]
        np.save(path, data_dicts, allow_pickle=True)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        d = self.data[idx]
        return (
            torch.from_numpy(d.public_cards),
            torch.from_numpy(d.hole_cards),
            torch.from_numpy(d.action_history),
            torch.from_numpy(d.style_features)
        )
    
    def add_sample(self, encoding: GameHistoryEncoding):
        """Add a single sample"""
        self.data.append(encoding)


# =============================================================================
# OSM Network
# =============================================================================

class StyleModelingNetwork(nn.Module):
    """
    Opponent Style Modeling Network.
    
    Architecture:
    - FC layers for public cards encoding
    - FC layers for hole cards encoding  
    - LSTM for action sequence encoding
    - Combined FC layers for style feature prediction
    
    Outputs: [vpip, pfr, afq, wtsd] predictions (4 values in [0,1])
    """
    
    def __init__(self,
                 public_dim: int = 10,      # 5 cards * 2 features
                 hole_dim: int = 4,          # 2 cards * 2 features
                 action_dim: int = 2,        # [is_target, action_type]
                 hidden_dim: int = 64,
                 lstm_hidden: int = 128,
                 lstm_layers: int = 3,
                 num_style_features: int = 4,
                 dropout: float = 0.1):
        
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.lstm_hidden = lstm_hidden
        
        # Public cards encoder
        self.public_encoder = nn.Sequential(
            nn.Linear(public_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Hole cards encoder
        self.hole_encoder = nn.Sequential(
            nn.Linear(hole_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Action sequence LSTM
        self.action_lstm = nn.LSTM(
            input_size=action_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Combined layers
        combined_dim = hidden_dim // 2 + hidden_dim // 2 + lstm_hidden
        self.combined = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_style_features),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, 
                public_cards: torch.Tensor,
                hole_cards: torch.Tensor,
                action_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            public_cards: (batch, 10)
            hole_cards: (batch, 4)
            action_history: (batch, seq_len, 2)
        
        Returns:
            style_features: (batch, 4) - predicted [vpip, pfr, afq, wtsd]
        """
        # Encode cards
        public_enc = self.public_encoder(public_cards)    # (batch, hidden/2)
        hole_enc = self.hole_encoder(hole_cards)          # (batch, hidden/2)
        
        # Encode action sequence with LSTM
        lstm_out, (h_n, _) = self.action_lstm(action_history)
        # Use last hidden state
        action_enc = h_n[-1]  # (batch, lstm_hidden)
        
        # Combine all features
        combined = torch.cat([public_enc, hole_enc, action_enc], dim=-1)
        
        # Predict style features
        style_features = self.combined(combined)
        
        return style_features


# =============================================================================
# Training
# =============================================================================

class OSMTrainer:
    """Trainer for the Opponent Style Modeling network"""
    
    def __init__(self,
                 model: StyleModelingNetwork,
                 learning_rate: float = 1e-3,
                 device: str = 'cpu'):
        
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for public, hole, actions, targets in dataloader:
            public = public.to(self.device)
            hole = hole.to(self.device)
            actions = actions.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(public, hole, actions)
            loss = self.loss_fn(predictions, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(1, num_batches)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> float:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for public, hole, actions, targets in dataloader:
            public = public.to(self.device)
            hole = hole.to(self.device)
            actions = actions.to(self.device)
            targets = targets.to(self.device)
            
            predictions = self.model(public, hole, actions)
            loss = self.loss_fn(predictions, targets)
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(1, num_batches)
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def train(self,
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              epochs: int = 100,
              verbose: bool = True) -> Dict:
        """Full training loop"""
        
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            
            val_loss = None
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict().copy()
            
            if verbose and epoch % 10 == 0:
                msg = f"Epoch {epoch:4d} | Train Loss: {train_loss:.4f}"
                if val_loss is not None:
                    msg += f" | Val Loss: {val_loss:.4f}"
                print(msg)
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss
        }
    
    def save(self, path: str):
        """Save model and training state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, path)
    
    def load(self, path: str):
        """Load model and training state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']


# =============================================================================
# Data Generation
# =============================================================================

def generate_osm_dataset(
    num_games: int = 100000,
    style_library = None,
    seed: int = 42,
    verbose: bool = True
) -> StyleFeatureDataset:
    """
    Generate a dataset of game histories with style labels.
    Simulates games between players from the style library.
    """
    from poker_env import PokerEnvironment
    from style_library import StyleLibrary
    
    if style_library is None:
        style_library = StyleLibrary(seed=seed)
    
    # First compute ground truth style features
    if not style_library.style_features:
        style_library.compute_style_features(num_games=10000, verbose=verbose)
    
    env = PokerEnvironment(num_players=6)
    rng = np.random.default_rng(seed)
    strategy_names = style_library.get_all_names()
    
    dataset = StyleFeatureDataset()
    
    if verbose:
        print(f"Generating OSM dataset with {num_games} games...")
    
    for game_idx in range(num_games):
        # Randomly select strategies for 6 players
        selected_names = [rng.choice(strategy_names) for _ in range(6)]
        selected_strategies = [style_library.get_strategy(n) for n in selected_names]
        
        state = env.reset(button_seat=game_idx % 6)
        
        # Store game info
        action_history = []
        player_hole_cards = {i: state.players[i].hole_cards.copy() for i in range(6)}
        
        while not state.is_terminal:
            seat = state.current_player
            strategy = selected_strategies[seat]
            player = state.players[seat]
            
            action, amount = strategy.get_action(
                state, seat, player.hole_cards, state.community_cards
            )
            
            action_history.append((seat, action, amount))
            state, _, done = env.step(action, amount)
        
        # Create encoded samples for each player
        for player_idx in range(6):
            style_name = selected_names[player_idx]
            style_features = style_library.style_features[style_name].to_vector()
            
            encoding = encode_game_history(
                public_cards=state.community_cards,
                hole_cards=player_hole_cards[player_idx],
                action_history=action_history,
                target_player=player_idx,
                style_features=style_features
            )
            dataset.add_sample(encoding)
        
        if verbose and (game_idx + 1) % 10000 == 0:
            print(f"  Generated {game_idx + 1}/{num_games} games ({len(dataset)} samples)")
    
    if verbose:
        print(f"Dataset complete: {len(dataset)} samples")
    
    return dataset


# =============================================================================
# Prediction
# =============================================================================

class StylePredictor:
    """
    Predictor for opponent style features using trained OSM model.
    Can aggregate predictions over multiple game histories.
    """
    
    def __init__(self, model: StyleModelingNetwork, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def predict_single(self,
                       public_cards: List[Card],
                       hole_cards: List[Card],
                       action_history: List[Tuple[int, Action, float]],
                       target_player: int) -> np.ndarray:
        """Predict style features from a single game"""
        
        # Encode
        public_enc = encode_cards(public_cards, 5)
        hole_enc = encode_cards(hole_cards, 2)
        
        action_enc = np.zeros((48, 2), dtype=np.float32)
        for i, (player_idx, action, _) in enumerate(action_history[:48]):
            action_enc[i] = encode_action(player_idx, action, target_player)
        
        # Convert to tensors
        public_t = torch.from_numpy(public_enc).unsqueeze(0).to(self.device)
        hole_t = torch.from_numpy(hole_enc).unsqueeze(0).to(self.device)
        action_t = torch.from_numpy(action_enc).unsqueeze(0).to(self.device)
        
        # Predict
        prediction = self.model(public_t, hole_t, action_t)
        return prediction.cpu().numpy()[0]
    
    def predict_aggregate(self,
                          game_histories: List[Dict],
                          target_player: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aggregate predictions over multiple games.
        Returns mean and std of predictions.
        """
        predictions = []
        
        for game in game_histories:
            pred = self.predict_single(
                public_cards=game['public_cards'],
                hole_cards=game['hole_cards'],
                action_history=game['action_history'],
                target_player=target_player
            )
            predictions.append(pred)
        
        predictions = np.array(predictions)
        return predictions.mean(axis=0), predictions.std(axis=0)


if __name__ == "__main__":
    # Test the OSM network
    print("Testing OSM Network...")
    
    # Create model
    model = StyleModelingNetwork(
        hidden_dim=64,
        lstm_hidden=128,
        lstm_layers=3
    )
    
    # Create dummy data
    batch_size = 32
    public = torch.randn(batch_size, 10)
    hole = torch.randn(batch_size, 4)
    actions = torch.randn(batch_size, 48, 2)
    
    # Forward pass
    output = model(public, hole, actions)
    print(f"Output shape: {output.shape}")  # Should be (32, 4)
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")  # Should be in [0, 1]
    
    # Test dataset generation (small)
    print("\nTesting dataset generation (100 games)...")
    dataset = generate_osm_dataset(num_games=100, seed=42, verbose=True)
    print(f"Dataset size: {len(dataset)}")
    
    # Test data loading
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    for batch in loader:
        public, hole, actions, targets = batch
        print(f"Batch shapes: public={public.shape}, hole={hole.shape}, "
              f"actions={actions.shape}, targets={targets.shape}")
        break
