"""
AMP3 Training with Real Poker Data
Trains the AMP3 model using your preprocessed poker dataset.

Expected data format:
- CSV: preflop_demo_full.csv with state, style, and action columns
- OR preprocessed .npy files: X_train.npy, y_train.npy, X_val.npy, y_val.npy

Features:
- 7 state features: position, to_call, pot, stack, spr, num_active, num_raises
- 6 style features: hero_vpip, hero_pfr, hero_aggr, mean_vpip, mean_pfr, mean_aggr
- 4 actions: FOLD (0), CALL (1), RAISE_SMALL (2), RAISE_LARGE (3)
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pathlib import Path


# =============================================================================
# AMP3 Policy Network
# =============================================================================

class AMP3Policy(nn.Module):
    """
    AMP3-style policy network that takes both state and style features.

    Architecture:
    - State encoder: processes game state features
    - Style encoder: processes opponent style features
    - Combined network: merges encoded features and outputs action logits
    """

    def __init__(self, state_dim=7, style_dim=6, hidden_dim=128, num_actions=4):
        super().__init__()

        self.state_dim = state_dim
        self.style_dim = style_dim

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Style encoder
        self.style_encoder = nn.Sequential(
            nn.Linear(style_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
        )

        # Combined policy head
        combined_dim = hidden_dim + hidden_dim // 2
        self.policy_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, state, style):
        """
        Args:
            state: (batch, state_dim) - game state features
            style: (batch, style_dim) - opponent style features

        Returns:
            logits: (batch, num_actions) - action logits
        """
        state_enc = self.state_encoder(state)
        style_enc = self.style_encoder(style)

        # Concatenate encodings
        combined = torch.cat([state_enc, style_enc], dim=-1)

        # Output action logits
        logits = self.policy_head(combined)
        return logits


# =============================================================================
# Dataset
# =============================================================================

class PokerDataset(Dataset):
    """Dataset for poker training data with state and style features."""

    def __init__(self, X, y, state_dim=7):
        """
        Args:
            X: (N, state_dim + style_dim) feature matrix
            y: (N,) action labels
            state_dim: number of state features (rest are style features)
        """
        self.X_state = torch.FloatTensor(X[:, :state_dim])
        self.X_style = torch.FloatTensor(X[:, state_dim:])
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_state[idx], self.X_style[idx], self.y[idx]


# =============================================================================
# Data Loading
# =============================================================================

def load_data_from_parquet(parquet_path):
    """Load and preprocess data from parquet file."""
    print(f"Loading data from {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} rows")
    return df


def process_dataframe(df):
    """Extract features and labels from dataframe."""
    # Define feature columns
    state_cols = [
        "hero_position_index",
        "to_call_bb",
        "pot_preflop_bb",
        "hero_stack_bb",
        "spr_preflop",
        "num_active_before",
        "num_raises_preflop_before",
    ]

    style_cols = [
        "hero_vpip",
        "hero_pfr",
        "hero_aggr",
        "mean_vpip_others",
        "mean_pfr_others",
        "mean_aggr_others",
    ]

    # Drop rows with missing data
    df = df.dropna(subset=state_cols + style_cols + ["hero_action_4way"])
    print(f"Rows after dropping NaNs: {len(df)}")

    # Build feature matrices
    X_state = df[state_cols].to_numpy(dtype=np.float32)
    X_style = df[style_cols].to_numpy(dtype=np.float32)
    X = np.concatenate([X_state, X_style], axis=1)

    # Map labels
    label_map = {
        "FOLD": 0,
        "CALL": 1,
        "RAISE_SMALL": 2,
        "RAISE_LARGE": 3,
    }
    y = df["hero_action_4way"].map(label_map).to_numpy(dtype=np.int64)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Label distribution: {np.bincount(y)}")

    return X, y


def load_data_from_csv(csv_path):
    """Load and preprocess data from CSV file."""
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")

    # Define feature columns
    state_cols = [
        "hero_position_index",
        "to_call_bb",
        "pot_preflop_bb",
        "hero_stack_bb",
        "spr_preflop",
        "num_active_before",
        "num_raises_preflop_before",
    ]

    style_cols = [
        "hero_vpip",
        "hero_pfr",
        "hero_aggr",
        "mean_vpip_others",
        "mean_pfr_others",
        "mean_aggr_others",
    ]

    # Drop rows with missing data
    df = df.dropna(subset=state_cols + style_cols + ["hero_action_4way"])
    print(f"Rows after dropping NaNs: {len(df)}")

    # Build feature matrices
    X_state = df[state_cols].to_numpy(dtype=np.float32)
    X_style = df[style_cols].to_numpy(dtype=np.float32)
    X = np.concatenate([X_state, X_style], axis=1)

    # Map labels
    label_map = {
        "FOLD": 0,
        "CALL": 1,
        "RAISE_SMALL": 2,
        "RAISE_LARGE": 3,
    }
    y = df["hero_action_4way"].map(label_map).to_numpy(dtype=np.int64)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Label distribution: {np.bincount(y)}")

    return X, y


def load_data_from_npy(data_dir):
    """Load preprocessed .npy files."""
    print(f"Loading preprocessed data from {data_dir}")

    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    return X_train, y_train, X_val, y_val


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for state, style, labels in train_loader:
        state = state.to(device)
        style = style.to(device)
        labels = labels.to(device)

        # Forward
        optimizer.zero_grad()
        logits = model(state, style)
        loss = criterion(logits, labels)

        # Backward
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        pred = logits.argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(train_loader), correct / total


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for state, style, labels in val_loader:
            state = state.to(device)
            style = style.to(device)
            labels = labels.to(device)

            logits = model(state, style)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            pred = logits.argmax(dim=-1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(val_loader), correct / total


def evaluate_model(model, val_loader, device):
    """Generate detailed evaluation metrics."""
    model.eval()
    all_true = []
    all_pred = []

    with torch.no_grad():
        for state, style, labels in val_loader:
            state = state.to(device)
            style = style.to(device)

            logits = model(state, style)
            pred = logits.argmax(dim=-1)

            all_true.append(labels.numpy())
            all_pred.append(pred.cpu().numpy())

    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)

    # Metrics
    action_names = ["FOLD", "CALL", "RAISE_SMALL", "RAISE_LARGE"]

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    acc = accuracy_score(all_true, all_pred)
    print(f"\nValidation Accuracy: {acc:.3f}")

    print("\nClassification Report:")
    print(classification_report(all_true, all_pred, target_names=action_names, digits=3))

    print("\nConfusion Matrix (rows=true, cols=pred):")
    cm = confusion_matrix(all_true, all_pred)
    print(cm)
    print()

    return acc


# =============================================================================
# Main Training Loop
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train AMP3 with real poker data')
    parser.add_argument('--data_dir', type=str,
                       default='/Users/ardaenfiyeci/Downloads',
                       help='Directory containing data files')
    parser.add_argument('--csv_file', type=str, default='preflop_demo_full.csv',
                       help='CSV filename')
    parser.add_argument('--use_csv', action='store_true', default=True,
                       help='Load from CSV file')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory to save model checkpoints')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden dimension size')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu or cuda)')

    args = parser.parse_args()

    # Setup
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"\nUsing device: {device}")

    # Load data from CSV
    csv_path = os.path.join(args.data_dir, args.csv_file)
    X, y = load_data_from_csv(csv_path)

    # Split into train/val
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    # Create datasets
    train_dataset = PokerDataset(X_train, y_train, state_dim=7)
    val_dataset = PokerDataset(X_val, y_val, state_dim=7)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create model
    model = AMP3Policy(
        state_dim=7,
        style_dim=6,
        hidden_dim=args.hidden_dim,
        num_actions=4
    ).to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # Training loop
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)

    best_val_acc = 0
    patience = 10
    patience_counter = 0

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(args.save_dir, 'best_model.pt'))

            print(f"  → Saved best model (val_acc={val_acc:.3f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Load best model and evaluate
    print(f"\nLoading best model (val_acc={best_val_acc:.3f})")
    checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])

    # Final evaluation
    final_acc = evaluate_model(model, val_loader, device)

    print(f"\n✅ Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.3f}")
    print(f"Model saved to: {args.save_dir}/best_model.pt")


if __name__ == '__main__':
    main()
