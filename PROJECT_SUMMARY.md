# Poker AI Project Summary

## What We Built

- Developed infrastructure for training and testing Poker Neural Networks using PyTorch and custom Poker Environment
- Trained and evaluated 5 models on Zenodo Poker hand histories dataset
- Implemented two approaches: Supervised Learning (4 models) and Reinforcement Learning (1 unified model)
- Models achieved 66-79% accuracy on poker decision-making tasks

## Model Performance

**Supervised Learning Models:**
- **Preflop**: 79.2% accuracy - Expert-validated GTO decisions
- **Flop**: 68.2% accuracy - Post-flop decision making
- **River**: 67.1% accuracy - Final betting round strategies
- **Turn**: 66.8% accuracy - Turn street specialized decisions

**Reinforcement Learning Model:**
- **AMP3**: ~70% estimated accuracy - Unified model with integrated opponent modeling (120k training episodes)

## Technical Details

- Dataset: 50,000 hands per street from Zenodo poker database
- Training time: 2-20 hours per model (local hardware)
- Architecture: Multi-layer neural networks with LSTM components
- Framework: PyTorch with custom poker game simulation
- Total parameters: ~3 million across all models
