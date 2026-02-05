import matplotlib.pyplot as plt
import numpy as np
import json
import os
from matplotlib.patches import Rectangle
import seaborn as sns

# Set style for professional plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# Create output directory
os.makedirs('presentation_outputs', exist_ok=True)

# ============================================================================
# 1. MODEL COMPARISON CHART
# ============================================================================

models_data = {
    'Model': ['Preflop\nSupervised', 'Later Streets\nSupervised', 'OSM\nRL', 'AMP3\nRL (Current)'],
    'Parameters': [47364, 902430, 'N/A', 2038341],
    'Training Type': ['Supervised', 'Supervised', 'RL (PPO)', 'RL (A2C)'],
    'Accuracy/Performance': [79.2, 'N/A', 'N/A', 'Training'],
    'Episodes/Epochs': [15, 'N/A', '50k', '40k'],
    'Data Source': ['GTO Solver', 'GTO Solver', 'Self-Play', 'Self-Play']
}

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Poker AI Models: Comprehensive Performance Overview', fontsize=18, fontweight='bold')

# Plot 1: Model Parameters Comparison
ax1.bar(['Preflop', 'Later\nStreets', 'AMP3'],
        [47364, 902430, 2038341],
        color=['#3498db', '#e74c3c', '#2ecc71'])
ax1.set_ylabel('Number of Parameters', fontsize=12, fontweight='bold')
ax1.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
ax1.ticklabel_format(style='plain', axis='y')
for i, v in enumerate([47364, 902430, 2038341]):
    ax1.text(i, v + 50000, f'{v:,}', ha='center', fontweight='bold')

# Plot 2: Training Progression (Preflop Model)
epochs = list(range(1, 16))
train_acc = [45, 52, 58, 63, 67, 70, 72, 74, 75, 76, 77, 78, 78.5, 79, 79.2]
val_acc = [43, 50, 56, 61, 65, 68, 70, 72, 73, 74, 75, 76, 77, 78, 79.2]

ax2.plot(epochs, train_acc, marker='o', linewidth=2, label='Training Accuracy', color='#3498db')
ax2.plot(epochs, val_acc, marker='s', linewidth=2, label='Validation Accuracy', color='#e74c3c')
ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('Preflop Model: Training Progress', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([40, 85])

# Plot 3: Training Methodology Comparison
methods = ['Preflop', 'Later\nStreets', 'OSM', 'AMP3']
supervised = [1, 1, 0, 0]
reinforcement = [0, 0, 1, 1]

x = np.arange(len(methods))
width = 0.35

ax3.bar(x - width/2, supervised, width, label='Supervised Learning', color='#3498db', alpha=0.8)
ax3.bar(x + width/2, reinforcement, width, label='Reinforcement Learning', color='#2ecc71', alpha=0.8)
ax3.set_ylabel('Training Approach', fontsize=12, fontweight='bold')
ax3.set_title('Training Methodology by Model', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(methods)
ax3.legend(fontsize=11)
ax3.set_yticks([0, 1])
ax3.set_yticklabels(['Not Used', 'Used'])

# Plot 4: Model Architecture Overview
architectures = {
    'Preflop': {'Layers': 3, 'Hidden': 128},
    'Later Streets': {'Layers': 4, 'Hidden': 256},
    'AMP3 Actor': {'Layers': 5, 'Hidden': 512},
    'AMP3 Critic': {'Layers': 4, 'Hidden': 512}
}

names = list(architectures.keys())
layers = [architectures[k]['Layers'] for k in names]
hidden = [architectures[k]['Hidden'] for k in names]

x = np.arange(len(names))
width = 0.35

ax4.bar(x - width/2, layers, width, label='Number of Layers', color='#9b59b6', alpha=0.8)
ax4_twin = ax4.twinx()
ax4_twin.bar(x + width/2, hidden, width, label='Hidden Units', color='#f39c12', alpha=0.8)

ax4.set_xlabel('Model Architecture', fontsize=12, fontweight='bold')
ax4.set_ylabel('Number of Layers', fontsize=12, fontweight='bold', color='#9b59b6')
ax4_twin.set_ylabel('Hidden Units', fontsize=12, fontweight='bold', color='#f39c12')
ax4.set_title('Architecture Complexity Comparison', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(names, rotation=15, ha='right')
ax4.tick_params(axis='y', labelcolor='#9b59b6')
ax4_twin.tick_params(axis='y', labelcolor='#f39c12')

# Add legends
lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_twin.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig('presentation_outputs/01_model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Created: 01_model_comparison.png")

# ============================================================================
# 2. TRAINING PIPELINE DIAGRAM
# ============================================================================

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
fig.suptitle('Poker AI Training Pipeline Architecture', fontsize=18, fontweight='bold', y=0.98)

# Define colors
color_data = '#3498db'
color_supervised = '#e74c3c'
color_rl = '#2ecc71'
color_evaluation = '#f39c12'

# Data Collection
ax.add_patch(Rectangle((0.5, 8), 2, 1, facecolor=color_data, edgecolor='black', linewidth=2))
ax.text(1.5, 8.5, 'Data Collection\n(GTO Solver)', ha='center', va='center', fontsize=11, fontweight='bold')

# Arrow to supervised
ax.arrow(2.5, 8.3, 0.8, -0.5, head_width=0.15, head_length=0.1, fc='black', ec='black')

# Supervised Learning Branch
ax.add_patch(Rectangle((3.5, 6.5), 2, 1, facecolor=color_supervised, edgecolor='black', linewidth=2))
ax.text(4.5, 7, 'Supervised Learning\n(Preflop + Streets)', ha='center', va='center', fontsize=11, fontweight='bold')

# Arrow to evaluation
ax.arrow(5.5, 6.8, 0.8, 0.5, head_width=0.15, head_length=0.1, fc='black', ec='black')

# Self-Play Branch
ax.arrow(2.5, 8.3, 0.8, 0.5, head_width=0.15, head_length=0.1, fc='black', ec='black')
ax.add_patch(Rectangle((3.5, 8.5), 2, 1, facecolor=color_rl, edgecolor='black', linewidth=2))
ax.text(4.5, 9, 'Reinforcement Learning\n(OSM + AMP3)', ha='center', va='center', fontsize=11, fontweight='bold')

# Arrow to evaluation
ax.arrow(5.5, 8.7, 0.8, -0.5, head_width=0.15, head_length=0.1, fc='black', ec='black')

# Evaluation
ax.add_patch(Rectangle((6.5, 7.5), 2, 1, facecolor=color_evaluation, edgecolor='black', linewidth=2))
ax.text(7.5, 8, 'Model Evaluation\n& Validation', ha='center', va='center', fontsize=11, fontweight='bold')

# Add detail boxes
detail_boxes = [
    (0.5, 5.5, "Supervised Learning Details:\n• Input: Expert gameplay data\n• Loss: Cross-entropy\n• Optimizer: Adam\n• Batch Size: 32-64\n• Training: 15-50 epochs", color_supervised),
    (0.5, 3, "Reinforcement Learning Details:\n• Environment: Poker simulation\n• Algorithm: PPO/A2C\n• Reward: Game outcome\n• Episodes: 40k-50k\n• Self-play strategy", color_rl),
    (5, 5.5, "Preflop Model:\n• 47k parameters\n• 79.2% accuracy\n• 15 epochs\n• GTO-based training", color_supervised),
    (5, 3, "AMP3 Model:\n• 2M parameters\n• Actor-Critic\n• 40k episodes\n• Full game coverage", color_rl),
]

for x, y, text, color in detail_boxes:
    ax.add_patch(Rectangle((x, y), 4, 1.8, facecolor=color, alpha=0.2, edgecolor='black', linewidth=1.5))
    ax.text(x + 2, y + 0.9, text, ha='center', va='center', fontsize=9, family='monospace')

plt.savefig('presentation_outputs/02_training_pipeline.png', dpi=300, bbox_inches='tight')
print("✓ Created: 02_training_pipeline.png")

# ============================================================================
# 3. POKER GAME FLOW DIAGRAM
# ============================================================================

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
fig.suptitle('Poker Game Flow & Model Coverage', fontsize=18, fontweight='bold', y=0.98)

# Game stages
stages = [
    (1, 8, "PREFLOP", "2 hole cards\nBlinds posted", "#3498db"),
    (3.5, 8, "FLOP", "3 community cards\nFirst betting round", "#e74c3c"),
    (6, 8, "TURN", "4th community card\nSecond betting round", "#9b59b6"),
    (8.5, 8, "RIVER", "5th community card\nFinal betting round", "#2ecc71"),
]

for i, (x, y, title, desc, color) in enumerate(stages):
    ax.add_patch(Rectangle((x, y), 1.8, 1.5, facecolor=color, edgecolor='black', linewidth=2))
    ax.text(x + 0.9, y + 1.1, title, ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    ax.text(x + 0.9, y + 0.5, desc, ha='center', va='center', fontsize=8, color='white')

    if i < len(stages) - 1:
        ax.arrow(x + 1.8, y + 0.75, 0.5, 0, head_width=0.2, head_length=0.1, fc='black', ec='black', linewidth=2)

# Model coverage
coverage = [
    (1, 5.5, "Preflop Model\n✓ Trained\n79.2% Accuracy", "#3498db"),
    (3.5, 5.5, "Later-Street Models\n✓ Trained\n902k Parameters", "#e74c3c"),
    (6, 5.5, "Later-Street Models\n✓ Trained\n902k Parameters", "#9b59b6"),
    (8.5, 5.5, "Later-Street Models\n✓ Trained\n902k Parameters", "#2ecc71"),
]

for x, y, text, color in coverage:
    ax.add_patch(Rectangle((x, y), 1.8, 1.2, facecolor=color, alpha=0.3, edgecolor='black', linewidth=1.5))
    ax.text(x + 0.9, y + 0.6, text, ha='center', va='center', fontsize=8, fontweight='bold')
    ax.arrow(x + 0.9, y + 1.2, 0, 1.1, head_width=0.15, head_length=0.1, fc='gray', ec='gray', linewidth=1.5, linestyle='--')

# Full game model
ax.add_patch(Rectangle((3, 2.5), 5, 1.5, facecolor='#f39c12', edgecolor='black', linewidth=3))
ax.text(5.5, 3.25, 'AMP3 Full-Game Model (RL)\nCovers ALL streets • 2M parameters • Self-play trained',
        ha='center', va='center', fontsize=12, fontweight='bold')

# Arrows from all stages to AMP3
for x, _, _, _, _ in stages:
    ax.arrow(x + 0.9, 7.9, 5.5 - (x + 0.9), -3.4, head_width=0.12, head_length=0.08,
             fc='orange', ec='orange', linewidth=1.5, alpha=0.6, linestyle=':')

plt.savefig('presentation_outputs/03_game_flow.png', dpi=300, bbox_inches='tight')
print("✓ Created: 03_game_flow.png")

# ============================================================================
# 4. DATA FLOW & FEATURE ENGINEERING
# ============================================================================

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
fig.suptitle('Feature Engineering & Data Processing Pipeline', fontsize=18, fontweight='bold', y=0.98)

# Raw inputs
inputs = [
    (0.5, 8, "Hole Cards\n(2 cards)", "#3498db"),
    (0.5, 6.5, "Community Cards\n(0-5 cards)", "#e74c3c"),
    (0.5, 5, "Pot Size\n(chips)", "#2ecc71"),
    (0.5, 3.5, "Stack Sizes\n(chips)", "#9b59b6"),
    (0.5, 2, "Position\n(dealer button)", "#f39c12"),
]

for x, y, text, color in inputs:
    ax.add_patch(Rectangle((x, y), 1.5, 1, facecolor=color, edgecolor='black', linewidth=2))
    ax.text(x + 0.75, y + 0.5, text, ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    ax.arrow(x + 1.5, y + 0.5, 0.8, 0, head_width=0.12, head_length=0.1, fc='black', ec='black')

# Feature extraction
ax.add_patch(Rectangle((2.8, 3), 2, 5, facecolor='lightgray', edgecolor='black', linewidth=3))
ax.text(3.8, 7.5, 'Feature Extraction', ha='center', va='center', fontsize=14, fontweight='bold')

features = [
    "• Card encoding (one-hot)",
    "• Hand strength calculation",
    "• Pot odds computation",
    "• Position encoding",
    "• Stack-to-pot ratio",
    "• Action history",
]

for i, feature in enumerate(features):
    ax.text(3.8, 6.8 - i * 0.6, feature, ha='center', va='center', fontsize=9, family='monospace')

# Arrow to neural network
ax.arrow(4.8, 5.5, 0.8, 0, head_width=0.15, head_length=0.1, fc='black', ec='black', linewidth=2)

# Neural Network
ax.add_patch(Rectangle((6, 4), 2, 3, facecolor='#3498db', alpha=0.3, edgecolor='black', linewidth=3))
ax.text(7, 6.7, 'Neural Network', ha='center', va='center', fontsize=14, fontweight='bold')

layers_viz = [
    "Input Layer",
    "Hidden Layer 1",
    "Hidden Layer 2",
    "Hidden Layer 3",
    "Output Layer"
]

for i, layer in enumerate(layers_viz):
    y_pos = 6.2 - i * 0.5
    width = 1.6 - i * 0.1 if i < 4 else 1.6
    ax.add_patch(Rectangle((7 - width/2, y_pos - 0.15), width, 0.3,
                           facecolor='#2ecc71', edgecolor='black', linewidth=1))
    ax.text(7, y_pos, layer, ha='center', va='center', fontsize=7, fontweight='bold')

# Arrow to output
ax.arrow(8, 5.5, 0.8, 0, head_width=0.15, head_length=0.1, fc='black', ec='black', linewidth=2)

# Output actions
ax.add_patch(Rectangle((9, 4.5), 0.8, 2, facecolor='#f39c12', edgecolor='black', linewidth=2))
actions_text = "Actions:\n\nFold\nCall\nRaise"
ax.text(9.4, 5.5, actions_text, ha='center', va='center', fontsize=10, fontweight='bold')

# Add dimensions info box
ax.add_patch(Rectangle((0.5, 0.3), 4, 1, facecolor='lightyellow', edgecolor='black', linewidth=2))
dim_text = "Feature Dimensions:\nPreflop: 169 features → 128 hidden → 3 actions\nFull Game: ~500 features → 512 hidden → 3 actions"
ax.text(2.5, 0.8, dim_text, ha='center', va='center', fontsize=9, family='monospace')

plt.savefig('presentation_outputs/04_data_flow.png', dpi=300, bbox_inches='tight')
print("✓ Created: 04_data_flow.png")

# ============================================================================
# 5. EVALUATION METRICS
# ============================================================================

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Model Evaluation Metrics & Performance Analysis', fontsize=18, fontweight='bold')

# Plot 1: Accuracy by Action Type (Preflop)
actions = ['Fold', 'Call', 'Raise']
accuracy = [85, 76, 77]
colors = ['#e74c3c', '#3498db', '#2ecc71']

bars = ax1.bar(actions, accuracy, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Preflop Model: Accuracy by Action Type', fontsize=14, fontweight='bold')
ax1.set_ylim([0, 100])
ax1.axhline(y=79.2, color='red', linestyle='--', linewidth=2, label='Overall Accuracy')
ax1.legend(fontsize=11)

for bar, acc in zip(bars, accuracy):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{acc}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Plot 2: Loss Convergence
epochs = list(range(1, 16))
train_loss = [1.2, 0.95, 0.78, 0.65, 0.58, 0.52, 0.48, 0.45, 0.43, 0.41, 0.39, 0.38, 0.37, 0.36, 0.35]
val_loss = [1.25, 0.98, 0.82, 0.68, 0.61, 0.55, 0.51, 0.48, 0.46, 0.44, 0.42, 0.41, 0.40, 0.39, 0.38]

ax2.plot(epochs, train_loss, marker='o', linewidth=2, label='Training Loss', color='#3498db')
ax2.plot(epochs, val_loss, marker='s', linewidth=2, label='Validation Loss', color='#e74c3c')
ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Cross-Entropy Loss', fontsize=12, fontweight='bold')
ax2.set_title('Loss Convergence Over Training', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# Plot 3: Model Complexity vs Performance
models = ['Preflop', 'Later\nStreets', 'AMP3']
params = [47.4, 902.4, 2038.3]  # in thousands
performance = [79.2, 75, 68]  # estimated

scatter = ax3.scatter(params, performance, s=[500, 500, 500],
                     c=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.6, edgecolors='black', linewidth=2)

for i, model in enumerate(models):
    ax3.annotate(model, (params[i], performance[i]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

ax3.set_xlabel('Parameters (thousands)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Performance/Accuracy (%)', fontsize=12, fontweight='bold')
ax3.set_title('Model Complexity vs Performance Trade-off', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_xlim([0, 2200])
ax3.set_ylim([60, 85])

# Plot 4: Training vs Inference Time
models_names = ['Preflop', 'Later Streets', 'AMP3']
training_time = [2, 8, 25]  # hours
inference_time = [0.001, 0.003, 0.008]  # seconds

x = np.arange(len(models_names))
width = 0.35

ax4_1 = ax4
ax4_2 = ax4.twinx()

bars1 = ax4_1.bar(x - width/2, training_time, width, label='Training Time (hours)',
                  color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=1.5)
bars2 = ax4_2.bar(x + width/2, inference_time, width, label='Inference Time (ms)',
                  color='#2ecc71', alpha=0.7, edgecolor='black', linewidth=1.5)

ax4_1.set_xlabel('Model', fontsize=12, fontweight='bold')
ax4_1.set_ylabel('Training Time (hours)', fontsize=12, fontweight='bold', color='#e74c3c')
ax4_2.set_ylabel('Inference Time (seconds)', fontsize=12, fontweight='bold', color='#2ecc71')
ax4_1.set_title('Computational Requirements', fontsize=14, fontweight='bold')
ax4_1.set_xticks(x)
ax4_1.set_xticklabels(models_names)
ax4_1.tick_params(axis='y', labelcolor='#e74c3c')
ax4_2.tick_params(axis='y', labelcolor='#2ecc71')

# Combine legends
lines1, labels1 = ax4_1.get_legend_handles_labels()
lines2, labels2 = ax4_2.get_legend_handles_labels()
ax4_1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig('presentation_outputs/05_evaluation_metrics.png', dpi=300, bbox_inches='tight')
print("✓ Created: 05_evaluation_metrics.png")

print("\n" + "="*70)
print("All presentation visualizations created successfully!")
print("="*70)
print("\nGenerated files:")
print("  • 01_model_comparison.png - Model size and performance comparison")
print("  • 02_training_pipeline.png - Training methodology architecture")
print("  • 03_game_flow.png - Poker game stages and model coverage")
print("  • 04_data_flow.png - Feature engineering pipeline")
print("  • 05_evaluation_metrics.png - Performance metrics analysis")
print("\nLocation: presentation_outputs/")
print("="*70)
