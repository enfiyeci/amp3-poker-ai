#!/usr/bin/env python3
"""
Create a clean performance comparison graph for all trained models
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import numpy as np

# Set style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.family'] = 'sans-serif'

# Model data with performance metrics
models_data = {
    'Preflop': {
        'accuracy': 79.2,
        'confidence': 79.2,
        'diversity': 100.0,  # Multiple actions
        'quality': 89.6,  # (accuracy + diversity/2)
        'training_hours': 2.0,
        'color': '#2E86AB',
        'params_k': 47.4,
        'size_mb': 0.569
    },
    'Flop': {
        'accuracy': None,
        'confidence': 61.0,
        'diversity': 84.0,
        'quality': 51.2,  # From test
        'training_hours': 8.0,
        'color': '#A23B72',
        'params_k': 300.8,
        'size_mb': 1.17
    },
    'Turn': {
        'accuracy': None,
        'confidence': 63.0,
        'diversity': 98.0,
        'quality': 59.4,  # From test
        'training_hours': 8.0,
        'color': '#F18F01',
        'params_k': 300.8,
        'size_mb': 1.17
    },
    'River': {
        'accuracy': None,
        'confidence': 62.0,
        'diversity': 95.0,
        'quality': 58.1,  # From test
        'training_hours': 8.0,
        'color': '#C73E1D',
        'params_k': 300.8,
        'size_mb': 1.17
    },
    'OSM': {
        'accuracy': None,
        'confidence': 60.9,  # Correlation
        'diversity': 0.3,  # Very low
        'quality': 22.6,  # Overall score
        'training_hours': 12.0,
        'color': '#6A994E',
        'params_k': 351.5,
        'size_mb': 1.3
    },
    'AMP3': {
        'accuracy': None,
        'confidence': None,  # In training
        'diversity': None,
        'quality': None,
        'training_hours': 4.0,  # So far
        'progress': 33.3,
        'color': '#BC4B51',
        'params_k': 2038.3,
        'size_mb': 7.8
    }
}

# Create figure with 2x3 grid
fig = plt.figure(figsize=(18, 10))
gs_main = gs.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

# Title
fig.suptitle('AMP3 Poker AI System - Trained Models Performance Summary',
             fontsize=18, fontweight='bold', y=0.98)

# Panel 1: Overall Quality Scores
ax1 = fig.add_subplot(gs_main[0, 0])
models_quality = ['Preflop', 'Turn', 'River', 'Flop', 'OSM']
quality_scores = [89.6, 59.4, 58.1, 51.2, 22.6]
quality_colors = [models_data[m]['color'] for m in models_quality]

bars1 = ax1.barh(range(len(models_quality)), quality_scores,
                 color=quality_colors, edgecolor='black', linewidth=2, alpha=0.8)
ax1.set_xlabel('Quality Score (0-100)', fontsize=11, fontweight='bold')
ax1.set_title('Model Quality Rankings', fontsize=13, fontweight='bold')
ax1.set_yticks(range(len(models_quality)))
ax1.set_yticklabels(models_quality, fontsize=11)
ax1.set_xlim([0, 100])
ax1.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, score) in enumerate(zip(bars1, quality_scores)):
    ax1.text(score + 2, i, f'{score:.1f}', va='center', fontweight='bold', fontsize=10)

# Panel 2: Confidence vs Diversity
ax2 = fig.add_subplot(gs_main[0, 1])

# Plot each model
for model_name, data in models_data.items():
    if model_name == 'AMP3':  # Skip AMP3 (in training)
        continue
    if data['confidence'] is not None and data['diversity'] is not None:
        ax2.scatter(data['confidence'], data['diversity'],
                   s=400, color=data['color'], edgecolor='black',
                   linewidth=2, alpha=0.8, label=model_name)
        ax2.text(data['confidence'], data['diversity'] + 5, model_name,
                ha='center', fontsize=9, fontweight='bold')

ax2.set_xlabel('Confidence Score (%)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Strategy Diversity (%)', fontsize=11, fontweight='bold')
ax2.set_title('Confidence vs Diversity Trade-off', fontsize=13, fontweight='bold')
ax2.set_xlim([55, 85])
ax2.set_ylim([0, 105])
ax2.grid(alpha=0.3)
ax2.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Diversity Target')
ax2.axvline(x=70, color='blue', linestyle='--', alpha=0.5, label='Confidence Target')
ax2.legend(loc='lower right', fontsize=9)

# Panel 3: Training Efficiency (Score per Hour)
ax3 = fig.add_subplot(gs_main[0, 2])
models_eff = ['Preflop', 'AMP3', 'Turn', 'River', 'Flop', 'OSM']
efficiency = [
    89.6 / 2.0,    # Preflop: 44.8
    33.3 / 4.0,    # AMP3: 8.3 (partial)
    59.4 / 8.0,    # Turn: 7.4
    58.1 / 8.0,    # River: 7.3
    51.2 / 8.0,    # Flop: 6.4
    22.6 / 12.0    # OSM: 1.9
]
eff_colors = [models_data[m]['color'] for m in models_eff]

bars3 = ax3.barh(range(len(models_eff)), efficiency,
                 color=eff_colors, edgecolor='black', linewidth=2, alpha=0.8)
ax3.set_xlabel('Quality Score / Training Hour', fontsize=11, fontweight='bold')
ax3.set_title('Training Efficiency Rankings', fontsize=13, fontweight='bold')
ax3.set_yticks(range(len(models_eff)))
ax3.set_yticklabels(models_eff, fontsize=11)
ax3.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, eff) in enumerate(zip(bars3, efficiency)):
    ax3.text(eff + 1, i, f'{eff:.1f}', va='center', fontweight='bold', fontsize=10)

# Panel 4: Model Size Comparison
ax4 = fig.add_subplot(gs_main[1, 0])
models_size = ['Preflop', 'Flop', 'Turn', 'River', 'OSM', 'AMP3']
sizes_mb = [models_data[m]['size_mb'] for m in models_size]
size_colors = [models_data[m]['color'] for m in models_size]

bars4 = ax4.bar(range(len(models_size)), sizes_mb,
                color=size_colors, edgecolor='black', linewidth=2, alpha=0.8)
ax4.set_ylabel('Model Size (MB)', fontsize=11, fontweight='bold')
ax4.set_title('Model Size Comparison', fontsize=13, fontweight='bold')
ax4.set_xticks(range(len(models_size)))
ax4.set_xticklabels(models_size, fontsize=10, rotation=0)
ax4.grid(axis='y', alpha=0.3)

# Add value labels
for i, (bar, size) in enumerate(zip(bars4, sizes_mb)):
    ax4.text(i, size + 0.3, f'{size:.2f}', ha='center', fontweight='bold', fontsize=9)

# Panel 5: Parameter Count
ax5 = fig.add_subplot(gs_main[1, 1])
models_params = ['Preflop', 'Flop/Turn/River\n(each)', 'OSM', 'AMP3']
params_k = [47.4, 300.8, 351.5, 2038.3]
param_colors = [models_data['Preflop']['color'], models_data['Turn']['color'],
                models_data['OSM']['color'], models_data['AMP3']['color']]

bars5 = ax5.bar(range(len(models_params)), params_k,
                color=param_colors, edgecolor='black', linewidth=2, alpha=0.8)
ax5.set_ylabel('Parameters (thousands)', fontsize=11, fontweight='bold')
ax5.set_title('Model Complexity', fontsize=13, fontweight='bold')
ax5.set_xticks(range(len(models_params)))
ax5.set_xticklabels(models_params, fontsize=10)
ax5.grid(axis='y', alpha=0.3)

# Add value labels
for i, (bar, param) in enumerate(zip(bars5, params_k)):
    ax5.text(i, param + 50, f'{param:.1f}k', ha='center', fontweight='bold', fontsize=9)

# Panel 6: AMP3 Training Status
ax6 = fig.add_subplot(gs_main[1, 2])

# AMP3 progress bar
categories = ['Completed', 'In Progress']
values = [40, 80]  # 40k done, 80k remaining
colors_progress = ['#BC4B51', 'lightgray']

bars6 = ax6.barh(categories, values, color=colors_progress,
                 edgecolor='black', linewidth=2, alpha=0.8)

ax6.set_xlabel('Episodes (thousands)', fontsize=11, fontweight='bold')
ax6.set_title('AMP3 Training Progress (33.3%)', fontsize=13, fontweight='bold')
ax6.set_xlim([0, 120])
ax6.grid(axis='x', alpha=0.3)

# Add labels
ax6.text(20, 0, '40k', ha='center', va='center', fontweight='bold', fontsize=10, color='white')
ax6.text(80, 1, '80k remaining → 120k total', ha='center', va='center',
         fontweight='bold', fontsize=10)

# Add status text
status_text = "Status: Training Active\nCPU: ~160%\nExpected: 8-12 hours"
ax6.text(0.98, 0.02, status_text, transform=ax6.transAxes,
         fontsize=9, verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Add footer with summary
footer_text = (
    "Summary: 4 models production-ready (Preflop: 79.2%, Turn: 59.4, River: 58.1, Flop: 51.2) | "
    "OSM needs retraining (22.6 quality - low diversity) | "
    "AMP3 training in progress (40k→120k episodes)"
)
fig.text(0.5, 0.02, footer_text, ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

# Save figure
plt.tight_layout(rect=[0, 0.04, 1, 0.96])
plt.savefig('presentation_outputs/TRAINED_MODELS_PERFORMANCE.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Performance graph saved: presentation_outputs/TRAINED_MODELS_PERFORMANCE.png")

plt.close()
