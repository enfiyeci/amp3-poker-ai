#!/usr/bin/env python3
"""
AMP3 vs Specialized Models: Capabilities & Performance Focus
No training time metrics - focuses on what the models can do
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import numpy as np

# Set style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.family'] = 'sans-serif'

# Model data
models_data = {
    'Preflop': {
        'quality': 89.6,
        'accuracy': 79.2,
        'confidence': 79.2,
        'diversity': 100.0,
        'params_k': 47.4,
        'size_mb': 0.569,
        'color': '#2E86AB',
        'opponent_adapt': 0,
        'streets_covered': 1,
        'learning_type': 'Supervised'
    },
    'Flop': {
        'quality': 51.2,
        'accuracy': None,
        'confidence': 61.0,
        'diversity': 84.0,
        'params_k': 300.8,
        'size_mb': 1.17,
        'color': '#A23B72',
        'opponent_adapt': 0,
        'streets_covered': 1,
        'learning_type': 'Supervised'
    },
    'Turn': {
        'quality': 59.4,
        'accuracy': None,
        'confidence': 63.0,
        'diversity': 98.0,
        'params_k': 300.8,
        'size_mb': 1.17,
        'color': '#F18F01',
        'opponent_adapt': 0,
        'streets_covered': 1,
        'learning_type': 'Supervised'
    },
    'River': {
        'quality': 58.1,
        'accuracy': None,
        'confidence': 62.0,
        'diversity': 95.0,
        'params_k': 300.8,
        'size_mb': 1.17,
        'color': '#C73E1D',
        'opponent_adapt': 0,
        'streets_covered': 1,
        'learning_type': 'Supervised'
    },
    'OSM': {
        'quality': 22.6,
        'accuracy': None,
        'confidence': 60.9,
        'diversity': 0.3,
        'params_k': 351.5,
        'size_mb': 1.3,
        'color': '#6A994E',
        'opponent_adapt': 1,
        'streets_covered': 4,
        'learning_type': 'Supervised'
    },
    'AMP3': {
        'quality': 65.0,  # Estimated final value
        'accuracy': None,
        'confidence': 68.0,  # Estimated
        'diversity': 92.0,  # Estimated
        'params_k': 2038.3,
        'size_mb': 7.8,
        'color': '#BC4B51',
        'opponent_adapt': 1,
        'streets_covered': 4,
        'learning_type': 'Reinforcement',
        'progress': 100.0
    }
}

# Create figure
fig = plt.figure(figsize=(20, 12))
gs_main = gs.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

# Main title
fig.suptitle('AMP3 vs Specialized Models: Capabilities & Performance Analysis',
             fontsize=20, fontweight='bold', y=0.98)

# ============================================================================
# ROW 1: Performance Metrics
# ============================================================================

# Panel 1: Quality Scores (Complete Models)
ax1 = fig.add_subplot(gs_main[0, 0])

models_quality = ['Preflop', 'AMP3', 'Turn', 'River', 'Flop', 'OSM']
quality_scores = [models_data[m]['quality'] for m in models_quality]
quality_colors = [models_data[m]['color'] for m in models_quality]

bars1 = ax1.barh(range(len(models_quality)), quality_scores,
                 color=quality_colors, edgecolor='black', linewidth=2, alpha=0.8)

# Highlight AMP3
bars1[1].set_edgecolor('red')
bars1[1].set_linewidth(3)

ax1.set_xlabel('Quality Score (0-100)', fontsize=12, fontweight='bold')
ax1.set_title('Performance Quality Rankings', fontsize=14, fontweight='bold')
ax1.set_yticks(range(len(models_quality)))
ax1.set_yticklabels(models_quality, fontsize=12)
ax1.set_xlim([0, 100])
ax1.grid(axis='x', alpha=0.3)

# Add values
for i, (bar, score) in enumerate(zip(bars1, quality_scores)):
    width = bar.get_width()
    label = f'{score:.1f}*' if models_quality[i] == 'AMP3' else f'{score:.1f}'
    ax1.text(width + 2, i, label, va='center', fontweight='bold', fontsize=11)

# Add AMP3 note
ax1.text(50, -1.3, 'AMP3: Estimated Quality (Training Complete)', ha='center',
         fontsize=11, fontweight='bold', color='#BC4B51',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Panel 2: Confidence vs Diversity Scatter
ax2 = fig.add_subplot(gs_main[0, 1])

for model_name, data in models_data.items():
    if data['confidence'] is not None and data['diversity'] is not None:
        size = 500 if model_name == 'Preflop' else 400
        if model_name == 'AMP3':
            size = 600
            ax2.scatter(data['confidence'], data['diversity'],
                       s=size, color=data['color'], edgecolor='red',
                       linewidth=3, alpha=0.8, label=model_name, zorder=4, marker='s')
            ax2.text(data['confidence'], data['diversity'] - 7, model_name + '*',
                    ha='center', fontsize=10, fontweight='bold', zorder=5)
        else:
            ax2.scatter(data['confidence'], data['diversity'],
                       s=size, color=data['color'], edgecolor='black',
                       linewidth=2, alpha=0.8, label=model_name, zorder=3)
            # Add labels
            offset_y = 7 if model_name != 'OSM' else -7
            ax2.text(data['confidence'], data['diversity'] + offset_y, model_name,
                    ha='center', fontsize=10, fontweight='bold', zorder=4)

ax2.set_xlabel('Confidence Score (%)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Strategy Diversity (%)', fontsize=12, fontweight='bold')
ax2.set_title('Confidence vs Diversity Trade-off', fontsize=14, fontweight='bold')
ax2.set_xlim([55, 85])
ax2.set_ylim([-10, 110])
ax2.grid(alpha=0.3, zorder=1)

# Add target zones
ax2.axhline(y=70, color='green', linestyle='--', alpha=0.4, label='Good Diversity', zorder=2)
ax2.axvline(x=70, color='blue', linestyle='--', alpha=0.4, label='Good Confidence', zorder=2)
ax2.fill_between([70, 85], 70, 110, alpha=0.1, color='green', zorder=1)
ax2.text(77.5, 90, 'Ideal Zone', fontsize=10, ha='center', style='italic', color='green')

# Panel 3: Model Complexity vs Quality
ax3 = fig.add_subplot(gs_main[0, 2])

# Plot complete models
models_complete = ['Preflop', 'Turn', 'River', 'Flop', 'OSM', 'AMP3']
for model in models_complete:
    data = models_data[model]
    size = 600 if model in ['Preflop', 'AMP3'] else 400
    if model == 'AMP3':
        ax3.scatter(data['params_k'], data['quality'],
                   s=size, color=data['color'], edgecolor='red',
                   linewidth=3, alpha=0.8, marker='s', zorder=4)
        ax3.text(data['params_k'], data['quality'] - 7, model + '*',
                ha='center', fontsize=10, fontweight='bold', color='#BC4B51', zorder=5)
    else:
        ax3.scatter(data['params_k'], data['quality'],
                   s=size, color=data['color'], edgecolor='black',
                   linewidth=2, alpha=0.8, zorder=3)
        # Add labels
        offset_y = 5 if model != 'OSM' else -5
        ax3.text(data['params_k'], data['quality'] + offset_y, model,
                ha='center', fontsize=10, fontweight='bold', zorder=4)

ax3.set_xlabel('Model Complexity (Parameters in thousands)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Quality Score', fontsize=12, fontweight='bold')
ax3.set_title('Complexity vs Quality', fontsize=14, fontweight='bold')
ax3.grid(alpha=0.3, zorder=1)
ax3.set_xlim([0, 2200])
ax3.set_ylim([0, 100])

# Add efficiency line
ax3.text(200, 85, 'Smaller models can be\nmore efficient', fontsize=9,
        style='italic', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

# ============================================================================
# ROW 2: Capabilities & Architecture
# ============================================================================

# Panel 4: Coverage & Capabilities Matrix
ax4 = fig.add_subplot(gs_main[1, 0])

capabilities = ['Preflop\nDecisions', 'Flop\nDecisions', 'Turn\nDecisions',
                'River\nDecisions', 'Opponent\nModeling', 'Strategy\nAdaptation']
models_cap = ['Preflop', 'Flop', 'Turn', 'River', 'OSM', 'AMP3']

# Create capability matrix
capability_matrix = np.array([
    [1, 0, 0, 0, 0, 0],  # Preflop
    [0, 1, 0, 0, 0, 0],  # Flop
    [0, 0, 1, 0, 0, 0],  # Turn
    [0, 0, 0, 1, 0, 0],  # River
    [0, 0, 0, 0, 1, 0],  # OSM
    [1, 1, 1, 1, 1, 1],  # AMP3
])

im = ax4.imshow(capability_matrix.T, cmap='RdYlGn', aspect='auto', alpha=0.8,
                vmin=0, vmax=1, interpolation='nearest')

ax4.set_xticks(range(len(models_cap)))
ax4.set_xticklabels(models_cap, fontsize=11, rotation=45, ha='right')
ax4.set_yticks(range(len(capabilities)))
ax4.set_yticklabels(capabilities, fontsize=11)
ax4.set_title('Capability Coverage Matrix', fontsize=14, fontweight='bold')

# Add checkmarks and X marks
for i in range(len(models_cap)):
    for j in range(len(capabilities)):
        text = '✓' if capability_matrix[i, j] == 1 else '✗'
        color = 'white' if capability_matrix[i, j] == 1 else 'gray'
        ax4.text(i, j, text, ha='center', va='center',
                fontsize=16, fontweight='bold', color=color)

# Add border for AMP3
from matplotlib.patches import Rectangle
rect = Rectangle((5-0.5, -0.5), 1, 6, linewidth=4, edgecolor='red',
                 facecolor='none', zorder=10)
ax4.add_patch(rect)
ax4.text(5, -1, 'Full Coverage', ha='center', fontsize=10,
         fontweight='bold', color='red')

# Panel 5: System Architecture Comparison
ax5 = fig.add_subplot(gs_main[1, 1])

# Specialized system
specialized_params = [
    models_data['Preflop']['params_k'],
    models_data['Flop']['params_k'],
    models_data['Turn']['params_k'],
    models_data['River']['params_k']
]
specialized_labels = ['Preflop', 'Flop', 'Turn', 'River']
specialized_colors = [models_data[m]['color'] for m in specialized_labels]

# Create stacked bar for specialized
bottom = 0
for i, (params, label, color) in enumerate(zip(specialized_params, specialized_labels, specialized_colors)):
    ax5.bar(0, params, bottom=bottom, color=color, edgecolor='black',
            linewidth=2, alpha=0.8, width=0.6)
    # Add label in the middle of each segment
    ax5.text(0, bottom + params/2, label, ha='center', va='center',
             fontweight='bold', fontsize=10, color='white')
    bottom += params

# Add OSM separately
ax5.bar(0.8, models_data['OSM']['params_k'], color=models_data['OSM']['color'],
        edgecolor='black', linewidth=2, alpha=0.8, width=0.6)
ax5.text(0.8, models_data['OSM']['params_k']/2, 'OSM', ha='center', va='center',
         fontweight='bold', fontsize=10, color='white')

# AMP3 unified
ax5.bar(2, models_data['AMP3']['params_k'], color=models_data['AMP3']['color'],
        edgecolor='red', linewidth=4, alpha=0.8, width=0.6)
ax5.text(2, models_data['AMP3']['params_k']/2, 'AMP3\nUnified', ha='center', va='center',
         fontweight='bold', fontsize=11, color='white')

ax5.set_ylabel('Parameters (thousands)', fontsize=12, fontweight='bold')
ax5.set_title('Architecture: Specialized vs Unified', fontsize=14, fontweight='bold')
ax5.set_xticks([0, 0.8, 2])
ax5.set_xticklabels(['Decision\nModels\n(4 models)', 'Opponent\nModel\n(1 model)',
                     'AMP3\n(1 unified)'], fontsize=10)
ax5.set_xlim([-0.5, 2.5])
ax5.grid(axis='y', alpha=0.3)

# Add totals
specialized_total = sum(specialized_params)
ax5.text(0, specialized_total + 100, f'Total:\n{specialized_total:.0f}k',
         ha='center', fontsize=9, fontweight='bold')
ax5.text(0.8, models_data['OSM']['params_k'] + 100, f'{models_data["OSM"]["params_k"]:.0f}k',
         ha='center', fontsize=9, fontweight='bold')
ax5.text(2, models_data['AMP3']['params_k'] + 100, f'{models_data["AMP3"]["params_k"]:.0f}k',
         ha='center', fontsize=9, fontweight='bold')

# Panel 6: Deployment Size Breakdown
ax6 = fig.add_subplot(gs_main[1, 2])

all_models = ['Preflop', 'Flop', 'Turn', 'River', 'OSM', 'AMP3']
sizes = [models_data[m]['size_mb'] for m in all_models]
colors_all = [models_data[m]['color'] for m in all_models]

bars6 = ax6.bar(range(len(all_models)), sizes, color=colors_all,
                edgecolor='black', linewidth=2, alpha=0.8)

# Highlight AMP3
bars6[-1].set_linewidth(4)
bars6[-1].set_edgecolor('red')

ax6.set_ylabel('File Size (MB)', fontsize=12, fontweight='bold')
ax6.set_title('Deployment Package Size', fontsize=14, fontweight='bold')
ax6.set_xticks(range(len(all_models)))
ax6.set_xticklabels(all_models, fontsize=10, rotation=45, ha='right')
ax6.grid(axis='y', alpha=0.3)

# Add values
for i, (bar, size) in enumerate(zip(bars6, sizes)):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height + 0.2,
             f'{size:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Add comparison annotations
specialized_size = sum(sizes[:4])
ax6.text(1.5, 6, f'4 Specialized Models:\n{specialized_size:.2f} MB total',
         ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
ax6.text(5, 9, f'AMP3: {sizes[5]:.2f} MB\n(1 unified model)',
         ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

# ============================================================================
# ROW 3: Strategic Analysis
# ============================================================================

# Panel 7: Model Strengths Radar Chart
ax7 = fig.add_subplot(gs_main[2, 0], projection='polar')

categories = ['Accuracy', 'Diversity', 'Compactness', 'Adaptability', 'Coverage']
N = len(categories)

# Preflop (best specialized)
preflop_scores = [89.6, 80, 95, 0, 25]  # Scaled to 0-100
angles = [n / float(N) * 2 * np.pi for n in range(N)]
preflop_scores += preflop_scores[:1]
angles += angles[:1]

ax7.plot(angles, preflop_scores, 'o-', linewidth=2, label='Preflop (Best Specialized)',
         color=models_data['Preflop']['color'])
ax7.fill(angles, preflop_scores, alpha=0.25, color=models_data['Preflop']['color'])

# AMP3 (estimated)
amp3_scores = [60, 70, 20, 100, 100]  # Estimated
amp3_scores += amp3_scores[:1]
ax7.plot(angles, amp3_scores, 's-', linewidth=2, label='AMP3 (Estimated)',
         color=models_data['AMP3']['color'])
ax7.fill(angles, amp3_scores, alpha=0.25, color=models_data['AMP3']['color'])

ax7.set_xticks(angles[:-1])
ax7.set_xticklabels(categories, fontsize=10)
ax7.set_ylim(0, 100)
ax7.set_title('Capability Profile Comparison', fontsize=14, fontweight='bold', pad=20)
ax7.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
ax7.grid(True)

# Panel 8: Use Case Suitability
ax8 = fig.add_subplot(gs_main[2, 1])
ax8.axis('off')

use_cases_text = """
BEST USE CASES:

SPECIALIZED MODELS (Preflop + Streets + OSM)
• Production deployment NOW
• Known, stable opponents
• Performance-critical applications
• Limited computational resources
• Proven, validated quality needed
• GTO-based strategy preferred

AMP3 (UNIFIED MODEL)
• Dynamic, varying opponents
• Adaptive strategy required
• Long-term self-improvement
• Research & development
• Future production (post-validation)
• Opponent exploitation focus
• Continuous learning systems
"""

ax8.text(0.05, 0.95, use_cases_text, transform=ax8.transAxes,
         fontsize=10, verticalalignment='top', horizontalalignment='left',
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

ax8.text(0.5, 1.05, 'Deployment Strategy', transform=ax8.transAxes,
         fontsize=14, fontweight='bold', ha='center')

# Panel 9: Key Differences Summary
ax9 = fig.add_subplot(gs_main[2, 2])
ax9.axis('off')

differences_text = """
FUNDAMENTAL DIFFERENCES:

LEARNING APPROACH
Specialized: Learn from expert data
AMP3: Learn through self-play

STRATEGY
Specialized: Static GTO strategy
AMP3: Adaptive, exploitative

ARCHITECTURE
Specialized: 4-5 separate models
AMP3: Single unified network

OPPONENT HANDLING
Specialized: Same strategy for all
AMP3: Adapts to each opponent

VALIDATION
Specialized: 79.2% accuracy proven
AMP3: 65.0 quality (estimated)

COMPLEXITY
Specialized: 950k total params
AMP3: 2,038k params (2.1x larger)

DEPLOYMENT
Specialized: 4.6 MB total
AMP3: 7.8 MB single file
"""

ax9.text(0.05, 0.95, differences_text, transform=ax9.transAxes,
         fontsize=10, verticalalignment='top', horizontalalignment='left',
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

ax9.text(0.5, 1.05, 'Core Distinctions', transform=ax9.transAxes,
         fontsize=14, fontweight='bold', ha='center')

# Footer
footer_text = (
    "SUMMARY: Specialized models offer proven quality and efficiency for production use. "
    "AMP3 provides unified architecture with opponent adaptation capabilities (estimated values*). "
    "Both approaches are valid: specialized for immediate deployment, AMP3 for adaptive future systems."
)
fig.text(0.5, 0.01, footer_text, ha='center', fontsize=10, style='italic',
         bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))

# Save
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig('presentation_outputs/AMP3_CAPABILITIES_COMPARISON.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("✅ AMP3 capabilities comparison saved: presentation_outputs/AMP3_CAPABILITIES_COMPARISON.png")

plt.close()
