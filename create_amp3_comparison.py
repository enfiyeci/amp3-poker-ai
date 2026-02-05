#!/usr/bin/env python3
"""
AMP3 Model Comparison Analysis
Detailed comparison of AMP3 vs other specialized models
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
        'type': 'Supervised',
        'quality': 89.6,
        'training_hours': 2.0,
        'params_k': 47.4,
        'size_mb': 0.569,
        'efficiency': 44.8,
        'color': '#2E86AB',
        'status': 'Complete'
    },
    'Flop': {
        'type': 'Supervised',
        'quality': 51.2,
        'training_hours': 8.0,
        'params_k': 300.8,
        'size_mb': 1.17,
        'efficiency': 6.4,
        'color': '#A23B72',
        'status': 'Complete'
    },
    'Turn': {
        'type': 'Supervised',
        'quality': 59.4,
        'training_hours': 8.0,
        'params_k': 300.8,
        'size_mb': 1.17,
        'efficiency': 7.4,
        'color': '#F18F01',
        'status': 'Complete'
    },
    'River': {
        'type': 'Supervised',
        'quality': 58.1,
        'training_hours': 8.0,
        'params_k': 300.8,
        'size_mb': 1.17,
        'efficiency': 7.3,
        'color': '#C73E1D',
        'status': 'Complete'
    },
    'OSM': {
        'type': 'Supervised',
        'quality': 22.6,
        'training_hours': 12.0,
        'params_k': 351.5,
        'size_mb': 1.3,
        'efficiency': 1.9,
        'color': '#6A994E',
        'status': 'Complete'
    },
    'AMP3': {
        'type': 'Reinforcement',
        'quality': None,  # In training
        'training_hours': 4.0,
        'params_k': 2038.3,
        'size_mb': 7.8,
        'efficiency': 8.3,  # Projected
        'color': '#BC4B51',
        'status': 'Training (33%)',
        'progress': 33.3,
        'episodes_current': 40000,
        'episodes_target': 120000
    }
}

# Create figure
fig = plt.figure(figsize=(20, 12))
gs_main = gs.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

# Main title
fig.suptitle('AMP3 vs Specialized Models: Comprehensive Comparison Analysis',
             fontsize=20, fontweight='bold', y=0.98)

# ============================================================================
# ROW 1: Model Architecture & Type Comparison
# ============================================================================

# Panel 1: Model Type & Approach
ax1 = fig.add_subplot(gs_main[0, 0])

supervised_models = ['Preflop', 'Flop', 'Turn', 'River', 'OSM']
rl_models = ['AMP3']

y_pos = 0
colors_list = []
labels_list = []

# Plot supervised models
for i, model in enumerate(supervised_models):
    ax1.barh(y_pos, 1, height=0.8, color=models_data[model]['color'],
             edgecolor='black', linewidth=2, alpha=0.8)
    ax1.text(0.5, y_pos, model, ha='center', va='center',
             fontweight='bold', fontsize=11, color='white')
    y_pos += 1

# Separator
y_pos += 0.5

# Plot RL model
ax1.barh(y_pos, 1, height=0.8, color=models_data['AMP3']['color'],
         edgecolor='black', linewidth=3, alpha=0.8)
ax1.text(0.5, y_pos, 'AMP3', ha='center', va='center',
         fontweight='bold', fontsize=12, color='white')

ax1.set_xlim([0, 1])
ax1.set_ylim([-0.5, y_pos + 1])
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title('Learning Paradigms', fontsize=14, fontweight='bold')

# Add labels
ax1.text(1.1, 2, 'Supervised\nLearning', fontsize=11, va='center', fontweight='bold')
ax1.text(1.1, y_pos, 'Reinforcement\nLearning', fontsize=11, va='center', fontweight='bold',
         color='#BC4B51')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)

# Panel 2: Parameter Count Comparison
ax2 = fig.add_subplot(gs_main[0, 1])

all_models = ['Preflop', 'Flop', 'Turn', 'River', 'OSM', 'AMP3']
params = [models_data[m]['params_k'] for m in all_models]
colors = [models_data[m]['color'] for m in all_models]

bars2 = ax2.bar(range(len(all_models)), params, color=colors,
                edgecolor='black', linewidth=2, alpha=0.8)

# Highlight AMP3
bars2[-1].set_linewidth(4)
bars2[-1].set_edgecolor('red')

ax2.set_ylabel('Parameters (thousands)', fontsize=12, fontweight='bold')
ax2.set_title('Model Complexity', fontsize=14, fontweight='bold')
ax2.set_xticks(range(len(all_models)))
ax2.set_xticklabels(all_models, fontsize=10, rotation=45, ha='right')
ax2.grid(axis='y', alpha=0.3)

# Add values
for i, (bar, param) in enumerate(zip(bars2, params)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 50,
             f'{param:.0f}k', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Add annotation for AMP3
ax2.annotate('43x larger\nthan Preflop', xy=(5, params[5]), xytext=(4, 1500),
             arrowprops=dict(arrowstyle='->', color='red', lw=2),
             fontsize=10, fontweight='bold', color='red')

# Panel 3: Model Size (MB)
ax3 = fig.add_subplot(gs_main[0, 2])

sizes = [models_data[m]['size_mb'] for m in all_models]

bars3 = ax3.bar(range(len(all_models)), sizes, color=colors,
                edgecolor='black', linewidth=2, alpha=0.8)

# Highlight AMP3
bars3[-1].set_linewidth(4)
bars3[-1].set_edgecolor('red')

ax3.set_ylabel('File Size (MB)', fontsize=12, fontweight='bold')
ax3.set_title('Deployment Size', fontsize=14, fontweight='bold')
ax3.set_xticks(range(len(all_models)))
ax3.set_xticklabels(all_models, fontsize=10, rotation=45, ha='right')
ax3.grid(axis='y', alpha=0.3)

# Add values
for i, (bar, size) in enumerate(zip(bars3, sizes)):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.2,
             f'{size:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# ============================================================================
# ROW 2: Training & Performance Comparison
# ============================================================================

# Panel 4: Training Time
ax4 = fig.add_subplot(gs_main[1, 0])

training_hours = [models_data[m]['training_hours'] for m in all_models]

bars4 = ax4.barh(range(len(all_models)), training_hours, color=colors,
                 edgecolor='black', linewidth=2, alpha=0.8)

# Highlight AMP3
bars4[-1].set_linewidth(4)
bars4[-1].set_edgecolor('red')

ax4.set_xlabel('Training Time (hours)', fontsize=12, fontweight='bold')
ax4.set_title('Training Duration', fontsize=14, fontweight='bold')
ax4.set_yticks(range(len(all_models)))
ax4.set_yticklabels(all_models, fontsize=11)
ax4.grid(axis='x', alpha=0.3)

# Add values and status
for i, (bar, hours) in enumerate(zip(bars4, training_hours)):
    width = bar.get_width()
    status = models_data[all_models[i]]['status']
    if i == 5:  # AMP3
        ax4.text(width + 0.3, i, f'{hours}h (ongoing → ~12h total)', va='center',
                 fontsize=9, fontweight='bold', color='red')
    else:
        ax4.text(width + 0.3, i, f'{hours}h', va='center', fontsize=9, fontweight='bold')

# Panel 5: Training Efficiency
ax5 = fig.add_subplot(gs_main[1, 1])

models_eff = ['Preflop', 'AMP3', 'Turn', 'River', 'Flop', 'OSM']
efficiency_vals = [models_data[m]['efficiency'] for m in models_eff]
colors_eff = [models_data[m]['color'] for m in models_eff]

bars5 = ax5.barh(range(len(models_eff)), efficiency_vals, color=colors_eff,
                 edgecolor='black', linewidth=2, alpha=0.8)

# Highlight AMP3
bars5[1].set_linewidth(4)
bars5[1].set_edgecolor('red')

ax5.set_xlabel('Quality Score / Training Hour', fontsize=12, fontweight='bold')
ax5.set_title('Training Efficiency (ROI)', fontsize=14, fontweight='bold')
ax5.set_yticks(range(len(models_eff)))
ax5.set_yticklabels(models_eff, fontsize=11)
ax5.grid(axis='x', alpha=0.3)

# Add values
for i, (bar, eff) in enumerate(zip(bars5, efficiency_vals)):
    width = bar.get_width()
    if i == 1:  # AMP3
        ax5.text(width + 1, i, f'{eff:.1f} (projected)', va='center',
                 fontsize=9, fontweight='bold', color='red')
    else:
        ax5.text(width + 1, i, f'{eff:.1f}', va='center', fontsize=9, fontweight='bold')

# Panel 6: Quality Scores (Complete models only)
ax6 = fig.add_subplot(gs_main[1, 2])

models_complete = ['Preflop', 'Turn', 'River', 'Flop', 'OSM']
quality_complete = [models_data[m]['quality'] for m in models_complete]
colors_complete = [models_data[m]['color'] for m in models_complete]

bars6 = ax6.barh(range(len(models_complete)), quality_complete,
                 color=colors_complete, edgecolor='black', linewidth=2, alpha=0.8)

ax6.set_xlabel('Quality Score (0-100)', fontsize=12, fontweight='bold')
ax6.set_title('Validated Performance', fontsize=14, fontweight='bold')
ax6.set_yticks(range(len(models_complete)))
ax6.set_yticklabels(models_complete, fontsize=11)
ax6.set_xlim([0, 100])
ax6.grid(axis='x', alpha=0.3)

# Add values
for i, (bar, qual) in enumerate(zip(bars6, quality_complete)):
    width = bar.get_width()
    ax6.text(width + 2, i, f'{qual:.1f}', va='center', fontsize=9, fontweight='bold')

# Add AMP3 note
ax6.text(50, -1.2, 'AMP3: Quality TBD (training in progress)',
         ha='center', fontsize=10, fontweight='bold', color='#BC4B51',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ============================================================================
# ROW 3: AMP3-Specific Analysis
# ============================================================================

# Panel 7: AMP3 Training Progress
ax7 = fig.add_subplot(gs_main[2, 0])

episodes = [40, 80]
labels = ['Completed\n(40k)', 'Remaining\n(80k → 120k)']
colors_progress = ['#BC4B51', 'lightgray']

bars7 = ax7.barh(range(len(labels)), episodes, color=colors_progress,
                 edgecolor='black', linewidth=2, alpha=0.8)

ax7.set_xlabel('Episodes (thousands)', fontsize=12, fontweight='bold')
ax7.set_title('AMP3 Training Progress (33.3%)', fontsize=14, fontweight='bold')
ax7.set_yticks(range(len(labels)))
ax7.set_yticklabels(labels, fontsize=11)
ax7.set_xlim([0, 120])
ax7.grid(axis='x', alpha=0.3)

# Add progress bar effect
ax7.text(20, 0, '40k', ha='center', va='center', fontweight='bold',
         fontsize=11, color='white')
ax7.text(80, 1, '80k episodes remaining', ha='center', va='center',
         fontweight='bold', fontsize=10)

# Panel 8: Specialized vs Unified Architecture
ax8 = fig.add_subplot(gs_main[2, 1])

# Specialized approach (4 models)
specialized_total = (models_data['Preflop']['params_k'] +
                     models_data['Flop']['params_k'] +
                     models_data['Turn']['params_k'] +
                     models_data['River']['params_k'])

# Unified approach (AMP3)
unified_total = models_data['AMP3']['params_k']

approaches = ['Specialized\n(4 models)', 'Unified\n(AMP3)']
param_totals = [specialized_total, unified_total]
approach_colors = ['#2E86AB', '#BC4B51']

bars8 = ax8.bar(range(len(approaches)), param_totals, color=approach_colors,
                edgecolor='black', linewidth=3, alpha=0.8)

ax8.set_ylabel('Total Parameters (thousands)', fontsize=12, fontweight='bold')
ax8.set_title('Architecture Comparison', fontsize=14, fontweight='bold')
ax8.set_xticks(range(len(approaches)))
ax8.set_xticklabels(approaches, fontsize=11)
ax8.grid(axis='y', alpha=0.3)

# Add values
for i, (bar, params) in enumerate(zip(bars8, param_totals)):
    height = bar.get_height()
    ax8.text(bar.get_x() + bar.get_width()/2., height + 50,
             f'{params:.0f}k', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add comparison
diff = unified_total - specialized_total
ax8.text(0.5, 1300, f'AMP3 is {diff:.0f}k parameters\nlarger (~{diff/specialized_total*100:.0f}% increase)',
         ha='center', fontsize=10, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

# Panel 9: Key Advantages & Trade-offs
ax9 = fig.add_subplot(gs_main[2, 2])
ax9.axis('off')

# Create comparison table
advantages_text = """
SPECIALIZED MODELS (Preflop + Streets)
✅ Faster training (2-8 hours each)
✅ Higher proven quality (79.2%, 59.4, 58.1, 51.2)
✅ Smaller individual models
✅ Easy to update specific streets
❌ No opponent adaptation
❌ Static strategy
❌ 4 separate models to deploy

AMP3 (UNIFIED MODEL)
✅ Single end-to-end model
✅ Opponent adaptation capability
✅ Self-improving via RL
✅ Handles all game situations
❌ Longer training (12 hours)
❌ Quality not yet validated
❌ Larger model size (7.8 MB)
❌ Limited by OSM diversity issue
"""

ax9.text(0.05, 0.95, advantages_text, transform=ax9.transAxes,
         fontsize=10, verticalalignment='top', horizontalalignment='left',
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

# Add title
ax9.text(0.5, 1.05, 'Strategic Trade-offs', transform=ax9.transAxes,
         fontsize=14, fontweight='bold', ha='center')

# Add footer with key insight
footer_text = (
    "KEY INSIGHT: AMP3 is 2-3x larger but provides opponent adaptation and unified architecture. "
    "Specialized models are production-ready now with proven quality. "
    "AMP3 represents future direction: adaptive, self-improving AI. "
    "Training completes in ~8 hours."
)
fig.text(0.5, 0.01, footer_text, ha='center', fontsize=10, style='italic',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

# Save
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig('presentation_outputs/AMP3_COMPARISON_ANALYSIS.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("✅ AMP3 comparison graph saved: presentation_outputs/AMP3_COMPARISON_ANALYSIS.png")

plt.close()
