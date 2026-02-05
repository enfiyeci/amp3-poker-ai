import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 12)
plt.rcParams['font.size'] = 11

# ============================================================================
# COMPARABLE PERFORMANCE DATA - All models normalized to 0-100 scale
# ============================================================================

models_data = {
    'Preflop': {
        'accuracy': 79.2,
        'params': 47364,
        'training_hours': 2,
        'type': 'Supervised',
        'purpose': 'Preflop Decisions',
        'color': '#3498db'
    },
    'Flop': {
        'quality': 51.2,
        'confidence': 61.1,
        'diversity': 84.0,
        'params': 300810,
        'training_hours': 8,
        'type': 'Supervised',
        'purpose': 'Flop Decisions',
        'color': '#e74c3c'
    },
    'Turn': {
        'quality': 59.4,
        'confidence': 60.6,
        'diversity': 98.0,
        'params': 300810,
        'training_hours': 8,
        'type': 'Supervised',
        'purpose': 'Turn Decisions',
        'color': '#9b59b6'
    },
    'River': {
        'quality': 58.1,
        'confidence': 61.2,
        'diversity': 95.0,
        'params': 300810,
        'training_hours': 8,
        'type': 'Supervised',
        'purpose': 'River Decisions',
        'color': '#16a085'
    },
    'OSM': {
        'quality': 22.6,
        'diversity': 0.3 * 4,  # Scale to 0-100 (was 0.3/25)
        'correlation': 86.8,  # 21.7/25 * 100
        'params': 351524,
        'training_hours': 12,
        'type': 'Analysis',
        'purpose': 'Opponent Modeling',
        'color': '#e67e22'
    },
    'AMP3': {
        'progress': 33.3,  # 40k/120k episodes
        'params': 2038341,
        'training_hours': 4,
        'type': 'RL (A2C)',
        'purpose': 'Full Game Adaptive',
        'color': '#f39c12'
    }
}

# ============================================================================
# FIGURE 1: OVERALL PERFORMANCE COMPARISON
# ============================================================================

fig1 = plt.figure(figsize=(20, 10))
gs1 = fig1.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

fig1.suptitle('Performance Comparison Across All Models', fontsize=20, fontweight='bold', y=0.98)

# Panel 1: Primary Performance Metric (Normalized to 0-100)
ax1 = fig1.add_subplot(gs1[0, :])

models = ['Preflop', 'Flop', 'Turn', 'River', 'OSM', 'AMP3']
performance_scores = [
    79.2,  # Preflop accuracy
    51.2,  # Flop quality
    59.4,  # Turn quality
    58.1,  # River quality
    22.6,  # OSM quality
    33.3   # AMP3 progress (40k/120k)
]
colors_list = [models_data[m]['color'] for m in models]

bars = ax1.bar(range(len(models)), performance_scores, color=colors_list,
               edgecolor='black', linewidth=2.5, alpha=0.8)

ax1.set_ylabel('Performance Score (0-100)', fontsize=14, fontweight='bold')
ax1.set_title('Primary Performance Metric Comparison (Higher is Better)', fontsize=16, fontweight='bold')
ax1.set_xticks(range(len(models)))
ax1.set_xticklabels(models, fontsize=13, fontweight='bold')
ax1.set_ylim([0, 100])
ax1.grid(axis='y', alpha=0.3, linewidth=1.5)
ax1.axhline(y=50, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Good Threshold (50)')
ax1.axhline(y=75, color='blue', linestyle='--', linewidth=2, alpha=0.5, label='Excellent Threshold (75)')
ax1.legend(fontsize=11, loc='upper right')

# Add value labels and status
for i, (bar, score, model) in enumerate(zip(bars, performance_scores, models)):
    height = bar.get_height()

    # Value label
    if model == 'Preflop':
        label = f'{score:.1f}%\nAccuracy'
    elif model == 'AMP3':
        label = f'{score:.1f}%\nComplete'
    else:
        label = f'{score:.1f}\nQuality'

    ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
             label, ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Status indicator
    if score >= 75:
        status = '✓ Excellent'
        status_color = 'green'
    elif score >= 50:
        status = '✓ Good'
        status_color = 'blue'
    elif score >= 30:
        status = '⚠ Moderate'
        status_color = 'orange'
    else:
        status = '✗ Low'
        status_color = 'red'

    ax1.text(bar.get_x() + bar.get_width()/2., 5,
             status, ha='center', va='bottom', fontsize=9,
             color=status_color, fontweight='bold')

# Panel 2: Model Complexity (Parameters)
ax2 = fig1.add_subplot(gs1[1, 0])

params_k = [models_data[m]['params']/1000 for m in models]
bars2 = ax2.barh(models, params_k, color=colors_list, edgecolor='black', linewidth=2, alpha=0.8)

ax2.set_xlabel('Parameters (thousands)', fontsize=12, fontweight='bold')
ax2.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

for i, (bar, param) in enumerate(zip(bars2, params_k)):
    width = bar.get_width()
    label = f'{param:.0f}k' if param < 1000 else f'{param/1000:.1f}M'
    ax2.text(width + 50, i, label, va='center', fontsize=10, fontweight='bold')

# Panel 3: Training Efficiency
ax3 = fig1.add_subplot(gs1[1, 1])

training_hours = [models_data[m]['training_hours'] for m in models]
bars3 = ax3.bar(range(len(models)), training_hours, color=colors_list,
                edgecolor='black', linewidth=2, alpha=0.8)

ax3.set_ylabel('Training Time (hours)', fontsize=12, fontweight='bold')
ax3.set_title('Training Duration Comparison', fontsize=14, fontweight='bold')
ax3.set_xticks(range(len(models)))
ax3.set_xticklabels(models, fontsize=11, rotation=15, ha='right')
ax3.grid(axis='y', alpha=0.3)

for i, (bar, hours) in enumerate(zip(bars3, training_hours)):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.3,
             f'{hours}h', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Panel 4: Efficiency Ratio (Performance / Training Hours)
ax4 = fig1.add_subplot(gs1[1, 2])

efficiency = [perf / hours for perf, hours in zip(performance_scores, training_hours)]
bars4 = ax4.bar(range(len(models)), efficiency, color=colors_list,
                edgecolor='black', linewidth=2, alpha=0.8)

ax4.set_ylabel('Efficiency (Score/Hour)', fontsize=12, fontweight='bold')
ax4.set_title('Training Efficiency Ratio', fontsize=14, fontweight='bold')
ax4.set_xticks(range(len(models)))
ax4.set_xticklabels(models, fontsize=11, rotation=15, ha='right')
ax4.grid(axis='y', alpha=0.3)

for i, (bar, eff) in enumerate(zip(bars4, efficiency)):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{eff:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('presentation_outputs/COMPARABLE_PERFORMANCE.png', dpi=300, bbox_inches='tight')
print("✓ Created: COMPARABLE_PERFORMANCE.png")

# ============================================================================
# FIGURE 2: DECISION MODELS DEEP DIVE
# ============================================================================

fig2 = plt.figure(figsize=(20, 10))
gs2 = fig2.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

fig2.suptitle('Decision Models: Detailed Performance Analysis', fontsize=20, fontweight='bold', y=0.98)

decision_models = ['Preflop', 'Flop', 'Turn', 'River']
decision_colors = [models_data[m]['color'] for m in decision_models]

# Panel 1: Performance Scores Side-by-Side
ax5 = fig2.add_subplot(gs2[0, :])

performance_values = [79.2, 51.2, 59.4, 58.1]
x_pos = np.arange(len(decision_models))

bars5 = ax5.bar(x_pos, performance_values, color=decision_colors,
                edgecolor='black', linewidth=3, alpha=0.8, width=0.6)

ax5.set_ylabel('Performance Score', fontsize=14, fontweight='bold')
ax5.set_title('Decision Models: Performance Score Comparison', fontsize=16, fontweight='bold')
ax5.set_xticks(x_pos)
ax5.set_xticklabels(decision_models, fontsize=13, fontweight='bold')
ax5.set_ylim([0, 100])
ax5.grid(axis='y', alpha=0.3, linewidth=1.5)

# Add metric type labels
for i, (bar, score, model) in enumerate(zip(bars5, performance_values, decision_models)):
    height = bar.get_height()
    metric_type = 'Accuracy (%)' if model == 'Preflop' else 'Quality Score'

    ax5.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{score:.1f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    ax5.text(bar.get_x() + bar.get_width()/2., height - 8,
             metric_type, ha='center', va='top', fontsize=9, style='italic')

# Panel 2: Confidence Levels (Later Streets)
ax6 = fig2.add_subplot(gs2[1, 0])

later_streets = ['Flop', 'Turn', 'River']
confidences = [61.1, 60.6, 61.2]
later_colors = [models_data[m]['color'] for m in later_streets]

bars6 = ax6.bar(range(len(later_streets)), confidences, color=later_colors,
                edgecolor='black', linewidth=2, alpha=0.8)

ax6.set_ylabel('Confidence (%)', fontsize=12, fontweight='bold')
ax6.set_title('Later Streets: Prediction Confidence', fontsize=14, fontweight='bold')
ax6.set_xticks(range(len(later_streets)))
ax6.set_xticklabels(later_streets, fontsize=11)
ax6.set_ylim([0, 100])
ax6.grid(axis='y', alpha=0.3)

for i, (bar, conf) in enumerate(zip(bars6, confidences)):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{conf:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Panel 3: Strategy Diversity (Later Streets)
ax7 = fig2.add_subplot(gs2[1, 1])

diversities = [84.0, 98.0, 95.0]
bars7 = ax7.bar(range(len(later_streets)), diversities, color=later_colors,
                edgecolor='black', linewidth=2, alpha=0.8)

ax7.set_ylabel('Strategy Diversity (%)', fontsize=12, fontweight='bold')
ax7.set_title('Later Streets: Strategic Balance', fontsize=14, fontweight='bold')
ax7.set_xticks(range(len(later_streets)))
ax7.set_xticklabels(later_streets, fontsize=11)
ax7.set_ylim([0, 100])
ax7.axhline(y=80, color='green', linestyle='--', linewidth=2, label='Balanced (80%+)')
ax7.grid(axis='y', alpha=0.3)
ax7.legend(fontsize=10)

for i, (bar, div) in enumerate(zip(bars7, diversities)):
    height = bar.get_height()
    ax7.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{div:.0f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Panel 4: Preflop Action Accuracy Breakdown
ax8 = fig2.add_subplot(gs2[1, 2])

actions = ['Overall', 'Fold', 'Call', 'Raise']
accuracies = [79.2, 85.0, 76.0, 77.0]
action_colors = ['#2c3e50', '#e74c3c', '#3498db', '#2ecc71']

bars8 = ax8.barh(actions, accuracies, color=action_colors, edgecolor='black', linewidth=2)

ax8.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax8.set_title('Preflop: Per-Action Accuracy', fontsize=14, fontweight='bold')
ax8.set_xlim([0, 100])
ax8.grid(axis='x', alpha=0.3)

for i, (bar, acc) in enumerate(zip(bars8, accuracies)):
    width = bar.get_width()
    ax8.text(width + 2, i, f'{acc:.1f}%', va='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('presentation_outputs/DECISION_MODELS_COMPARISON.png', dpi=300, bbox_inches='tight')
print("✓ Created: DECISION_MODELS_COMPARISON.png")

# ============================================================================
# FIGURE 3: SYSTEM MODELS (OSM + AMP3)
# ============================================================================

fig3 = plt.figure(figsize=(20, 10))
gs3 = fig3.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

fig3.suptitle('System Models: OSM & AMP3 Analysis', fontsize=20, fontweight='bold', y=0.98)

# Panel 1: OSM Component Scores
ax9 = fig3.add_subplot(gs3[0, 0])

osm_components = ['Diversity', 'Range\nUsage', 'Correlation', 'Overall']
osm_scores = [1.2, 2.4, 86.8, 22.6]  # Normalized to 0-100

bars9 = ax9.barh(osm_components, osm_scores, color=models_data['OSM']['color'],
                 edgecolor='black', linewidth=2, alpha=0.8)

ax9.set_xlabel('Score (0-100)', fontsize=12, fontweight='bold')
ax9.set_title('OSM: Component Scores', fontsize=14, fontweight='bold')
ax9.set_xlim([0, 100])
ax9.axvline(x=60, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Good Threshold')
ax9.grid(axis='x', alpha=0.3)
ax9.legend(fontsize=9)

for i, (bar, score) in enumerate(zip(bars9, osm_scores)):
    width = bar.get_width()
    ax9.text(width + 2, i, f'{score:.1f}', va='center', fontsize=10, fontweight='bold')

# Panel 2: OSM Style Feature Predictions
ax10 = fig3.add_subplot(gs3[0, 1])

osm_features = ['VPIP', 'PFR', 'AFq', 'WTSD']
osm_means = [61.2, 42.3, 26.7, 39.5]
osm_expected_min = [15, 10, 30, 15]
osm_expected_max = [50, 30, 60, 35]

x = np.arange(len(osm_features))
width = 0.35

# Expected range as error bars
expected_mids = [(min_val + max_val)/2 for min_val, max_val in zip(osm_expected_min, osm_expected_max)]
expected_errs = [(max_val - min_val)/2 for min_val, max_val in zip(osm_expected_min, osm_expected_max)]

ax10.bar(x - width/2, expected_mids, width, yerr=expected_errs, label='Expected Range',
         color='lightgreen', edgecolor='black', linewidth=1.5, alpha=0.5, capsize=5)
ax10.bar(x + width/2, osm_means, width, label='OSM Predictions',
         color=models_data['OSM']['color'], edgecolor='black', linewidth=1.5, alpha=0.8)

ax10.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax10.set_title('OSM: Style Predictions vs Expected', fontsize=14, fontweight='bold')
ax10.set_xticks(x)
ax10.set_xticklabels(osm_features, fontsize=11)
ax10.legend(fontsize=10)
ax10.grid(axis='y', alpha=0.3)
ax10.set_ylim([0, 80])

# Panel 3: AMP3 Training Progress
ax11 = fig3.add_subplot(gs3[0, 2])

amp3_metrics = ['Current\nEpisodes', 'Target\nEpisodes', 'Progress\n(%)']
amp3_values = [40000, 120000, 33.3]
amp3_colors = [models_data['AMP3']['color'], 'lightgray', models_data['AMP3']['color']]

bars11 = ax11.bar(range(len(amp3_metrics)), amp3_values,
                  color=amp3_colors, edgecolor='black', linewidth=2, alpha=0.7)

ax11.set_ylabel('Value', fontsize=12, fontweight='bold')
ax11.set_title('AMP3: Training Progress', fontsize=14, fontweight='bold')
ax11.set_xticks(range(len(amp3_metrics)))
ax11.set_xticklabels(amp3_metrics, fontsize=11)
ax11.grid(axis='y', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars11, amp3_values)):
    height = bar.get_height()
    if i < 2:
        label = f'{val:,.0f}'
    else:
        label = f'{val:.1f}%'
    ax11.text(bar.get_x() + bar.get_width()/2., height + max(amp3_values)*0.02,
              label, ha='center', va='bottom', fontsize=11, fontweight='bold')

# Panel 4: Model Complexity Comparison (All Models)
ax12 = fig3.add_subplot(gs3[1, :])

all_models = ['Preflop', 'Flop/Turn/River\n(each)', 'OSM', 'AMP3\n(Total)', 'AMP3\nActor', 'AMP3\nCritic']
all_params = [47.4, 300.8, 351.5, 2038.3, 1028.1, 1010.3]  # in thousands
all_colors = [models_data['Preflop']['color'], models_data['Turn']['color'],
              models_data['OSM']['color'], models_data['AMP3']['color'],
              models_data['AMP3']['color'], models_data['AMP3']['color']]

bars12 = ax12.bar(range(len(all_models)), all_params, color=all_colors,
                  edgecolor='black', linewidth=2, alpha=0.7)

ax12.set_ylabel('Parameters (thousands)', fontsize=13, fontweight='bold')
ax12.set_title('Model Complexity: Parameter Count Comparison', fontsize=15, fontweight='bold')
ax12.set_xticks(range(len(all_models)))
ax12.set_xticklabels(all_models, fontsize=11)
ax12.grid(axis='y', alpha=0.3)

for i, (bar, param) in enumerate(zip(bars12, all_params)):
    height = bar.get_height()
    label = f'{param:.0f}k' if param < 1000 else f'{param/1000:.1f}M'
    ax12.text(bar.get_x() + bar.get_width()/2., height + 50,
              label, ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('presentation_outputs/SYSTEM_MODELS_COMPARISON.png', dpi=300, bbox_inches='tight')
print("✓ Created: SYSTEM_MODELS_COMPARISON.png")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("COMPARABLE GRAPHS CREATED")
print("="*70)
print("\nGenerated 3 comprehensive comparison graphs:")
print("  1. COMPARABLE_PERFORMANCE.png")
print("     - Overall performance across all 6 models")
print("     - Model complexity and training efficiency")
print("     - Side-by-side comparisons")
print("")
print("  2. DECISION_MODELS_COMPARISON.png")
print("     - Detailed analysis of 4 decision models")
print("     - Confidence and diversity metrics")
print("     - Preflop action breakdown")
print("")
print("  3. SYSTEM_MODELS_COMPARISON.png")
print("     - OSM component analysis")
print("     - OSM predictions vs expected ranges")
print("     - AMP3 training progress")
print("     - Complete parameter comparison")
print("="*70 + "\n")
