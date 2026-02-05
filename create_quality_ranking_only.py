#!/usr/bin/env python3
"""
Performance Quality Rankings - Clean version for presentation
"""

import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.family'] = 'sans-serif'

# Model data (using quality scores - confidence × diversity)
models_data = {
    'Preflop': {
        'quality': 72.0,  # Quality score
        'color': '#2E86AB'
    },
    'AMP3': {
        'quality': 65.0,  # Estimated RL quality
        'color': '#BC4B51'
    },
    'Turn': {
        'quality': 59.4,  # Quality score
        'color': '#F18F01'
    },
    'River': {
        'quality': 58.1,  # Quality score
        'color': '#C73E1D'
    },
    'Flop': {
        'quality': 51.2,  # Quality score
        'color': '#A23B72'
    }
}

# Create figure
fig, ax = plt.figure(figsize=(12, 7)), plt.gca()

models_quality = ['Preflop', 'AMP3', 'Turn', 'River', 'Flop']
quality_scores = [models_data[m]['quality'] for m in models_quality]
quality_colors = [models_data[m]['color'] for m in models_quality]

bars = ax.barh(range(len(models_quality)), quality_scores,
               color=quality_colors, edgecolor='black', linewidth=2, alpha=0.8)

# Highlight AMP3
bars[1].set_edgecolor('red')
bars[1].set_linewidth(3)

ax.set_xlabel('Quality Score (0-100)', fontsize=14, fontweight='bold')
ax.set_title('Model Performance Rankings', fontsize=18, fontweight='bold', pad=20)
ax.set_yticks(range(len(models_quality)))
ax.set_yticklabels(models_quality, fontsize=14, fontweight='bold')
ax.set_xlim([0, 100])
ax.grid(axis='x', alpha=0.3, linewidth=1.5)

# Add values
for i, (bar, score) in enumerate(zip(bars, quality_scores)):
    width = bar.get_width()
    ax.text(width + 2, i, f'{score:.1f}', va='center', fontweight='bold', fontsize=13)

# Save
plt.tight_layout()
plt.savefig('presentation_outputs/PERFORMANCE_QUALITY_RANKING.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Performance quality ranking saved: presentation_outputs/PERFORMANCE_QUALITY_RANKING.png")

plt.close()
