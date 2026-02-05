#!/usr/bin/env python3
"""
Test OSM (Opponent Style Modeling) Network
Evaluates the trained OSM model's ability to predict opponent styles
"""

import torch
import numpy as np
from collections import defaultdict
from osm_network import StyleModelingNetwork

print("="*70)
print("OSM (OPPONENT STYLE MODELING) EVALUATION")
print("="*70)

# Load OSM model
checkpoint_path = 'checkpoints_20hr/osm_best.pt'
print(f"\nLoading OSM model from: {checkpoint_path}")

osm_model = StyleModelingNetwork()
state_dict = torch.load(checkpoint_path, map_location='cpu')
osm_model.load_state_dict(state_dict)
osm_model.eval()

print("✓ OSM model loaded successfully")
print(f"  Parameters: {sum(p.numel() for p in osm_model.parameters()):,}")

# Test OSM with synthetic game scenarios
num_tests = 1000
print(f"\nTesting OSM with {num_tests} synthetic game scenarios...")

predictions = []
feature_ranges = defaultdict(list)

with torch.no_grad():
    for _ in range(num_tests):
        # Create synthetic inputs
        # Public cards: 5 cards * 2 features
        public_cards = torch.rand(1, 10) * 0.8  # Normalized card values

        # Hole cards: 2 cards * 2 features
        hole_cards = torch.rand(1, 4) * 0.8

        # Action history: variable length sequence
        seq_len = np.random.randint(10, 48)
        action_history = torch.zeros(1, 48, 2)
        action_history[0, :seq_len, 0] = torch.rand(seq_len)  # is_target_player
        action_history[0, :seq_len, 1] = torch.rand(seq_len)  # action_type

        # Get style predictions
        style_pred = osm_model(public_cards, hole_cards, action_history)
        style_pred = style_pred.squeeze().numpy()

        predictions.append(style_pred)

        # Track feature ranges
        feature_ranges['vpip'].append(style_pred[0])
        feature_ranges['pfr'].append(style_pred[1])
        feature_ranges['afq'].append(style_pred[2])
        feature_ranges['wtsd'].append(style_pred[3])

predictions = np.array(predictions)

# Calculate statistics
print(f"\n{'='*70}")
print("OSM PREDICTION ANALYSIS")
print(f"{'='*70}\n")

feature_names = ['VPIP', 'PFR', 'AFq', 'WTSD']
feature_descriptions = [
    'Voluntarily Put $ In Pot',
    'Pre-Flop Raise',
    'Aggression Frequency',
    'Went To ShowDown'
]

print("Style Feature Statistics:")
print(f"{'Feature':<8} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8} {'Range':<8}")
print("-" * 70)

for i, (name, desc) in enumerate(zip(feature_names, feature_descriptions)):
    values = feature_ranges[name.lower()]
    mean = np.mean(values)
    std = np.std(values)
    min_val = np.min(values)
    max_val = np.max(values)
    range_val = max_val - min_val

    print(f"{name:<8} {mean:>7.3f} {std:>7.3f} {min_val:>7.3f} {max_val:>7.3f} {range_val:>7.3f}")

# Evaluate prediction quality
print(f"\n{'='*70}")
print("QUALITY METRICS")
print(f"{'='*70}\n")

# 1. Prediction diversity (should not always predict same values)
avg_std = np.mean([np.std(feature_ranges[f]) for f in ['vpip', 'pfr', 'afq', 'wtsd']])
print(f"Average Feature Std Dev: {avg_std:.3f}")
if avg_std > 0.15:
    print("  ✓ GOOD: Predictions are diverse (not stuck on one value)")
elif avg_std > 0.08:
    print("  ⚠️  MODERATE: Some diversity, could be better")
else:
    print("  ❌ LOW: Predictions lack diversity")

# 2. Output range utilization (should use full [0,1] range)
avg_range = np.mean([max(feature_ranges[f]) - min(feature_ranges[f])
                     for f in ['vpip', 'pfr', 'afq', 'wtsd']])
print(f"\nAverage Feature Range: {avg_range:.3f}")
if avg_range > 0.6:
    print("  ✓ EXCELLENT: Uses wide range of values")
elif avg_range > 0.4:
    print("  ✓ GOOD: Uses reasonable range")
elif avg_range > 0.2:
    print("  ⚠️  MODERATE: Limited range usage")
else:
    print("  ❌ LOW: Very narrow predictions")

# 3. Realistic poker statistics
# Typical ranges: VPIP (15-50%), PFR (10-30%), AFq (30-60%), WTSD (15-35%)
realistic_vpip = 0.15 < np.mean(feature_ranges['vpip']) < 0.50
realistic_pfr = 0.10 < np.mean(feature_ranges['pfr']) < 0.30
realistic_afq = 0.30 < np.mean(feature_ranges['afq']) < 0.60
realistic_wtsd = 0.15 < np.mean(feature_ranges['wtsd']) < 0.35

realism_score = sum([realistic_vpip, realistic_pfr, realistic_afq, realistic_wtsd])

print(f"\nRealistic Value Ranges:")
print(f"  VPIP in [15-50%]: {('✓' if realistic_vpip else '✗')} (mean={np.mean(feature_ranges['vpip'])*100:.1f}%)")
print(f"  PFR in [10-30%]:  {('✓' if realistic_pfr else '✗')} (mean={np.mean(feature_ranges['pfr'])*100:.1f}%)")
print(f"  AFq in [30-60%]:  {('✓' if realistic_afq else '✗')} (mean={np.mean(feature_ranges['afq'])*100:.1f}%)")
print(f"  WTSD in [15-35%]: {('✓' if realistic_wtsd else '✗')} (mean={np.mean(feature_ranges['wtsd'])*100:.1f}%)")
print(f"\nRealism Score: {realism_score}/4")

# 4. Feature correlation (some features should correlate)
# VPIP and PFR should correlate (aggressive players raise more)
vpip_pfr_corr = np.corrcoef(feature_ranges['vpip'], feature_ranges['pfr'])[0, 1]
print(f"\nFeature Correlations:")
print(f"  VPIP-PFR correlation: {vpip_pfr_corr:.3f}")
if vpip_pfr_corr > 0.5:
    print("    ✓ GOOD: Strong positive correlation (expected)")
elif vpip_pfr_corr > 0.2:
    print("    ⚠️  MODERATE: Weak correlation")
else:
    print("    ❌ LOW: No correlation (unexpected)")

# Overall quality score
diversity_score = min(avg_std / 0.2, 1.0) * 25  # 25 points max
range_score = min(avg_range / 0.8, 1.0) * 25     # 25 points max
realism_score = (realism_score / 4) * 25         # 25 points max
correlation_score = min(abs(vpip_pfr_corr) / 0.7, 1.0) * 25  # 25 points max

overall_score = diversity_score + range_score + realism_score + correlation_score

print(f"\n{'='*70}")
print("OVERALL OSM QUALITY SCORE")
print(f"{'='*70}")
print(f"  Diversity:   {diversity_score:>5.1f}/25")
print(f"  Range Usage: {range_score:>5.1f}/25")
print(f"  Realism:     {realism_score:>5.1f}/25")
print(f"  Correlation: {correlation_score:>5.1f}/25")
print(f"  " + "-" * 20)
print(f"  TOTAL:       {overall_score:>5.1f}/100")

if overall_score >= 75:
    status = "✅ EXCELLENT"
elif overall_score >= 60:
    status = "✅ GOOD"
elif overall_score >= 45:
    status = "⚠️  MODERATE"
else:
    status = "❌ NEEDS IMPROVEMENT"

print(f"\n  Status: {status}")

print(f"\n{'='*70}")
print("CONCLUSION")
print(f"{'='*70}")
print("OSM Model Capabilities:")
print("  • Predicts 4 opponent style features (VPIP, PFR, AFq, WTSD)")
print("  • Processes game history with 3-layer LSTM")
print("  • Outputs values in [0, 1] range")
print(f"  • Quality Score: {overall_score:.1f}/100")
print("\nOSM Purpose:")
print("  • Feeds style predictions to AMP3 Actor network")
print("  • Enables adaptive policy based on opponent type")
print("  • Core component of the AMP3 system")
print(f"\n{'='*70}\n")

# Save results
results = {
    'overall_score': overall_score,
    'diversity': avg_std,
    'range': avg_range,
    'realism_score': realism_score,
    'vpip_pfr_correlation': vpip_pfr_corr,
    'feature_means': {
        'vpip': np.mean(feature_ranges['vpip']),
        'pfr': np.mean(feature_ranges['pfr']),
        'afq': np.mean(feature_ranges['afq']),
        'wtsd': np.mean(feature_ranges['wtsd'])
    }
}

torch.save(results, 'osm_test_results.pt')
print("✓ Results saved to: osm_test_results.pt\n")
