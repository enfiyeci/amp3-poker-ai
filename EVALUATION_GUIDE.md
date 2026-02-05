# AMP3 Poker AI - Evaluation Guide

This guide explains how to evaluate your trained AMP3 models using comprehensive metrics.

---

## Quick Start

### Evaluate Preflop Model

```bash
bash run_evaluation.sh checkpoints/best_model.pt
```

### Evaluate Full AMP3 Model (after all training completes)

```bash
bash run_evaluation.sh checkpoints_20hr/amp3_actor_best.pt
```

---

## Evaluation Metrics

### 1. Classification Metrics

- **Accuracy**: Overall prediction accuracy on validation set
- **Precision/Recall/F1**: Per-action performance metrics
- **Confusion Matrix**: Shows which actions are confused with each other
- **Per-Action Confidence**: Average confidence for each action type

**What to look for:**
- Accuracy > 75%: Excellent
- Accuracy 65-75%: Good
- Accuracy 55-65%: Acceptable
- Accuracy < 55%: Needs improvement

### 2. Head-to-Head Performance

Tests your AI against baseline strategies:
- **Tight-Passive**: Conservative play style
- **Loose-Aggressive**: Aggressive play style
- **Sklansky Regular**: Based on Sklansky hand rankings
- **Chen Regular**: Based on Chen formula
- **Rule-Based Regular**: Simple heuristic strategy

**Metrics:**
- **Win Rate**: % of hands won
- **BB/100**: Big blinds won per 100 hands (positive is good)
- **VPIP**: Voluntarily Put $ In Pot (normal: 15-30%)
- **AFq**: Aggression Factor (normal: 1.5-3.0)

**What to look for:**
- Win Rate > 50%: Good
- BB/100 > 0: Profitable
- BB/100 > 5: Excellent

### 3. Expected Value (EV) Analysis

Calculates the expected value of each action type:
- **FOLD**: Should be negative (opportunity cost)
- **CALL**: Should be near zero or slightly positive
- **RAISE_SMALL**: Should be positive
- **RAISE_LARGE**: Should be positive but more variance

**What to look for:**
- Raises have higher EV than calls
- Model chooses high-EV actions frequently

### 4. Exploitability Analysis

Measures how vulnerable the AI is to counter-strategies:
- **Worst-Case BB/100**: Performance vs best counter-strategy
- **Exploitability Score**: Lower is better (< 5 = excellent, < 15 = good)

**What to look for:**
- Exploitability Score < 10: Robust strategy
- Worst-case BB/100 > -20: Hard to exploit

---

## Running Manual Evaluation

### Python Script

```bash
python3 evaluate_poker_ai.py \
    --model_path checkpoints/best_model.pt \
    --data_path /Users/ardaenfiyeci/Downloads/preflop_demo_full.csv \
    --num_sim_hands 3000 \
    --device cpu
```

### Parameters

- `--model_path`: Path to trained model checkpoint
- `--data_path`: Path to CSV validation data
- `--num_sim_hands`: Number of hands to simulate (default: 1000)
- `--device`: cpu or cuda (default: cpu)

---

## Interpreting Results

### Example Output

```
============================================================
1. CLASSIFICATION METRICS
============================================================

Overall Accuracy: 0.792

Per-Action Performance:
------------------------------------------------------------
FOLD         | Samples:  15234 | Accuracy: 0.812 | Confidence: 0.856
CALL         | Samples:  23455 | Accuracy: 0.789 | Confidence: 0.742
RAISE_SMALL  | Samples:  12876 | Accuracy: 0.756 | Confidence: 0.698
RAISE_LARGE  | Samples:   8765 | Accuracy: 0.801 | Confidence: 0.823

============================================================
2. HEAD-TO-HEAD PERFORMANCE
============================================================

Results:
--------------------------------------------------------------------------------
Opponent             |  Win Rate |     BB/100 |     VPIP |      AFq
--------------------------------------------------------------------------------
tight_passive        |      52.3% |        6.2 |    24.1% |     2.15
loose_aggressive     |      48.7% |        1.3 |    26.8% |     2.42
sklansky_regular     |      51.1% |        4.5 |    22.3% |     1.98
chen_regular         |      50.8% |        3.8 |    23.5% |     2.05
rule_based_regular   |      49.2% |        2.1 |    25.2% |     2.18

Average Performance:
  Win Rate: 50.4%
  BB/100:   3.6

============================================================
3. EXPECTED VALUE (EV) ANALYSIS
============================================================

Expected Value by Action:
------------------------------------------------------------
Action          | Mean EV (chips) |   EV (BB) |   Std Dev |  Count
------------------------------------------------------------
FOLD            |           -45.2 |      -0.45 |      125.3 |    234
CALL            |            12.5 |       0.13 |      187.6 |    456
RAISE_SMALL     |            78.3 |       0.78 |      245.1 |    189
RAISE_LARGE     |           142.7 |       1.43 |      412.8 |     121

============================================================
4. EXPLOITABILITY ANALYSIS
============================================================

Performance vs Exploitative Opponents:
------------------------------------------------------------
  tight_passive        | BB/100:     5.23
  loose_aggressive     | BB/100:     1.45
  very_aggressive      | BB/100:    -2.15

Worst-Case BB/100:            -2.15
Avg vs Exploiters:             1.51
Exploitability Score:          2.15

✅ Low exploitability - robust strategy

============================================================
SUMMARY
============================================================

✓ Classification Accuracy:  79.2%
✓ Average Win Rate:         50.4%
✓ Average BB/100:           +3.6
✓ Exploitability Score:     2.15

🎉 Model Performance: EXCELLENT
```

---

## Performance Benchmarks

### Preflop Model (baseline)
- Classification: 75-80%
- Win Rate: 48-52%
- BB/100: 0-5
- Exploitability: 5-15

### Full AMP3 Model (with RL)
- Classification: 80-85%
- Win Rate: 52-58%
- BB/100: 5-15
- Exploitability: 2-8

---

## Troubleshooting

### Error: "Model file not found"
Make sure you're pointing to the correct checkpoint:
```bash
ls -lh checkpoints/*.pt
ls -lh checkpoints_20hr/*.pt
```

### Error: "Data file not found"
Verify the CSV path:
```bash
ls -lh /Users/ardaenfiyeci/Downloads/preflop_demo_full.csv
```

### Low Performance
- Check training logs for convergence
- Verify sufficient training epochs
- Ensure validation data is representative
- Try training longer or with more data

---

## Next Steps

After evaluation, you can:

1. **If results are good (EXCELLENT/GOOD)**:
   - Deploy the model for play testing
   - Fine-tune with more data
   - Run tournaments against other AIs

2. **If results are acceptable**:
   - Continue training for more epochs
   - Adjust hyperparameters
   - Add more training data

3. **If results need improvement**:
   - Review training logs for issues
   - Check data quality
   - Adjust model architecture
   - Increase training time

---

## Advanced: Comparing Multiple Models

Create a comparison script:

```bash
#!/bin/bash
for model in checkpoints/*.pt checkpoints_20hr/*.pt; do
    echo "Evaluating $model..."
    python3 evaluate_poker_ai.py --model_path "$model" --num_sim_hands 1000
done
```

This helps you identify which training checkpoint performs best.
