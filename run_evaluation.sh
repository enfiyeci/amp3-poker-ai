#!/bin/bash
# Run comprehensive evaluation on trained AMP3 models

echo "================================================================"
echo "  AMP3 Poker AI - Comprehensive Evaluation"
echo "================================================================"
echo ""

# Set paths
MODEL_PATH="${1:-checkpoints/best_model.pt}"
DATA_PATH="${2:-/Users/ardaenfiyeci/Downloads/preflop_demo_full.csv}"
OUTPUT_DIR="evaluation_results"

mkdir -p "$OUTPUT_DIR"

# Run evaluation
echo "Running evaluation on model: $MODEL_PATH"
echo "Using validation data: $DATA_PATH"
echo "Output directory: $OUTPUT_DIR"
echo ""

python3 evaluate_poker_ai.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --num_sim_hands 3000 \
    --device cpu \
    2>&1 | tee "$OUTPUT_DIR/evaluation_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "================================================================"
echo "  Evaluation Complete!"
echo "================================================================"
echo "Results saved to: $OUTPUT_DIR/"
