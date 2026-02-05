#!/bin/bash
# 20-Hour Optimized Training Plan for AMP3

echo "=========================================="
echo "AMP3 20-Hour Training Plan"
echo "=========================================="
echo ""

# Create checkpoint directory
mkdir -p checkpoints_20hr

echo "Phase 1: OSM Training (3-4 hours)"
echo "Start time: $(date)"
echo "------------------------------------------"
python3 train_amp3.py --stage osm_training --config-override "osm_num_games=5000"
echo "OSM Complete: $(date)"
echo ""

echo "Phase 2: Later-Street Models (12-15 hours, PARALLEL)"
echo "Start time: $(date)"
echo "------------------------------------------"
echo "Starting Flop, Turn, River training in parallel..."

# Run all three in background
python3 train_amp3.py --stage flop_model --config-override "street_num_samples=50000" > checkpoints_20hr/flop.log 2>&1 &
FLOP_PID=$!

python3 train_amp3.py --stage turn_model --config-override "street_num_samples=50000" > checkpoints_20hr/turn.log 2>&1 &
TURN_PID=$!

python3 train_amp3.py --stage river_model --config-override "street_num_samples=50000" > checkpoints_20hr/river.log 2>&1 &
RIVER_PID=$!

echo "Flop PID: $FLOP_PID"
echo "Turn PID: $TURN_PID"
echo "River PID: $RIVER_PID"
echo ""
echo "Monitoring progress (check logs in checkpoints_20hr/)..."

# Wait for all to complete
wait $FLOP_PID
echo "Flop training complete!"

wait $TURN_PID
echo "Turn training complete!"

wait $RIVER_PID
echo "River training complete!"

echo "All later-street models complete: $(date)"
echo ""

echo "Phase 3: AMP3 RL Training (3-4 hours)"
echo "Start time: $(date)"
echo "------------------------------------------"
python3 train_amp3.py --stage amp3_rl --config-override "amp3_episodes=20000,amp3_actor_lr=5e-4,amp3_critic_lr=5e-4"
echo "AMP3 Complete: $(date)"
echo ""

echo "=========================================="
echo "Training Complete!"
echo "End time: $(date)"
echo "=========================================="
echo ""
echo "Trained models:"
ls -lh checkpoints_20hr/*.pt
