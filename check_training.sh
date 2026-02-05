#!/bin/bash
# Quick training status checker

echo "======================================"
echo "AMP3 Training Status"
echo "======================================"
echo ""

# Check if training is running
RUNNING=$(ps aux | grep "train_amp3.py" | grep -v grep | wc -l)
if [ $RUNNING -gt 0 ]; then
    echo "🟢 Training is RUNNING"
    echo ""
    echo "Process details:"
    ps aux | grep "train_amp3.py" | grep -v grep | awk '{printf "  PID: %s | CPU: %s%% | Memory: %s | Runtime: %s\n", $2, $3, $6/1024 "MB", $10}'
    echo ""
else
    echo "🔴 No training process running"
    echo ""
fi

# Check latest files
echo "Latest checkpoint files:"
ls -lht checkpoints_20hr/*.pt 2>/dev/null | head -5 || echo "  No .pt files yet"
echo ""

# Check dataset files
echo "Dataset files:"
ls -lh checkpoints_20hr/*.npy checkpoints_20hr/*.json 2>/dev/null | tail -5
echo ""

# Estimate progress
if [ -f "checkpoints_20hr/osm_dataset.pt.npy" ]; then
    echo "✅ OSM dataset generated (339 MB)"
fi

if [ -f "checkpoints_20hr/osm_best.pt" ]; then
    echo "✅ OSM training complete"
fi

if [ -f "checkpoints_20hr/flop_best.pt" ]; then
    echo "✅ Flop model complete"
fi

if [ -f "checkpoints_20hr/turn_best.pt" ]; then
    echo "✅ Turn model complete"
fi

if [ -f "checkpoints_20hr/river_best.pt" ]; then
    echo "✅ River model complete"
fi

if [ -f "checkpoints_20hr/amp3_actor_best.pt" ]; then
    echo "✅ AMP3 Actor complete"
fi

echo ""
echo "======================================"
