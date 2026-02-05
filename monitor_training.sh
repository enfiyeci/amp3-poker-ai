#!/bin/bash
# Monitor AMP3 Training Progress

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║              AMP3 TRAINING MONITOR                               ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Check if training is running
echo "🔍 Training Process Status:"
ps aux | grep "train_amp3.py --stage amp3" | grep -v grep | awk '{printf "  PID: %s | CPU: %s%% | Memory: %s MB | Runtime: %s\n", $2, $3, int($6/1024), $10}'

if [ $? -ne 0 ]; then
    echo "  ❌ No training process found"
    exit 1
fi

echo ""
echo "📊 Latest Checkpoints:"
ls -lht checkpoints_improved/amp3_checkpoint_*.pt 2>/dev/null | head -5 | awk '{printf "  %s - %s\n", $9, $5}'

if [ $? -ne 0 ]; then
    echo "  ⏳ No checkpoints yet (training just started)"
fi

echo ""
echo "📈 Training Progress:"

# Count total episodes trained
LATEST_CHECKPOINT=$(ls -t checkpoints_improved/amp3_checkpoint_*.pt 2>/dev/null | head -1)
if [ ! -z "$LATEST_CHECKPOINT" ]; then
    EPISODE=$(echo $LATEST_CHECKPOINT | grep -o '[0-9]*' | tail -1)
    PROGRESS=$(echo "scale=1; $EPISODE / 120000 * 100" | bc)
    echo "  Current Episode: $EPISODE / 120,000"
    echo "  Progress: $PROGRESS%"
    echo "  Remaining: $((120000 - EPISODE)) episodes"
else
    echo "  Starting from: 40,000 episodes"
    echo "  Target: 120,000 episodes"
    echo "  Remaining: 80,000 episodes"
fi

echo ""
echo "📝 Recent Log Output:"
tail -10 logs/amp3_training.log 2>/dev/null || echo "  (Log file not found or empty)"

echo ""
echo "💾 Disk Usage:"
du -sh checkpoints_improved 2>/dev/null || echo "  0 MB (no checkpoints yet)"

echo ""
echo "═════════════════════════════════════════════════════════════════"
echo "Commands:"
echo "  Monitor continuously: watch -n 30 ./monitor_training.sh"
echo "  View full log: tail -f logs/amp3_training.log"
echo "  Stop training: pkill -f 'train_amp3.py --stage amp3'"
echo "═════════════════════════════════════════════════════════════════"
