# Quick Start: Training Guide

## 🎯 Current Status

| Model | Status | Next Step |
|-------|--------|-----------|
| Preflop | ✅ 79.2% accuracy | Deploy (ready) |
| Flop/Turn/River | ✅ Trained & tested | Deploy (ready) |
| OSM | ⚠️ 22.6 quality (low diversity) | **Retrain (recommended)** |
| AMP3 | 🔄 40k/120k episodes (33%) | **Continue training** |

---

## 🚀 Option 1: Quick Continue (Recommended)

Continue AMP3 training from where it left off (fastest path to completion):

```bash
# Continue AMP3 from 40k → 120k episodes (~8 hours)
python3 train_amp3.py --stage amp3 \
  --amp3_episodes 120000 \
  --amp3_resume checkpoints_20hr/amp3_checkpoint_40000.pt \
  > logs/amp3_continue.log 2>&1 &

# Monitor progress
tail -f logs/amp3_continue.log
```

**Duration**: ~8 hours
**Result**: Complete AMP3 model (120k episodes)

---

## 🔧 Option 2: Fix OSM Then Continue AMP3 (Best Quality)

Retrain OSM with better diversity, then continue AMP3:

### Step 1: Retrain OSM (~24 hours)

```bash
# Retrain OSM with 10x more data and better diversity
python3 train_amp3.py --stage osm \
  --osm_num_games 1000000 \
  --osm_epochs 200 \
  --osm_batch_size 256 \
  --osm_lr 0.0005 \
  > logs/osm_improved.log 2>&1 &

# Monitor OSM training
tail -f logs/osm_improved.log
```

**Wait for OSM to complete**, then:

### Step 2: Retrain AMP3 from Scratch (~12 hours)

```bash
# Start fresh AMP3 with improved OSM
python3 train_amp3.py --stage amp3 \
  --amp3_episodes 120000 \
  > logs/amp3_fresh.log 2>&1 &

# Monitor training
tail -f logs/amp3_fresh.log
```

**Total Duration**: ~36 hours
**Result**: Best quality AMP3 with proper opponent adaptation

---

## 🎮 Option 3: Interactive Training Script

Use the interactive script (easiest):

```bash
python3 train_improved.py
```

This will ask you:
1. Retrain OSM?
2. Continue AMP3?
3. Both?

And start training based on your choice.

---

## 📊 Monitoring Training

### Check if Training is Running

```bash
# Check for active training
ps aux | grep train_amp3.py
```

### Monitor Progress

```bash
# AMP3 progress
tail -f logs/amp3_continue.log

# OSM progress
tail -f logs/osm_improved.log

# Check checkpoints being saved
ls -lht checkpoints_improved/ | head -10
```

### Stop Training

```bash
# Stop all training
pkill -f train_amp3.py

# Stop specific training
pkill -f "train_amp3.py --stage amp3"
```

---

## 🎯 What Each Training Does

### AMP3 Training (Continue)
- **From**: 40,000 episodes
- **To**: 120,000 episodes
- **Adds**: 80,000 more self-play games
- **Duration**: ~8 hours
- **Improves**: Win rate, strategy adaptation
- **File**: `checkpoints_improved/amp3_final.pt`

### OSM Training (Retrain)
- **Games**: 1,000,000 (was 100,000)
- **Epochs**: 200 (was 100)
- **Duration**: ~24 hours
- **Fixes**: Low diversity issue (22.6 → 70+ quality)
- **Improves**: Opponent predictions vary properly
- **File**: `checkpoints_improved/osm/osm_best.pt`

---

## 📈 Expected Results

### After Continuing AMP3 (Option 1)
✅ AMP3 complete (120k episodes)
⚠️ OSM still has low diversity
→ **Use for deployment, but limited adaptation**

### After Retraining OSM + AMP3 (Option 2)
✅ OSM high diversity (70+ quality)
✅ AMP3 complete with proper adaptation
→ **Best quality, full opponent adaptation**

---

## 🔍 How to Verify Success

### OSM Diversity Check
```bash
# After OSM retraining completes
python3 test_osm.py

# Look for:
# - Diversity: >15/25 (was 0.3/25)
# - Range: >40/25 (was 0.6/25)
# - Overall: >70/100 (was 22.6/100)
```

### AMP3 Progress Check
```bash
# Check latest checkpoint
ls -lh checkpoints_improved/amp3_checkpoint_*.pt | tail -1

# Should see checkpoints at 50k, 60k, 70k, ..., 120k
```

---

## 💾 Disk Space Requirements

- **AMP3 Continue**: ~10 MB (8 checkpoints × 7.8 MB)
- **OSM Retrain**: ~50 MB (checkpoints + logs)
- **Both**: ~60 MB total

Ensure you have at least **100 MB free**.

---

## ⚡ Quick Commands Reference

```bash
# Continue AMP3 (8 hours)
python3 train_amp3.py --stage amp3 --amp3_episodes 120000 \
  --amp3_resume checkpoints_20hr/amp3_checkpoint_40000.pt > logs/amp3.log 2>&1 &

# Retrain OSM (24 hours)
python3 train_amp3.py --stage osm --osm_num_games 1000000 \
  --osm_epochs 200 > logs/osm.log 2>&1 &

# Monitor
tail -f logs/amp3.log
tail -f logs/osm.log

# Stop
pkill -f train_amp3.py

# Check progress
ls -lht checkpoints_improved/ | head
```

---

## 🎯 Recommendation

**For Fastest Results**: Use **Option 1** (continue AMP3)
**For Best Quality**: Use **Option 2** (fix OSM first)
**For Ease**: Use **Option 3** (interactive script)

I recommend **Option 2** if you have time - the OSM diversity issue is significant and fixing it will give you a much better final model.

---

## 📞 Need Help?

Check training logs for errors:
```bash
tail -100 logs/amp3_continue.log
tail -100 logs/osm_improved.log
```

Common issues:
- "Out of memory" → Reduce batch size in command
- "Checkpoint not found" → Check path is correct
- Training seems stuck → Check if process is running: `ps aux | grep train_amp3`
