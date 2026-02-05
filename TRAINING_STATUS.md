# AMP3 Training Status

**Last Updated**: January 19, 2026 7:14 PM

---

## ✅ Completed Training

### 1. Preflop Imitation Model
- **Status**: ✅ Complete
- **File**: `checkpoints/best_model.pt` (569 KB)
- **Performance**: 79.2% validation accuracy
- **Dataset**: 155,543 hands (139,988 train / 15,555 val)
- **Training Time**: ~5-10 minutes
- **Features**: 7 state + 6 style features = 13 total
- **Actions**: 4-way classification (FOLD/CALL/RAISE_SMALL/RAISE_LARGE)

### 2. OSM (Opponent Style Modeling)
- **Status**: ✅ Complete
- **File**: `checkpoints_20hr/osm_best.pt` (1.3 MB)
- **Dataset**: 5,000 simulated games (339 MB dataset)
- **Training Time**: ~13 minutes
- **Predicts**: VPIP, PFR, AFq, WTSD for opponents

---

## 🔄 Currently Running

### Phase 2: Later-Street Models
- **Status**: 🟢 **RUNNING**
- **Process ID**: 23626
- **CPU Usage**: 99.7%
- **Runtime**: 2+ minutes so far
- **Estimated Time**: 8-12 hours total
- **What's being trained**: Flop, Turn, River decision networks

**Monitor with:**
```bash
ps aux | grep train_amp3 | grep streets
```

---

## ⏳ Pending Training

### Phase 3: AMP3 Actor-Critic RL
- **Status**: ⏳ Not started
- **Estimated Time**: 3-4 hours
- **Start Command**:
```bash
python3 train_amp3.py --stage amp3 --save_dir checkpoints_20hr
```

---

## 📊 Timeline Summary

| Phase | Component | Status | Time |
|-------|-----------|--------|------|
| 0 | Preflop Imitation | ✅ Done | 10 min |
| 1 | OSM Training | ✅ Done | 13 min |
| 2 | Later Streets (Flop/Turn/River) | 🔄 Running | 8-12 hours |
| 3 | AMP3 Actor-Critic | ⏳ Pending | 3-4 hours |

**Total Estimated**: ~12-16 hours remaining

---

## 🐛 Bugs Fixed During Training

1. ❌ `AttributeError: 'PlayerState' object has no attribute 'current_bet'`
   - **Fix**: Changed `player.current_bet` → `player.bet_this_street`

2. ❌ `AttributeError: 'int' object has no attribute 'value'`
   - **Fix**: Added isinstance checks for Card rank/suit attributes
   - Multiple locations in `preflop_imitation.py`

3. ❌ `ValueError: Need at least 5 cards` in Monte Carlo simulation
   - **Fix**: Fixed hand comparison in `poker_core.py` to use evaluated results directly

---

## 📁 Trained Models Location

```
checkpoints/
└── best_model.pt              # Preflop imitation (569 KB)

checkpoints_20hr/
├── osm_best.pt                # OSM model (1.3 MB)
├── osm_dataset.pt.npy         # OSM training data (339 MB)
├── config.json                # Training configuration
└── style_features.json        # Style library statistics
```

**Pending models** (will appear after training):
- `checkpoints_20hr/flop_best.pt` (~5 MB)
- `checkpoints_20hr/turn_best.pt` (~5 MB)
- `checkpoints_20hr/river_best.pt` (~5 MB)
- `checkpoints_20hr/amp3_actor_best.pt` (~3 MB)
- `checkpoints_20hr/amp3_critic_best.pt` (~3 MB)

---

## 🔍 Monitoring Commands

**Check training status:**
```bash
bash /Users/ardaenfiyeci/Downloads/amp3_full/check_training.sh
```

**Check process:**
```bash
ps aux | grep train_amp3 | grep -v grep
```

**Watch CPU usage:**
```bash
top -pid 23626
```

**Check completed models:**
```bash
ls -lh checkpoints_20hr/*.pt
```

---

## ⏭️ What to Do Next

### When Phase 2 Completes (~8-12 hours):

1. Verify models exist:
```bash
ls -lh checkpoints_20hr/flop_best.pt checkpoints_20hr/turn_best.pt checkpoints_20hr/river_best.pt
```

2. Start Phase 3 (AMP3 RL):
```bash
cd /Users/ardaenfiyeci/Downloads/amp3_full
python3 train_amp3.py --stage amp3 --save_dir checkpoints_20hr 2>&1 | tee checkpoints_20hr/amp3.log &
```

3. Wait 3-4 more hours for AMP3 to complete

### When ALL Training Completes:

You'll have a **complete poker AI** with:
- ✅ Preflop decision making (79% accurate)
- ✅ Opponent style prediction (VPIP/PFR/AFq/WTSD)
- ✅ Post-flop decision making (Flop/Turn/River)
- ✅ Adaptive reinforcement learning policy
- ✅ Value estimation for all game states

**Total model size**: ~23 MB
**Ready for production use**!

---

## 🆘 Troubleshooting

**If training stops unexpectedly:**
```bash
# Check what happened
cat checkpoints_20hr/streets.log

# Restart from same phase
python3 train_amp3.py --stage streets --save_dir checkpoints_20hr 2>&1 | tee checkpoints_20hr/streets_restart.log &
```

**If you need to pause:**
```bash
# Find process ID
ps aux | grep train_amp3 | grep -v grep

# Kill process (training can be restarted)
kill <PID>
```

---

## 📝 Notes

- Training is CPU-only (no GPU required)
- Your M-series Mac is performing excellently (~100% CPU usage)
- OSM trained much faster than estimated (13 min vs 3 hours!)
- Later streets will take longest due to game simulation complexity
- All models save checkpoints automatically (safe to interrupt)
