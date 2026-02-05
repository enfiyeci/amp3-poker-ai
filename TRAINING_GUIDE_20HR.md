# 20-Hour AMP3 Training Guide

## Overview
This guide trains all essential AMP3 components in 20 hours:
- ✅ Preflop Imitation: Already trained (79.2% accuracy)
- 🎯 OSM (Opponent Modeling): 3-4 hours
- 🎯 Later-Street Models: 12-15 hours (parallel)
- 🎯 AMP3 Actor-Critic: 3-4 hours

**Total: ~18-20 hours**

---

## Quick Start

### Option 1: Automated (Recommended)
```bash
cd /Users/ardaenfiyeci/Downloads/amp3_full
python3 train_20hr.py
```

This will automatically run all phases and monitor progress.

---

### Option 2: Manual Control

#### **Phase 1: OSM Training** (~3 hours)

Open terminal and run:
```bash
cd /Users/ardaenfiyeci/Downloads/amp3_full
python3 train_amp3.py --stage osm_training
```

Watch for completion message. Model saves to `amp3_checkpoints/osm_network.pt`

---

#### **Phase 2: Later-Street Models** (~12-15 hours, parallel)

Open **THREE separate terminals** and run one command in each:

**Terminal 1 (Flop):**
```bash
cd /Users/ardaenfiyeci/Downloads/amp3_full
python3 train_amp3.py --stage flop_model | tee logs/flop.log
```

**Terminal 2 (Turn):**
```bash
cd /Users/ardaenfiyeci/Downloads/amp3_full
python3 train_amp3.py --stage turn_model | tee logs/turn.log
```

**Terminal 3 (River):**
```bash
cd /Users/ardaenfiyeci/Downloads/amp3_full
python3 train_amp3.py --stage river_model | tee logs/river.log
```

All three will run simultaneously. Wait for all to finish before proceeding.

**Models save to:**
- `amp3_checkpoints/flop_network.pt`
- `amp3_checkpoints/turn_network.pt`
- `amp3_checkpoints/river_network.pt`

---

#### **Phase 3: AMP3 Actor-Critic** (~3 hours)

After Phase 2 completes, run:
```bash
cd /Users/ardaenfiyeci/Downloads/amp3_full
python3 train_amp3.py --stage amp3_rl
```

Model saves to `amp3_checkpoints/amp3_actor.pt` and `amp3_checkpoints/amp3_critic.pt`

---

## Monitoring Progress

### Check training progress:
```bash
# OSM training
tail -f amp3_checkpoints/osm_training.log

# Later-street training (in separate terminals)
tail -f logs/flop.log
tail -f logs/turn.log
tail -f logs/river.log

# AMP3 RL training
tail -f amp3_checkpoints/amp3_rl.log
```

### Check trained models:
```bash
ls -lh amp3_checkpoints/*.pt
```

---

## Expected Output

After 20 hours, you should have:

```
amp3_checkpoints/
├── best_model.pt           (569 KB)  - Preflop imitation ✅
├── osm_network.pt          (~2 MB)   - Opponent modeling
├── flop_network.pt         (~5 MB)   - Flop decisions
├── turn_network.pt         (~5 MB)   - Turn decisions
├── river_network.pt        (~5 MB)   - River decisions
├── amp3_actor.pt           (~3 MB)   - RL policy
└── amp3_critic.pt          (~3 MB)   - RL value network
```

**Total model size: ~23 MB**

---

## Troubleshooting

### If training is interrupted:
- Just restart the same command
- The training script checks for existing models and skips completed phases

### If you want to speed things up:
Edit `train_amp3.py` and reduce these values:
```python
'osm_num_games': 5000,         # Line 77
'street_num_samples': 50000,   # Line 92
'amp3_episodes': 20000,        # Line 98
```

### If you run out of time:
Priority order:
1. ✅ Preflop (done)
2. 🎯 OSM (3h) - Most important for adaptation
3. 🎯 Flop model (12h) - Most common post-flop street
4. Turn model (optional)
5. River model (optional)
6. AMP3 RL (optional - imitation is often sufficient)

---

## What Happens After Training?

You'll have a complete poker AI that can:
- ✅ Make preflop decisions (79% accuracy)
- ✅ Predict opponent playing styles (VPIP/PFR/etc)
- ✅ Play flop, turn, river optimally
- ✅ Adapt strategy based on opponent tendencies
- ✅ Learn from experience via RL

You can use these models in your poker application!

---

## Time Estimates by Hardware

| Component | CPU (M1/M2) | CPU (Intel) |
|-----------|-------------|-------------|
| OSM       | 2-3 hours   | 3-4 hours   |
| Flop      | 10-12 hours | 12-15 hours |
| Turn      | 10-12 hours | 12-15 hours |
| River     | 10-12 hours | 12-15 hours |
| AMP3 RL   | 3-4 hours   | 4-5 hours   |

**Your system**: Likely M-series Mac based on path
**Expected total**: ~18-20 hours with parallel training
