# AMP3 Live Demo Instructions

## Quick Start

### Option 1: Quick Demo (30 seconds)
```bash
cd /Users/ardaenfiyeci/Downloads/amp3_full
python3 demo_amp3_live.py
```

### Option 2: Full Demo (1 minute)
```bash
python3 demo_amp3_live.py full
```

### Option 3: Extended Demo (2 minutes)
```bash
python3 demo_amp3_live.py extended
```

## What the Demo Shows

The demo will:
1. Load the AMP3 model (100k episodes trained)
2. Present realistic poker scenarios
3. Show AMP3's decision-making process
4. Display action probabilities and confidence levels
5. Demonstrate opponent modeling capabilities

## Example Output

```
======================================================================
     AMP3 POKER AI - LIVE DECISION MAKING DEMO
     Reinforcement Learning Agent with Opponent Modeling
======================================================================

Loading AMP3 model...
✓ Model loaded from episode 100,000
✓ Parameters: 1,028,068
✓ Training progress: 83.3%

SCENARIO 1/3
──────────────────────────────────────────────────────────────────────

GAME SITUATION:
  Position:    Button
  Street:      Flop
  Hole Cards:  A♠ K♥
  Pot Size:    $800
  Your Stack:  $10000
  To Call:     $200

  → AMP3 analyzing situation...
  → Processing opponent tendencies...
  → Calculating optimal strategy...

AMP3 DECISION:
  Recommended Action: RAISE SMALL
  Confidence: 67.3%

  Action Probabilities:
    FOLD        :  8.2% ███
    CALL        : 15.4% ██████
    RAISE SMALL : 67.3% ███████████████████████████ ←
    RAISE LARGE :  9.1% ████
```

## Presentation Tips

1. **Before Demo**:
   - Open terminal
   - Navigate to project directory
   - Test the demo once to ensure it works

2. **During Demo**:
   - Explain that AMP3 was trained through self-play (no human data)
   - Point out the confidence levels and probability distributions
   - Mention the 2M+ parameters and 100k+ training episodes

3. **Key Points to Highlight**:
   - ✅ Unified model covering all game streets
   - ✅ Opponent modeling integration
   - ✅ Adaptive decision making
   - ✅ Trained via reinforcement learning (not supervised)

## Troubleshooting

If you get an error:
```bash
# Make sure you're in the right directory
cd /Users/ardaenfiyeci/Downloads/amp3_full

# Check the checkpoint exists
ls -lh checkpoints_improved/amp3_checkpoint_100000.pt

# Try running with Python 3 explicitly
python3 demo_amp3_live.py
```

## Customization

You can modify the demo by editing `demo_amp3_live.py`:
- Change `num_scenarios` for more/fewer examples
- Adjust `delay` for faster/slower pacing
- Customize the poker situations in `simulate_game_situation()`
