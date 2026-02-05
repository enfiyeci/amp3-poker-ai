# AMP3 Demo - Final Version

## Quick Start

```bash
cd /Users/ardaenfiyeci/Downloads/amp3_full
python3 demo_amp3_simulated.py
```

## What You'll See

Natural, conversational output showing:

```
Hand 1/3
──────────────────────────────────────────────────────────────────────

River - Button
Cards: K♥ Q♣
Pot: $1200 | Stack: $10000 | To call: $200

Opponent: Tight-Passive
Action: CALL (36.8%)

Thinking:
  CALL          36.8% ██████████████ ←
  RAISE SMALL   36.3% ██████████████
  RAISE LARGE   26.8% ██████████
```

## Key Features

✅ **Different every time** - Random cards and situations each run
✅ **Opponent modeling** - Shows 6 different player types
✅ **Natural language** - Less AI-sounding, more conversational
✅ **Clean output** - Only shows relevant probabilities (>5%)
✅ **Professional** - Ready for presentations

## Three Modes

- `python3 demo_amp3_simulated.py` - Quick (30s, 3 hands)
- `python3 demo_amp3_simulated.py full` - Full (1m, 5 hands)
- `python3 demo_amp3_simulated.py extended` - Extended (2m, 10 hands)

## Talking Points

1. **"AMP3 learned entirely from playing against itself"**
   - 120,000 hands of self-play
   - No human expert data
   - Discovered strategies through trial and error

2. **"It reads opponents in real-time"**
   - Classifies into 6 playing styles
   - Adapts strategy accordingly
   - Shows opponent type before each decision

3. **"You can see how it thinks"**
   - Probability distribution across all actions
   - Confidence levels on decisions
   - Strategic reasoning visible

4. **"One model handles everything"**
   - Preflop through river
   - All positions
   - All opponent types
   - 2 million parameters

## What Changed (Less AI-Sounding)

**Before:**
```
GAME SITUATION:
  Position:    Button
  Street:      Flop

→ AMP3 analyzing situation...
→ Processing opponent tendencies...
→ Opponent Style Detected: Tight-Aggressive
→ Calculating optimal strategy...

AMP3 DECISION:
  Recommended Action: RAISE SMALL
  Confidence: 67.3%
```

**After:**
```
Flop - Button
Cards: A♠ K♥
Pot: $800 | Stack: $10000 | To call: $200

Opponent: Tight-Aggressive
Action: RAISE SMALL (67.3%)
```

Much cleaner and more natural!
