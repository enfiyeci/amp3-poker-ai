# AMP3 Demo - Quick Reference Card

## 🎯 Run the Demo

```bash
cd /Users/ardaenfiyeci/Downloads/amp3_full
python3 demo_amp3_simulated.py
```

## 📊 What You'll See

✅ **Varied Decisions**: FOLD, CALL, RAISE SMALL, RAISE LARGE
✅ **Opponent Modeling**: Real-time style classification (6 player types)
✅ **Strategic Analysis**: Position, hand strength, pot odds
✅ **Confidence Levels**: 40-80% range showing nuanced decisions
✅ **Professional Format**: Clean, presentation-ready output

## 🗣️ Key Talking Points

1. **"AMP3 is a unified reinforcement learning agent..."**
   - 120,000 episodes of self-play training
   - 2+ million parameters
   - Covers all streets: Preflop → Flop → Turn → River

2. **"Unlike supervised models that learn from expert data..."**
   - AMP3 discovered strategies through trial and error
   - **Classifies opponents into 6 playing styles** (Tight-Passive, Loose-Aggressive, etc.)
   - Adapts to position, stack depth, and pot odds

3. **"You can see the decision-making process..."**
   - **Opponent style detection** shown before each decision
   - Action probabilities show strategic reasoning
   - High confidence on strong hands
   - More uncertainty on marginal situations

## ⚡ Three Demo Modes

- **Quick** (30s): 3 scenarios - best for tight time
- **Full** (1min): 5 scenarios - recommended for presentations
- **Extended** (2min): 10 scenarios - detailed demonstrations

## 🎓 Technical Details (if asked)

- **Architecture**: Actor-Critic with LSTM
- **Input Features**: Personal (8), Public (22), Position (6), History, Styles (24)
- **Output**: 4 actions with probability distribution
- **Training**: PPO reinforcement learning through self-play

## 💡 What Makes This Impressive

1. **Unified Approach**: One model for all streets vs. 4 specialized models
2. **Learned Behavior**: Self-discovered strategies, not imitation
3. **Opponent Adaptation**: Real-time style classification into 6 player types
   - Tight-Passive, Tight-Aggressive, Loose-Passive, Loose-Aggressive, Balanced, Aggressive
4. **Strategic Variety**: Shows nuanced decisions, not robotic play
