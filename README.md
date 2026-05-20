# AMP3 Poker AI

Implementation of the AMP3 learning method for 6-player No-Limit Texas Hold'em, combining LSTM-based opponent style modeling with actor-critic reinforcement learning, Deep CFR, and Neural Fictitious Self-Play. Built as a senior design project at the University of Pennsylvania.

Based on Yang et al., "AMP3: An Adaptive Multi-player Poker Policy Learning Method Based on Opponent Style Modeling," *Neural Computing and Applications*, 2025 ([DOI 10.1007/s00521-025-11262-x](https://doi.org/10.1007/s00521-025-11262-x)).

## Architecture

```
                  +-----------------------+
   action history | OSM Network (LSTM)    |  predicts opponent style
   public cards   |  -> VPIP, PFR,        |  (4 behavioral features)
                  |     AFq, WTSD         |
                  +-----------+-----------+
                              |
                              v
                  +-----------------------+
   game state     | AMP3 Actor-Critic     |  action policy adapted to
   opponent style |  Actor: policy head   |  modeled opponent style
                  |  Critic: Q value      |
                  +-----------+-----------+
                              |
                              v
                  +-----------------------+
                  | Auxiliary heads       |
                  |  preflop imitation    |
                  |  Deep CFR, NFSP       |
                  +-----------------------+
```

## File layout

| File | Role |
|------|------|
| `poker_core.py` | Card and hand evaluation, Sklansky groups, Chen formula |
| `poker_env.py` | 6-player NLHE environment |
| `style_library.py` | 64 player strategies (Random, Sklansky, Chen, RuleBased) |
| `osm_network.py` | LSTM opponent style modeling network |
| `amp3_network.py` | Actor-critic with style-conditioned policy |
| `cfr_networks.py` | Deep CFR and NFSP implementations |
| `preflop_imitation.py` | Preflop imitation plus flop/turn/river models |
| `train_amp3.py` | End-to-end training pipeline |
| `demo_amp3_*.py` | Live and simulated play demos |

## Quick start

```bash
pip install -r requirements.txt

# Full pipeline
python train_amp3.py --stage all --save_dir ./checkpoints

# Or one stage at a time: osm, preflop, streets, amp3, cfr, nfsp
python train_amp3.py --stage osm --save_dir ./checkpoints
```

Full pipeline takes roughly 24 to 48 hours on a single GPU.

## Style features

The OSM network estimates four standard poker telemetry features for each opponent:

| Feature | Meaning |
|---------|---------|
| VPIP | Voluntarily Put money In Pot |
| PFR | Pre-Flop Raise rate |
| AFq | Post-flop aggression frequency |
| WTSD | Went To ShowDown rate |

## License

Research and educational use.
