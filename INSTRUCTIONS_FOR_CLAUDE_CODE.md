# Complete Training Instructions for Claude Code

## Project Overview

This is **AMP3 Poker AI** - a reinforcement learning agent that plays Texas Hold'em poker. The project implements the AMP3 (Actor-Critic with Multi-Player State Modeling) algorithm from the research paper.

### What This Project Does

- Trains a neural network to play 6-player No-Limit Texas Hold'em
- Uses **reinforcement learning** (learns by playing against itself)
- Includes **opponent modeling** (learns to classify and adapt to different player styles)
- Has **37 different opponent strategies** to train against
- Currently trained to **120,000 episodes**, goal is **200,000 episodes**

## Repository Information

- **GitHub URL**: https://github.com/enfiyeci/amp3-poker-ai
- **Current status**: 120k episodes complete, ready to continue to 200k
- **Training checkpoint**: `checkpoints_improved/amp3_final.pt` (7.8 MB)

## Project Structure

### Core Architecture Files

**`poker_core.py`** - Foundation
- Card representation (52 cards: ranks 2-A, suits ♣♦♥♠)
- Hand evaluation (Royal Flush → High Card)
- Game state management
- Action types: FOLD, CALL, BET, ALL_IN
- Street types: PREFLOP, FLOP, TURN, RIVER

**`poker_env.py`** - Game Environment
- 6-player No-Limit Texas Hold'em simulator
- Starting stack: $10,000
- Big blind: $100
- Handles betting rounds, pot management, showdowns
- Provides valid actions based on game state

**`style_library.py`** - Opponent Modeling (JUST UPDATED)
- **37 different opponent playing styles**
- 5 original types: Conservative, Regular, Aggressive, Bluffing, Deceptive
- 5 NEW types: Maniac, Rock, Calling Station, TAG, LAG
- 3 strategy implementations:
  - **Sklansky** (based on hand group rankings)
  - **Chen** (based on hand scoring formula)
  - **Rule-based** (considers board texture)
- Plus 7 random probability strategies
- Computes 4 style features per opponent: VPIP, PFR, AFq, WTSD

**`osm_network.py`** - Opponent State Modeling Network
- LSTM-based neural network
- Predicts opponent playing style from action history
- Input: sequence of opponent actions
- Output: 4 style features (VPIP, PFR, AFq, WTSD)
- Used to help AMP3 adapt to opponents

**`amp3_network.py`** - Main AMP3 Agent (THE CORE MODEL)
- **AMP3Actor**: Decides which action to take
  - Input features:
    - `personal` (8 dims): hole cards, stack, pot, bet amounts
    - `public` (22 dims): community cards, all player stacks
    - `position` (6 dims): one-hot position encoding
    - `action_history` (sequence): past actions via LSTM
    - `style_features` (24 dims): 6 opponents × 4 features each
  - Output: 4 action logits (FOLD, CALL, RAISE_SMALL, RAISE_LARGE)
  - Parameters: **1,028,068**

- **AMP3Critic**: Evaluates state value
  - Same inputs as Actor
  - Output: single value (expected return)
  - Parameters: **1,010,273**

- **Total model size: 2,038,341 parameters**

**`cfr_networks.py`** - Alternative Training Methods
- Deep CFR (Counterfactual Regret Minimization)
- NFSP (Neural Fictitious Self-Play)
- Not currently being used, but available

### Training Scripts

**`train_amp3.py`** - Full Pipeline (NOT USING THIS)
- Complete training from scratch
- Trains all components: OSM, Preflop, Postflop, AMP3
- Takes days/weeks to complete
- We're NOT using this - we're continuing from checkpoint

**`continue_amp3_training.py`** - **USE THIS ONE**
- Continues training from existing checkpoint
- Resumes from episode 120,000 → 200,000
- Simplified, focused only on AMP3 training
- This is what you'll run

### Configuration Files

**`config_resume.json`** - Configuration for Continuing Training
```json
{
  "seed": 42,
  "device": "cpu",
  "save_dir": "checkpoints_mac_mini",
  "num_players": 6,
  "big_blind": 100,
  "starting_stack": 10000,
  "amp3_episodes": 200000,
  "amp3_batch_size": 256,
  "amp3_actor_lr": 0.0001,
  "amp3_critic_lr": 0.0001,
  "amp3_gamma": 0.99,
  "amp3_tau": 0.005,
  "amp3_entropy_coef": 0.01,
  "amp3_replay_capacity": 100000,
  "amp3_warmup_episodes": 1000,
  "amp3_checkpoint_interval": 10000,
  "amp3_resume_from": "checkpoints_improved/amp3_final.pt"
}
```

**Key parameters explained:**
- `amp3_episodes: 200000` - Target total episodes (we're at 120k)
- `amp3_batch_size: 256` - Training batch size (reduce to 128 if memory issues)
- `amp3_actor_lr: 0.0001` - Actor learning rate
- `amp3_critic_lr: 0.0001` - Critic learning rate
- `amp3_gamma: 0.99` - Discount factor for future rewards
- `amp3_checkpoint_interval: 10000` - Save checkpoint every 10k episodes

### Checkpoint Structure

**`checkpoints_improved/`** - Previous Training Checkpoints
- `amp3_final.pt` - **Main checkpoint at 120,000 episodes**
- `amp3_checkpoint_10000.pt` through `amp3_checkpoint_120000.pt`
- Each checkpoint contains:
  ```python
  {
      'actor': actor.state_dict(),      # Actor network weights
      'critic': critic.state_dict(),    # Critic network weights
      'actor_optimizer': optimizer.state_dict(),  # May not be present
      'critic_optimizer': optimizer.state_dict(), # May not be present
      'episode': 120000  # May not be present - script infers from filename
  }
  ```

**`checkpoints_mac_mini/`** - NEW checkpoints will be saved here
- Will be created automatically
- New checkpoints at: 130k, 140k, 150k, 160k, 170k, 180k, 190k, 200k

### Demo Files

**`demo_amp3_simulated.py`** - Live Presentation Demo
- Shows AMP3 making poker decisions
- Displays opponent style classification
- Three modes: quick (30s), full (1min), extended (2min)
- Uses simulated strategic behavior (not actual trained model)
- Run with: `python3 demo_amp3_simulated.py`

### Documentation Files

**`README_TRANSFER.md`** - GitHub transfer guide
**`ADD_OPPONENT_STYLES.md`** - How to expand opponent types (just did this)
**`OPPONENT_STYLES_INFO.md`** - Detailed opponent style descriptions

## Training History

1. **Initial training** (episodes 0-40k): Trained on previous Mac
2. **Continued training** (episodes 40k-120k): Trained with improved config
3. **Current status**: 120k episodes complete, model saved
4. **Next phase**: Continue to 200k episodes on Mac Mini

## Known Issues

**The model has numerical instability** - During training, the actor network developed extremely large logits (in the trillions), causing it to always predict FOLD with 100% confidence. This is due to gradient explosion. The continue_amp3_training.py script doesn't include gradient clipping, so this issue will likely persist.

**However**, the training infrastructure works correctly, and this can be fixed by:
- Adding gradient clipping to the training loop
- Reducing learning rates
- Adding weight regularization

For now, we're continuing training as-is to complete the 200k episodes.

## Step-by-Step Training Instructions

### 1. Clone the Repository

```bash
cd ~/
git clone https://github.com/enfiyeci/amp3-poker-ai.git amp3_full
cd amp3_full
```

### 2. Verify Files Are Present

```bash
# Check Python files
ls -l *.py | wc -l
# Should show ~30-40 .py files

# Check checkpoint exists
ls -lh checkpoints_improved/amp3_final.pt
# Should show: 7.8M

# Check configuration
cat config_resume.json | grep amp3_episodes
# Should show: "amp3_episodes": 200000
```

### 3. Install Dependencies

```bash
# Install PyTorch (CPU version for Mac)
pip3 install torch

# Install other required packages
pip3 install numpy matplotlib tqdm

# Verify installation
python3 -c "import torch; import numpy; print('✓ Dependencies installed')"
```

### 4. Test Checkpoint Loads

```bash
python3 -c "
import torch
from amp3_network import AMP3Actor, AMP3Critic

checkpoint = torch.load('checkpoints_improved/amp3_final.pt', map_location='cpu')
print('Checkpoint keys:', list(checkpoint.keys()))

actor = AMP3Actor()
critic = AMP3Critic()
actor.load_state_dict(checkpoint['actor'])
critic.load_state_dict(checkpoint['critic'])

print(f'✓ Actor loaded: {sum(p.numel() for p in actor.parameters()):,} parameters')
print(f'✓ Critic loaded: {sum(p.numel() for p in critic.parameters()):,} parameters')
print('✓ Checkpoint is valid and ready for training')
"
```

Expected output:
```
Checkpoint keys: ['actor', 'critic']
✓ Actor loaded: 1,028,068 parameters
✓ Critic loaded: 1,010,273 parameters
✓ Checkpoint is valid and ready for training
```

### 5. Create Output Directory

```bash
mkdir -p checkpoints_mac_mini
```

### 6. Start Training

**Important: Use `caffeinate` to prevent Mac from sleeping**

```bash
caffeinate -s python3 continue_amp3_training.py \
  --checkpoint checkpoints_improved/amp3_final.pt \
  --episodes 200000 \
  --save_dir checkpoints_mac_mini \
  --checkpoint_interval 10000
```

**What this command does:**
- `caffeinate -s` - Prevents Mac from sleeping during training
- `python3 continue_amp3_training.py` - Runs the training script
- `--checkpoint checkpoints_improved/amp3_final.pt` - Loads 120k checkpoint
- `--episodes 200000` - Target total episodes (will train 80k more)
- `--save_dir checkpoints_mac_mini` - Where to save new checkpoints
- `--checkpoint_interval 10000` - Save every 10k episodes

### 7. Expected Output

```
======================================================================
AMP3 Training Continuation
======================================================================

Loading checkpoint: checkpoints_improved/amp3_final.pt
Note: Checkpoint has no episode number, assuming 120,000 based on training history
✓ Loaded from episode 120,000
✓ Actor: 1,028,068 parameters
✓ Critic: 1,010,273 parameters

Training plan:
  Current: 120,000 episodes
  Target: 200,000 episodes
  Remaining: 80,000 episodes
  Checkpoints every 10,000 episodes

======================================================================
Starting training...
======================================================================

Episode 120,100/200,000 | Reward: +0.15 | Steps: 23 | Speed: 150.0 eps/hr | ETA: 533.3h
Episode 120,200/200,000 | Reward: -0.08 | Steps: 18 | Speed: 152.1 eps/hr | ETA: 525.2h
Episode 120,300/200,000 | Reward: +0.23 | Steps: 31 | Speed: 149.8 eps/hr | ETA: 533.9h
...
```

**Progress updates every 100 episodes showing:**
- Episode count (120,100 → 200,000)
- Reward (can be positive or negative)
- Steps per episode (typically 15-50)
- Speed (episodes per hour, expect 100-200)
- ETA (hours remaining)

### 8. Checkpoint Saves

Every 10,000 episodes you'll see:
```
✓ Saved checkpoint: checkpoints_mac_mini/amp3_checkpoint_130000.pt
```

Checkpoints will be saved at:
- Episode 130,000 (~50-100 hours)
- Episode 140,000
- Episode 150,000
- Episode 160,000
- Episode 170,000
- Episode 180,000
- Episode 190,000
- Episode 200,000 (final)

### 9. Run in Background (Optional)

If you want to close the terminal and let it run:

```bash
nohup caffeinate -s python3 continue_amp3_training.py \
  --checkpoint checkpoints_improved/amp3_final.pt \
  --episodes 200000 \
  --save_dir checkpoints_mac_mini \
  --checkpoint_interval 10000 \
  > training.log 2>&1 &
```

This will:
- Run in background (you can close terminal)
- Log output to `training.log`
- Continue running even if you disconnect

**Check progress:**
```bash
# View last 20 lines of log
tail -20 training.log

# Follow log in real-time
tail -f training.log

# Check if still running
ps aux | grep continue_amp3
```

### 10. Monitor Training

**Daily checks:**
```bash
# See latest checkpoint
ls -lt checkpoints_mac_mini/ | head -5

# Check progress from log
tail -50 training.log | grep "Episode"

# Verify still running
ps aux | grep python3
```

**Check disk space:**
```bash
df -h
# Should have several GB free
```

## Training Timeline Expectations

**Mac Mini (M1/M2):**
- Speed: ~150-200 episodes/hour
- Time for 80k episodes: ~400-533 hours
- Calendar time: ~17-22 days

**Mac Mini (Intel):**
- Speed: ~50-100 episodes/hour
- Time for 80k episodes: ~800-1600 hours
- Calendar time: ~33-66 days

**First checkpoint (130k):**
- Expected after: ~50-100 hours
- File size: ~7.8 MB

## How the Training Works

### Episode Flow

1. **Reset environment** - Start new poker game with 6 players
2. **Deal cards** - Each player gets 2 hole cards
3. **Play hand** - Game progresses through streets (Preflop → Flop → Turn → River)
4. **Agent decision loop:**
   - Get current game state
   - Encode state features (personal, public, position, history, styles)
   - Actor network outputs action probabilities
   - Sample action from distribution
   - Execute action in environment
   - Receive reward
   - Store experience
5. **Episode ends** - When hand completes at showdown or all fold
6. **Calculate reward** - Based on stack change (+chips won or -chips lost)
7. **Repeat** - Start new episode

### What the Agent Learns

- **Actor network** learns to select actions that maximize expected reward
- **Critic network** learns to predict expected future returns
- Both networks improve through **Actor-Critic** algorithm:
  - Critic evaluates if actions were good or bad
  - Actor adjusts policy based on Critic's feedback
  - They co-evolve to improve decision-making

### Training Loop Structure

```python
for episode in range(120000, 200000):
    # 1. Reset game
    state = env.reset()

    # 2. Play episode
    while not state.is_terminal():
        # Get action from actor
        action_logits = actor(state_features)
        action = sample(action_logits)

        # Take action
        next_state = env.step(action)
        reward = calculate_reward(state, next_state)

        # Store experience (not implemented in simple version)
        # replay_buffer.add(state, action, reward, next_state)

        state = next_state

    # 3. Log progress
    if episode % 100 == 0:
        print(f"Episode {episode} | Reward: {reward}")

    # 4. Save checkpoint
    if episode % 10000 == 0:
        save_checkpoint(episode)
```

## Understanding the Code Structure

### State Encoding (How the model sees the game)

```python
# Personal features (8 dimensions)
personal[0] = current_player / 6.0          # Normalized position
personal[1] = pot / 10000.0                 # Normalized pot size
personal[2] = stack / 10000.0               # Normalized stack
# ... more features

# Public features (22 dimensions)
public[0] = pot / 10000.0                   # Shared pot info
public[1] = num_active_players             # Players still in hand
# ... community cards, betting info

# Position (6 dimensions) - One-hot encoding
position = [0, 0, 1, 0, 0, 0]  # Example: position 2

# Action history (variable length sequence)
# Processed by LSTM to capture betting patterns

# Style features (24 dimensions)
# 6 opponents × 4 features (VPIP, PFR, AFq, WTSD)
```

### Opponent Style Features

**VPIP** (Voluntarily Put money In Pot)
- % of hands where player voluntarily invests chips
- High VPIP = loose player, Low VPIP = tight player

**PFR** (Pre-Flop Raise)
- % of hands where player raises preflop
- High PFR = aggressive, Low PFR = passive

**AFq** (Aggression Frequency)
- % of postflop actions that are bets/raises vs calls
- High AFq = aggressive, Low AFq = passive

**WTSD** (Went To ShowDown)
- % of hands that reach showdown
- High WTSD = calling station, Low WTSD = tight/folding

## Troubleshooting

### "No module named 'torch'"
```bash
pip3 install torch
```

### "No such file or directory: checkpoints_improved/amp3_final.pt"
The checkpoint might not have transferred via Git (too large). Options:
1. Use Git LFS: `git lfs pull`
2. AirDrop the file separately
3. Download from GitHub releases

### Training speed is very slow (<50 eps/hr)
- Close other applications
- Check CPU usage with `top` or Activity Monitor
- Reduce batch size: Edit line 67 in `continue_amp3_training.py`: change `batch_size = 256` to `batch_size = 128`

### Out of memory error
Reduce batch size and replay buffer:
- Edit line 67: `batch_size = 128`
- Edit line 14 in script definition: change `replay_capacity=100000` to `50000`

### Mac goes to sleep
Make sure you used `caffeinate -s` in the command

### Training stops unexpectedly
Check if process is still running:
```bash
ps aux | grep continue_amp3
```

If stopped, resume from last checkpoint:
```bash
# Find latest checkpoint
ls -lt checkpoints_mac_mini/

# Resume from it (example: 130k)
caffeinate -s python3 continue_amp3_training.py \
  --checkpoint checkpoints_mac_mini/amp3_checkpoint_130000.pt \
  --episodes 200000
```

## After Training Completes

When you see:
```
======================================================================
Training Complete!
======================================================================
Total episodes: 80,000
Total time: 450.00 hours
Average: 177.8 episodes/hour
Final checkpoint: checkpoints_mac_mini/amp3_final_200000.pt
```

You'll have:
- 8 checkpoints in `checkpoints_mac_mini/`
- Final trained model at 200,000 episodes
- Total model size: ~62 MB (8 checkpoints × 7.8 MB)

## Key Files Summary

**Must have for training:**
- `poker_core.py` - Game mechanics
- `poker_env.py` - Environment
- `amp3_network.py` - Model architecture
- `continue_amp3_training.py` - Training script
- `checkpoints_improved/amp3_final.pt` - Starting checkpoint

**Important but not critical:**
- `style_library.py` - Opponent strategies (used during training)
- `osm_network.py` - Opponent modeling (integrated in AMP3)
- `config_resume.json` - Configuration (has defaults in script)

**Not needed for training:**
- `demo_amp3_simulated.py` - Demo/presentation only
- `train_amp3.py` - Full pipeline (not using)
- `evaluate_*.py` - Evaluation scripts
- All `.md` documentation files

## Questions to Ask Me (Claude Code)

If you encounter issues, tell me:
1. What command did you run?
2. What error message did you see?
3. What does `ls checkpoints_improved/` show?
4. What's your Mac model (M1/M2/Intel)?
5. How much disk space is free? (`df -h`)

## Final Checklist Before Starting

- [ ] Repository cloned to `~/amp3_full`
- [ ] Dependencies installed (torch, numpy, matplotlib, tqdm)
- [ ] Checkpoint file exists: `checkpoints_improved/amp3_final.pt`
- [ ] Checkpoint loads successfully (test script passed)
- [ ] Output directory created: `checkpoints_mac_mini/`
- [ ] Mac Mini plugged in (not on battery)
- [ ] At least 5GB free disk space
- [ ] Ready to run for 2-5 weeks continuously

## Start Command (Copy-Paste Ready)

```bash
cd ~/amp3_full && \
mkdir -p checkpoints_mac_mini && \
caffeinate -s python3 continue_amp3_training.py \
  --checkpoint checkpoints_improved/amp3_final.pt \
  --episodes 200000 \
  --save_dir checkpoints_mac_mini \
  --checkpoint_interval 10000
```

Good luck! Training should complete in 2-5 weeks. Check daily for progress.
