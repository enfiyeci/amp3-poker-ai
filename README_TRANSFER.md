# Transfer AMP3 to Mac Mini via GitHub

## Quick Method (5 minutes)

### 1. Push to GitHub (Current Mac)

```bash
cd /Users/ardaenfiyeci/Downloads/amp3_full

# Initialize git if needed
git init

# Create .gitignore
cat > .gitignore << 'EOF'
__pycache__/
*.pyc
.DS_Store
.claude/
checkpoints_20hr/
checkpoints_test/
amp3_checkpoints/
checkpoints_mac_mini/
*.log
*.csv
Pool_Fusball League/
poker_amp3_preflop_demo/
EOF

# Add and commit
git add .
git commit -m "AMP3 poker AI - ready for Mac Mini transfer"

# Create repo on GitHub, then:
# (Replace YOUR_USERNAME/REPO_NAME with your actual GitHub repo)
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
git push -u origin main
```

### 2. Clone on Mac Mini

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/REPO_NAME.git ~/amp3_full
cd ~/amp3_full

# Install dependencies
pip3 install torch numpy matplotlib tqdm

# Verify
python3 -c "import torch; print('✓ Ready')"
```

### 3. Start Training

```bash
cd ~/amp3_full
mkdir -p checkpoints_mac_mini

caffeinate -s python3 continue_amp3_training.py \
  --checkpoint checkpoints_improved/amp3_final.pt \
  --episodes 200000 \
  --save_dir checkpoints_mac_mini
```

## Note on Checkpoint File

The checkpoint file (`amp3_final.pt`, 7.8 MB) should transfer fine via GitHub. If you get "file too large" error:

**Just AirDrop it separately:**
1. On current Mac: Open `checkpoints_improved/` folder
2. AirDrop `amp3_final.pt` to Mac Mini
3. On Mac Mini: Move it to `~/amp3_full/checkpoints_improved/`

## That's It!

Training will resume from episode 120,000 and continue to 200,000.

Monitor with:
```bash
tail -f training.log
```
