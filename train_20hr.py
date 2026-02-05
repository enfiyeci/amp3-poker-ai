#!/usr/bin/env python3
"""
20-Hour Optimized Training Plan for AMP3

Trains models in optimal sequence with reduced sample sizes
to fit within 20-hour time budget while maintaining quality.
"""

import os
import sys
import time
import subprocess
from datetime import datetime, timedelta

def print_banner(text):
    print("\n" + "="*60)
    print(text)
    print("="*60 + "\n")

def print_phase(phase_num, name, duration):
    print(f"\n{'─'*60}")
    print(f"Phase {phase_num}: {name}")
    print(f"Estimated duration: {duration}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('─'*60 + "\n")

def run_command(cmd, log_file=None):
    """Run command and optionally log to file"""
    if log_file:
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=f,
                stderr=subprocess.STDOUT
            )
        return process
    else:
        return subprocess.run(cmd, shell=True, check=True)

def main():
    print_banner("AMP3 20-Hour Training Plan")

    start_time = datetime.now()
    print(f"Training started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Expected completion: {(start_time + timedelta(hours=20)).strftime('%Y-%m-%d %H:%M:%S')}")

    # Create checkpoint directory
    os.makedirs('checkpoints_20hr', exist_ok=True)

    # Phase 1: OSM Training (3-4 hours)
    print_phase(1, "OSM Training", "3-4 hours")
    phase1_start = time.time()

    # Check if OSM already trained
    if os.path.exists('checkpoints_20hr/osm_network.pt'):
        print("⚠️  OSM model already exists. Skipping...")
    else:
        print("Training Opponent Style Modeling network...")
        print("Config: 5,000 games, 50 epochs")

        # Update config in train_amp3.py temporarily
        run_command(
            "python3 train_amp3.py --stage osm_training",
            log_file='checkpoints_20hr/osm_training.log'
        )

    phase1_time = (time.time() - phase1_start) / 3600
    print(f"✅ Phase 1 complete in {phase1_time:.1f} hours")

    # Phase 2: Later-Street Models (12-15 hours, parallel)
    print_phase(2, "Later-Street Models (Parallel)", "12-15 hours")
    phase2_start = time.time()

    print("Starting Flop, Turn, and River training in parallel...")
    print("Monitor progress:")
    print("  tail -f checkpoints_20hr/flop.log")
    print("  tail -f checkpoints_20hr/turn.log")
    print("  tail -f checkpoints_20hr/river.log")
    print()

    # Start all three in parallel
    processes = []

    if not os.path.exists('checkpoints_20hr/flop_network.pt'):
        print("→ Starting Flop model training...")
        proc = run_command(
            "python3 train_amp3.py --stage flop_model",
            log_file='checkpoints_20hr/flop.log'
        )
        processes.append(('Flop', proc))
    else:
        print("⚠️  Flop model exists. Skipping...")

    if not os.path.exists('checkpoints_20hr/turn_network.pt'):
        print("→ Starting Turn model training...")
        proc = run_command(
            "python3 train_amp3.py --stage turn_model",
            log_file='checkpoints_20hr/turn.log'
        )
        processes.append(('Turn', proc))
    else:
        print("⚠️  Turn model exists. Skipping...")

    if not os.path.exists('checkpoints_20hr/river_network.pt'):
        print("→ Starting River model training...")
        proc = run_command(
            "python3 train_amp3.py --stage river_model",
            log_file='checkpoints_20hr/river.log'
        )
        processes.append(('River', proc))
    else:
        print("⚠️  River model exists. Skipping...")

    # Wait for all processes
    print(f"\nWaiting for {len(processes)} training jobs to complete...")
    for name, proc in processes:
        proc.wait()
        print(f"✅ {name} training complete!")

    phase2_time = (time.time() - phase2_start) / 3600
    print(f"\n✅ Phase 2 complete in {phase2_time:.1f} hours")

    # Phase 3: AMP3 RL Training (3-4 hours)
    print_phase(3, "AMP3 Actor-Critic RL", "3-4 hours")
    phase3_start = time.time()

    if os.path.exists('checkpoints_20hr/amp3_actor.pt'):
        print("⚠️  AMP3 model already exists. Skipping...")
    else:
        print("Training AMP3 Actor-Critic with reinforcement learning...")
        print("Config: 20,000 episodes (quick but functional)")

        run_command(
            "python3 train_amp3.py --stage amp3_rl",
            log_file='checkpoints_20hr/amp3_rl.log'
        )

    phase3_time = (time.time() - phase3_start) / 3600
    print(f"✅ Phase 3 complete in {phase3_time:.1f} hours")

    # Summary
    total_time = (time.time() - phase1_start) / 3600
    end_time = datetime.now()

    print_banner("Training Complete!")

    print(f"Start time:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time:    {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time:  {total_time:.1f} hours")
    print()
    print("Phase breakdown:")
    print(f"  Phase 1 (OSM):          {phase1_time:.1f}h")
    print(f"  Phase 2 (Later Streets): {phase2_time:.1f}h")
    print(f"  Phase 3 (AMP3 RL):      {phase3_time:.1f}h")
    print()
    print("Trained models:")

    for f in sorted(os.listdir('checkpoints_20hr')):
        if f.endswith('.pt'):
            size = os.path.getsize(f'checkpoints_20hr/{f}') / (1024*1024)
            print(f"  ✓ {f:<30} ({size:.1f} MB)")

    print("\n" + "="*60)
    print("Your AMP3 poker AI is ready to use!")
    print("="*60 + "\n")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        sys.exit(1)
