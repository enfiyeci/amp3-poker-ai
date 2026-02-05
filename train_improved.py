#!/usr/bin/env python3
"""
Improved Training Script
1. Retrain OSM with more diverse data and longer training
2. Continue AMP3 training from checkpoint
"""

import os
import sys
import time
import subprocess
from datetime import datetime

def print_banner(text):
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70 + "\n")

def print_section(text):
    print("\n" + "─"*70)
    print(text)
    print("─"*70)

def main():
    print_banner("IMPROVED TRAINING PIPELINE")

    start_time = datetime.now()
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Ask user what to train
    print("What would you like to train?")
    print("  1. Retrain OSM with improved diversity (recommended)")
    print("  2. Continue AMP3 training (from 40k → 120k episodes)")
    print("  3. Both (OSM first, then AMP3)")
    print()

    choice = input("Enter choice (1/2/3): ").strip()

    if choice == "1" or choice == "3":
        print_section("PHASE 1: IMPROVED OSM TRAINING")
        print("\nImprovements:")
        print("  • 10x more training games (1M instead of 100k)")
        print("  • Longer training (200 epochs instead of 100)")
        print("  • Data augmentation for diversity")
        print("  • Validation monitoring")
        print()

        proceed = input("Start OSM training? This will take ~24 hours (y/n): ").strip().lower()

        if proceed == 'y':
            print("\nStarting improved OSM training...")
            print("Monitor progress: tail -f logs/osm_improved.log\n")

            cmd = (
                "python3 train_amp3.py --stage osm "
                "--osm_num_games 1000000 "
                "--osm_epochs 200 "
                "--osm_batch_size 256 "
                "> logs/osm_improved.log 2>&1"
            )

            os.makedirs('logs', exist_ok=True)

            print(f"Command: {cmd}")
            print("\nThis will run in the background...")

            subprocess.Popen(cmd, shell=True)

            print("\n✓ OSM training started!")
            print("  Check progress: tail -f logs/osm_improved.log")
            print("  Expected duration: ~24 hours")
            print()

    if choice == "2" or choice == "3":
        if choice == "3":
            print("\nNote: You should wait for OSM training to complete before training AMP3")
            print("      AMP3 will benefit from the improved OSM predictions")
            proceed = input("Continue anyway? (y/n): ").strip().lower()
            if proceed != 'y':
                print("Exiting. Run again when OSM is ready.")
                return

        print_section("PHASE 2: CONTINUE AMP3 TRAINING")
        print("\nCurrent state:")
        print("  • Episodes completed: 40,000")
        print("  • Episodes remaining: 80,000")
        print("  • Estimated time: ~8 hours")
        print()

        proceed = input("Continue AMP3 training from checkpoint? (y/n): ").strip().lower()

        if proceed == 'y':
            print("\nStarting AMP3 training from episode 40,000...")
            print("Monitor progress: tail -f logs/amp3_continue.log\n")

            cmd = (
                "python3 train_amp3.py --stage amp3 "
                "--amp3_episodes 120000 "
                "--amp3_resume checkpoints_20hr/amp3_checkpoint_40000.pt "
                "> logs/amp3_continue.log 2>&1"
            )

            os.makedirs('logs', exist_ok=True)

            print(f"Command: {cmd}")
            print("\nThis will run in the background...")

            subprocess.Popen(cmd, shell=True)

            print("\n✓ AMP3 training started!")
            print("  Check progress: tail -f logs/amp3_continue.log")
            print("  Expected duration: ~8 hours")
            print()

    print_banner("TRAINING STARTED")
    print("Active training sessions:")

    if choice == "1" or choice == "3":
        print("  → OSM: tail -f logs/osm_improved.log")

    if choice == "2" or choice == "3":
        print("  → AMP3: tail -f logs/amp3_continue.log")

    print("\nTo stop training: pkill -f train_amp3.py")
    print()

if __name__ == "__main__":
    main()
