#!/usr/bin/env python3
"""
Run steering optimization for all 21 target personas in parallel on 2 GPUs.
"""

import subprocess
import sys
from pathlib import Path

# Excluded personas (insufficient data)
EXCLUDED_PERSONAS = {
    "episode-204347_A", "episode-225888_A", "episode-239427_A",
    "episode-37624_A", "episode-38144_A", "episode-51953_A", "episode-98947_A"
}

def get_target_personas():
    """Get list of target personas (21 total)"""
    personas_dir = Path("personas_cc")
    all_personas = [p.name for p in personas_dir.iterdir()
                    if p.is_dir() and p.name.startswith("episode-")]
    return sorted([p for p in all_personas if p not in EXCLUDED_PERSONAS])

def main():
    personas = get_target_personas()
    print(f"Total personas to optimize: {len(personas)}")

    # Split personas across 2 GPUs
    mid = len(personas) // 2
    gpu0_personas = personas[:mid]
    gpu1_personas = personas[mid:]

    print(f"\nGPU 0: {len(gpu0_personas)} personas")
    print(f"GPU 1: {len(gpu1_personas)} personas")

    # Prepare commands
    cmd_gpu0 = [
        "CUDA_VISIBLE_DEVICES=0",
        "python", "scripts/optimize_steering_new.py",
        "--personas"] + gpu0_personas + [
        "--max_generations", "15",
        "--population_size", "8",
        "--sigma", "5.0",
        "--alpha", "2.0",
        ">", "steering_optimization_gpu0.log", "2>&1"
    ]

    cmd_gpu1 = [
        "CUDA_VISIBLE_DEVICES=1",
        "python", "scripts/optimize_steering_new.py",
        "--personas"] + gpu1_personas + [
        "--max_generations", "15",
        "--population_size", "8",
        "--sigma", "5.0",
        "--alpha", "2.0",
        ">", "steering_optimization_gpu1.log", "2>&1"
    ]

    # Run both processes
    print("\nStarting parallel optimization...")
    print("GPU 0 log: steering_optimization_gpu0.log")
    print("GPU 1 log: steering_optimization_gpu1.log")

    proc0 = subprocess.Popen(" ".join(cmd_gpu0), shell=True)
    proc1 = subprocess.Popen(" ".join(cmd_gpu1), shell=True)

    print(f"\nGPU 0 process: PID {proc0.pid}")
    print(f"GPU 1 process: PID {proc1.pid}")
    print("\nMonitor progress with:")
    print("  tail -f steering_optimization_gpu0.log")
    print("  tail -f steering_optimization_gpu1.log")

    # Wait for completion
    proc0.wait()
    proc1.wait()

    if proc0.returncode == 0 and proc1.returncode == 0:
        print("\n✅ All optimizations completed successfully!")
    else:
        print(f"\n⚠️  Some optimizations failed:")
        print(f"  GPU 0 exit code: {proc0.returncode}")
        print(f"  GPU 1 exit code: {proc1.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    main()
