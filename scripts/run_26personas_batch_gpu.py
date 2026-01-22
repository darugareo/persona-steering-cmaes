#!/usr/bin/env python3
"""
Batch Optimization for 26 Personas with GPU Parallelization
============================================================

Run CMA-ES optimization for multiple personas on a single GPU.

Usage:
    # GPU 0 (personas 0-12)
    CUDA_VISIBLE_DEVICES=0 python scripts/run_26personas_batch_gpu.py --gpu_id 0 --start 0 --end 13

    # GPU 1 (personas 13-25)
    CUDA_VISIBLE_DEVICES=1 python scripts/run_26personas_batch_gpu.py --gpu_id 1 --start 13 --end 26
"""

import argparse
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from persona_opt.cmaes_persona_optimizer import optimize_persona_weights

def main():
    parser = argparse.ArgumentParser(
        description="Batch CMA-ES optimization for 26 personas"
    )

    parser.add_argument(
        "--gpu_id",
        type=int,
        required=True,
        help="GPU ID (0 or 1)"
    )

    parser.add_argument(
        "--start",
        type=int,
        required=True,
        help="Start index (inclusive)"
    )

    parser.add_argument(
        "--end",
        type=int,
        required=True,
        help="End index (exclusive)"
    )

    parser.add_argument(
        "--layer",
        type=int,
        default=20,
        help="Layer to apply steering (default: 20)"
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=2.0,
        help="Steering strength (default: 2.0)"
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum CMA-ES iterations (default: 10)"
    )

    parser.add_argument(
        "--population-size",
        type=int,
        default=8,
        help="CMA-ES population size (default: 8)"
    )

    parser.add_argument(
        "--num-prompts",
        type=int,
        default=10,
        help="Number of evaluation prompts (default: 10 for speed)"
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default="optimization_results_26personas",
        help="Directory to save results"
    )

    args = parser.parse_args()

    # Load persona list
    personas_file = Path("personas_final_26.txt")
    with open(personas_file) as f:
        all_personas = [line.strip() for line in f if line.strip()]

    # Select personas for this GPU
    personas_to_process = all_personas[args.start:args.end]

    print("=" * 80)
    print(f"BATCH OPTIMIZATION - GPU {args.gpu_id}")
    print("=" * 80)
    print(f"Total personas in system: {len(all_personas)}")
    print(f"Personas for GPU {args.gpu_id}: {len(personas_to_process)}")
    print(f"Range: [{args.start}, {args.end})")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Population size: {args.population_size}")
    print(f"Eval prompts: {args.num_prompts}")
    print("=" * 80)

    # Create save directory
    save_dir = Path(args.save_dir) / f"gpu{args.gpu_id}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Batch log
    batch_log = {
        "gpu_id": args.gpu_id,
        "start_index": args.start,
        "end_index": args.end,
        "personas": personas_to_process,
        "config": {
            "layer": args.layer,
            "alpha": args.alpha,
            "max_iterations": args.max_iterations,
            "population_size": args.population_size,
            "num_prompts": args.num_prompts
        },
        "results": [],
        "start_time": datetime.now().isoformat()
    }

    # Load eval prompts
    eval_prompts_file = Path("data/eval_prompts/persona_eval_prompts_v1.json")
    if eval_prompts_file.exists():
        with open(eval_prompts_file) as f:
            data = json.load(f)
            all_prompts = [p["text"] for p in data["prompts"]]
            eval_prompts = all_prompts[:args.num_prompts]
    else:
        eval_prompts = None  # Use defaults

    print(f"\n✓ Loaded {len(eval_prompts) if eval_prompts else 'default'} evaluation prompts")

    # Process each persona
    successful = 0
    failed = 0

    for i, persona_id in enumerate(personas_to_process, 1):
        print("\n" + "=" * 80)
        print(f"GPU {args.gpu_id} | Persona {i}/{len(personas_to_process)}: {persona_id}")
        print("=" * 80)

        persona_start_time = time.time()

        try:
            # Run optimization
            results = optimize_persona_weights(
                persona_id=persona_id,
                layer=args.layer,
                alpha=args.alpha,
                max_iterations=args.max_iterations,
                population_size=args.population_size,
                sigma0=1.0,
                save_dir=str(save_dir),
                eval_prompts=eval_prompts
            )

            persona_elapsed = time.time() - persona_start_time

            # Log result
            batch_log["results"].append({
                "persona_id": persona_id,
                "status": "success",
                "best_score": results["best_score"],
                "num_iterations": results["num_iterations"],
                "elapsed_time_seconds": persona_elapsed,
                "timestamp": datetime.now().isoformat()
            })

            successful += 1

            print(f"\n✅ {persona_id} completed in {persona_elapsed/60:.1f} min")
            print(f"   Best score: {results['best_score']:.4f}")
            print(f"   Iterations: {results['num_iterations']}")

        except Exception as e:
            persona_elapsed = time.time() - persona_start_time

            print(f"\n❌ {persona_id} failed: {e}")

            batch_log["results"].append({
                "persona_id": persona_id,
                "status": "failed",
                "error": str(e),
                "elapsed_time_seconds": persona_elapsed,
                "timestamp": datetime.now().isoformat()
            })

            failed += 1

        # Save intermediate batch log
        batch_log_file = save_dir / f"batch_log_gpu{args.gpu_id}.json"
        with open(batch_log_file, 'w') as f:
            json.dump(batch_log, f, indent=2)

        print(f"\n📊 Progress: {successful + failed}/{len(personas_to_process)} | Success: {successful} | Failed: {failed}")

    # Final batch log
    batch_log["end_time"] = datetime.now().isoformat()
    batch_log["summary"] = {
        "total_personas": len(personas_to_process),
        "successful": successful,
        "failed": failed,
        "success_rate": successful / len(personas_to_process) if personas_to_process else 0
    }

    with open(batch_log_file, 'w') as f:
        json.dump(batch_log, f, indent=2)

    print("\n" + "=" * 80)
    print(f"BATCH COMPLETE - GPU {args.gpu_id}")
    print("=" * 80)
    print(f"Total: {len(personas_to_process)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/len(personas_to_process)*100:.1f}%")
    print(f"\nResults saved to: {save_dir}")
    print(f"Batch log: {batch_log_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
