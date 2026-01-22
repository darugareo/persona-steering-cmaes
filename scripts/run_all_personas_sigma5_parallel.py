#!/usr/bin/env python3
"""
全ペルソナをσ=5.0で並列最適化

GPU 0とGPU 1を使用して並列実行
"""

import subprocess
import json
from pathlib import Path
import time
from datetime import datetime
import multiprocessing as mp


# 全ペルソナリスト
PERSONAS = [
    "episode-118328_B",
    "episode-128744_B",
    "episode-134226_A",
    "episode-136981_B",
    "episode-137872_B",
    "episode-140544_B",
    "episode-14330_A",
    "episode-145896_A",
    "episode-158821_B",
    "episode-16276_B",
    "episode-175246_A",
    "episode-179307_A",
    "episode-184019_A",
    "episode-19493_A",
    "episode-204347_A",
    "episode-223194_B",
    "episode-225888_A",
    "episode-239427_A",
    "episode-24275_A",
    "episode-36796_A",
    "episode-36796_B",
    "episode-37624_A",
    "episode-38144_A",
    "episode-51953_A",
    "episode-74475_A",
    "episode-84804_A",
    "episode-98323_A",
    "episode-98947_A",
]

OUTPUT_DIR = "optimization_results_sigma5"
SIGMA = 5.0
ALPHA = 2.0


def optimize_persona(persona_id: str, device: str):
    """1ペルソナを最適化"""
    cmd = [
        "python", "scripts/optimize_single_persona_sigma5.py",
        "--persona_id", persona_id,
        "--device", device,
        "--output_dir", OUTPUT_DIR,
        "--sigma", str(SIGMA),
        "--alpha", str(ALPHA),
        "--max_iterations", "20",
        "--population_size", "10"
    ]

    log_file = Path(OUTPUT_DIR) / persona_id / "optimization_log.txt"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting: {persona_id} on {device}")

    with open(log_file, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

    if result.returncode == 0:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Completed: {persona_id}")
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Failed: {persona_id}")

    return persona_id, result.returncode


def parallel_optimize(personas: list, num_gpus: int = 2):
    """複数GPUで並列最適化"""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    print(f"\n{'='*80}")
    print(f"PARALLEL OPTIMIZATION: σ={SIGMA}")
    print(f"{'='*80}")
    print(f"  Total personas: {len(personas)}")
    print(f"  GPUs: {num_gpus}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"{'='*80}\n")

    start_time = time.time()

    # GPU割り当て
    persona_gpu_pairs = [(p, f"cuda:{i % num_gpus}") for i, p in enumerate(personas)]

    results = []
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = {
            executor.submit(optimize_persona, persona_id, device): persona_id
            for persona_id, device in persona_gpu_pairs
        }

        for future in as_completed(futures):
            persona_id = futures[future]
            try:
                result_id, return_code = future.result()
                results.append({
                    "persona_id": result_id,
                    "success": return_code == 0
                })
            except Exception as e:
                print(f"❌ Exception for {persona_id}: {e}")
                results.append({
                    "persona_id": persona_id,
                    "success": False,
                    "error": str(e)
                })

    end_time = time.time()
    duration = end_time - start_time

    # サマリー生成
    print(f"\n{'='*80}")
    print(f"✅ ALL OPTIMIZATIONS COMPLETE")
    print(f"{'='*80}")
    print(f"  Duration: {duration/3600:.2f} hours")
    print(f"  Successful: {sum(1 for r in results if r['success'])}/{len(personas)}")
    print(f"{'='*80}\n")

    # 結果を集計
    aggregate_results(personas)


def aggregate_results(personas: list):
    """全ペルソナの結果を集計"""
    print(f"\n{'='*80}")
    print(f"AGGREGATING RESULTS")
    print(f"{'='*80}\n")

    all_results = []

    for persona_id in personas:
        summary_path = Path(OUTPUT_DIR) / persona_id / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                data = json.load(f)
                all_results.append(data)
        else:
            print(f"⚠️  Missing: {persona_id}")

    if not all_results:
        print("No results found.")
        return

    # L2ノルム統計
    l2_norms = [r["l2_norm"] for r in all_results]
    import numpy as np

    print(f"L2 Norm Statistics:")
    print(f"  Count: {len(l2_norms)}")
    print(f"  Mean: {np.mean(l2_norms):.3f}")
    print(f"  Std: {np.std(l2_norms):.3f}")
    print(f"  Min: {np.min(l2_norms):.3f}")
    print(f"  Max: {np.max(l2_norms):.3f}")
    print(f"  Median: {np.median(l2_norms):.3f}")

    # 分布
    print(f"\nL2 Norm Distribution:")
    print(f"  < 3.0: {sum(1 for x in l2_norms if x < 3.0)}")
    print(f"  3.0-5.0: {sum(1 for x in l2_norms if 3.0 <= x < 5.0)}")
    print(f"  5.0-7.0: {sum(1 for x in l2_norms if 5.0 <= x < 7.0)}")
    print(f"  7.0-10.0: {sum(1 for x in l2_norms if 7.0 <= x < 10.0)}")
    print(f"  >= 10.0: {sum(1 for x in l2_norms if x >= 10.0)}")

    # Top 5 & Bottom 5
    sorted_results = sorted(all_results, key=lambda x: x["l2_norm"], reverse=True)

    print(f"\nTop 5 L2 Norms:")
    for r in sorted_results[:5]:
        print(f"  {r['persona_id']}: {r['l2_norm']:.3f}")

    print(f"\nBottom 5 L2 Norms:")
    for r in sorted_results[-5:]:
        print(f"  {r['persona_id']}: {r['l2_norm']:.3f}")

    # 保存
    output_path = Path(OUTPUT_DIR) / "all_results_summary.json"
    with open(output_path, "w") as f:
        json.dump({
            "total_personas": len(all_results),
            "sigma": SIGMA,
            "alpha": ALPHA,
            "l2_norm_stats": {
                "mean": float(np.mean(l2_norms)),
                "std": float(np.std(l2_norms)),
                "min": float(np.min(l2_norms)),
                "max": float(np.max(l2_norms)),
                "median": float(np.median(l2_norms))
            },
            "results": all_results,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)

    print(f"\n📁 Saved aggregate results: {output_path}")


def main():
    print(f"\n{'='*80}")
    print(f"σ=5.0 BATCH OPTIMIZATION")
    print(f"{'='*80}")
    print(f"  Target: {len(PERSONAS)} personas")
    print(f"  Expected duration: ~{len(PERSONAS)*30/60/2:.1f} hours (2 GPUs)")
    print(f"{'='*80}\n")

    response = input("Start optimization? [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled.")
        return

    parallel_optimize(PERSONAS, num_gpus=2)


if __name__ == "__main__":
    main()
