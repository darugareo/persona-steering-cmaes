"""
Multi-Persona CMA-ES Optimization

Runs CMA-ES optimization for multiple personas in batch.
Reads persona list from YAML config and runs optimization for each persona.

Usage:
    python scripts/run_multi_persona_optimization.py \
      --persona-list persona-opt/personas.yaml \
      --max-iters 10 \
      --seed 1
"""

import argparse
import subprocess
import sys
import yaml
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple


def load_persona_list(config_path: str) -> List[str]:
    """Load persona list from YAML config."""
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Persona list config not found: {config_path}")

    with open(config_file, 'r') as f:
        data = yaml.safe_load(f)

    personas = data.get('personas', [])

    if not personas:
        raise ValueError(f"No personas found in {config_path}")

    return personas


def optimize_single_persona(
    persona_id: str,
    layer: int,
    alpha: float,
    max_iters: int,
    num_prompts: int,
    seed: int,
    sigma0: float,
    model: str,
    save_dir: str,
    log_file
) -> Tuple[bool, str]:
    """
    Run CMA-ES optimization for a single persona.

    Returns:
        (success, message)
    """
    print(f"\n{'='*80}", file=log_file)
    print(f"Optimizing persona: {persona_id}", file=log_file)
    print(f"Timestamp: {datetime.now().isoformat()}", file=log_file)
    print(f"{'='*80}", file=log_file, flush=True)

    # Build command
    cmd = [
        "python", "-u", "-B", "scripts/run_persona_optimization.py",
        "--persona-id", persona_id,
        "--layer", str(layer),
        "--alpha", str(alpha),
        "--max-iterations", str(max_iters),
        "--num-prompts", str(num_prompts),
        "--sigma0", str(sigma0),
        "--model", model,
        "--save-dir", save_dir,
    ]

    print(f"Command: {' '.join(cmd)}", file=log_file, flush=True)

    try:
        # Run optimization
        result = subprocess.run(
            cmd,
            check=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True
        )

        # Check if best_weights.json was created
        weights_file = Path(save_dir) / f"{persona_id}_best_weights.json"

        if weights_file.exists():
            msg = f"✓ Success: {weights_file}"
            print(f"\n{msg}", file=log_file, flush=True)
            return True, str(weights_file)
        else:
            msg = f"✗ Warning: Optimization completed but weights file not found: {weights_file}"
            print(f"\n{msg}", file=log_file, flush=True)
            return False, msg

    except subprocess.CalledProcessError as e:
        msg = f"✗ Failed with error code {e.returncode}"
        print(f"\n{msg}", file=log_file, flush=True)
        return False, msg
    except Exception as e:
        msg = f"✗ Failed with exception: {e}"
        print(f"\n{msg}", file=log_file, flush=True)
        return False, msg


def main():
    parser = argparse.ArgumentParser(
        description="Run CMA-ES optimization for multiple personas"
    )

    parser.add_argument(
        "--persona-list",
        type=str,
        default="persona-opt/personas.yaml",
        help="Path to persona list YAML config (default: persona-opt/personas.yaml)"
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
        "--max-iters",
        type=int,
        default=10,
        help="Maximum CMA-ES iterations per persona (default: 10)"
    )

    parser.add_argument(
        "--num-prompts",
        type=int,
        default=10,
        help="Number of evaluation prompts (default: 10)"
    )

    parser.add_argument(
        "--sigma0",
        type=float,
        default=1.0,
        help="Initial CMA-ES sigma (default: 1.0)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed (default: 1)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Model name (default: Meta-Llama-3-8B-Instruct)"
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default="persona-opt",
        help="Directory to save optimization results (default: persona-opt)"
    )

    args = parser.parse_args()

    # Load persona list
    try:
        persona_list = load_persona_list(args.persona_list)
    except Exception as e:
        print(f"✗ Failed to load persona list: {e}")
        sys.exit(1)

    print("=" * 80)
    print("Multi-Persona CMA-ES Optimization")
    print("=" * 80)
    print(f"Persona list config: {args.persona_list}")
    print(f"Number of personas: {len(persona_list)}")
    print(f"Personas: {', '.join(persona_list)}")
    print(f"Model: {args.model}")
    print(f"Layer: {args.layer}")
    print(f"Alpha: {args.alpha}")
    print(f"Max iterations per persona: {args.max_iters}")
    print(f"Num prompts: {args.num_prompts}")
    print(f"Sigma0: {args.sigma0}")
    print(f"Seed: {args.seed}")
    print(f"Save directory: {args.save_dir}")
    print("=" * 80)

    # Create log directory
    log_dir = Path("logs/phase1")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"multi_persona_optimization_seed{args.seed}.log"

    # Open log file
    with open(log_path, 'w') as log_file:
        # Write header
        print("=" * 80, file=log_file)
        print("Multi-Persona CMA-ES Optimization", file=log_file)
        print("=" * 80, file=log_file)
        print(f"Started: {datetime.now().isoformat()}", file=log_file)
        print(f"Persona list: {args.persona_list}", file=log_file)
        print(f"Number of personas: {len(persona_list)}", file=log_file)
        print(f"Personas: {', '.join(persona_list)}", file=log_file)
        print(f"Model: {args.model}", file=log_file)
        print(f"Layer: {args.layer}", file=log_file)
        print(f"Alpha: {args.alpha}", file=log_file)
        print(f"Max iterations: {args.max_iters}", file=log_file)
        print(f"Num prompts: {args.num_prompts}", file=log_file)
        print(f"Sigma0: {args.sigma0}", file=log_file)
        print(f"Seed: {args.seed}", file=log_file)
        print(f"Save directory: {args.save_dir}", file=log_file)
        print("=" * 80, file=log_file, flush=True)

        # Track results
        results: Dict[str, Tuple[bool, str]] = {}

        # Optimize each persona
        for i, persona_id in enumerate(persona_list, 1):
            print(f"\n[{i}/{len(persona_list)}] Processing persona: {persona_id}")

            success, message = optimize_single_persona(
                persona_id=persona_id,
                layer=args.layer,
                alpha=args.alpha,
                max_iters=args.max_iters,
                num_prompts=args.num_prompts,
                seed=args.seed,
                sigma0=args.sigma0,
                model=args.model,
                save_dir=args.save_dir,
                log_file=log_file
            )

            results[persona_id] = (success, message)

            # Print progress to console
            if success:
                print(f"  ✓ {persona_id}: {message}")
            else:
                print(f"  ✗ {persona_id}: {message}")

        # Write summary
        print("\n" + "=" * 80, file=log_file)
        print("OPTIMIZATION SUMMARY", file=log_file)
        print("=" * 80, file=log_file)
        print(f"Completed: {datetime.now().isoformat()}", file=log_file)
        print(f"Total personas: {len(persona_list)}", file=log_file)

        successful = [p for p, (success, _) in results.items() if success]
        failed = [p for p, (success, _) in results.items() if not success]

        print(f"Successful: {len(successful)}", file=log_file)
        print(f"Failed: {len(failed)}", file=log_file)
        print("=" * 80, file=log_file)

        if successful:
            print("\nSuccessful optimizations:", file=log_file)
            for persona_id in successful:
                _, weights_path = results[persona_id]
                print(f"  ✓ {persona_id}: {weights_path}", file=log_file)

        if failed:
            print("\nFailed optimizations:", file=log_file)
            for persona_id in failed:
                _, error_msg = results[persona_id]
                print(f"  ✗ {persona_id}: {error_msg}", file=log_file)

        print("=" * 80, file=log_file, flush=True)

    # Print summary to console
    print("\n" + "=" * 80)
    print("OPTIMIZATION SUMMARY")
    print("=" * 80)
    print(f"Total personas: {len(persona_list)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print("=" * 80)

    if successful:
        print("\n✓ Successful optimizations:")
        for persona_id in successful:
            _, weights_path = results[persona_id]
            print(f"  • {persona_id}")
            print(f"    {weights_path}")

    if failed:
        print("\n✗ Failed optimizations:")
        for persona_id in failed:
            _, error_msg = results[persona_id]
            print(f"  • {persona_id}: {error_msg}")

    print(f"\n✓ Log saved to: {log_path}")
    print("=" * 80)

    # Exit with error code if any failed
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
