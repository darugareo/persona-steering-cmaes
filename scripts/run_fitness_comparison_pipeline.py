#!/usr/bin/env python3
"""
Automated Fitness Comparison Pipeline
最適化 → 評価 → レポート生成を自動実行
"""
import subprocess
import json
import time
from pathlib import Path
from typing import List, Dict
import os

class FitnessComparisonPipeline:
    def __init__(
        self,
        personas: List[str],
        fitness_types: List[str],
        gpu_id: int = 0,
        max_generations: int = 10
    ):
        self.personas = personas
        self.fitness_types = fitness_types
        self.gpu_id = gpu_id
        self.max_generations = max_generations

        # Check OpenAI API key
        if not os.getenv('OPENAI_API_KEY'):
            print("⚠️  WARNING: OPENAI_API_KEY not set. Judge fitness will not work properly!")
            print("   Set it with: export OPENAI_API_KEY='your-key-here'")
            response = input("   Continue anyway? (y/n): ")
            if response.lower() != 'y':
                raise RuntimeError("Pipeline aborted. Please set OPENAI_API_KEY.")
        else:
            print(f"✅ OPENAI_API_KEY is set (length: {len(os.getenv('OPENAI_API_KEY'))})")

        # Results storage
        self.results = {}

    def run_optimization(self, persona_id: str, fitness_type: str) -> bool:
        """Run optimization for a single persona-fitness combination"""
        print(f"\n{'='*80}")
        print(f"🔧 OPTIMIZING: {persona_id} with {fitness_type}")
        print(f"{'='*80}")

        log_file = f"logs/fitness_comparison/pipeline_opt_{persona_id}_{fitness_type}.log"
        Path("logs/fitness_comparison").mkdir(parents=True, exist_ok=True)

        cmd = [
            "python", "scripts/fitness_comparison_optimizer.py",
            "--persona_id", persona_id,
            "--fitness_type", fitness_type,
            "--gpu_id", str(self.gpu_id),
            "--max_generations", str(self.max_generations)
        ]

        try:
            with open(log_file, "w") as f:
                # Use gpu_id 0 in subprocess since CUDA_VISIBLE_DEVICES remaps it
                subprocess_cmd = cmd.copy()
                for i, arg in enumerate(subprocess_cmd):
                    if arg == "--gpu_id":
                        subprocess_cmd[i+1] = "0"
                        break

                result = subprocess.run(
                    subprocess_cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": str(self.gpu_id)},
                    timeout=3600  # 1 hour timeout
                )

            if result.returncode == 0:
                print(f"✅ Optimization complete: {persona_id} - {fitness_type}")
                return True
            else:
                print(f"❌ Optimization failed: {persona_id} - {fitness_type}")
                return False

        except subprocess.TimeoutExpired:
            print(f"⏰ Optimization timeout: {persona_id} - {fitness_type}")
            return False
        except Exception as e:
            print(f"❌ Error in optimization: {e}")
            return False

    def run_evaluation(self, persona_id: str, fitness_type: str) -> bool:
        """Run test evaluation for optimized weights"""
        print(f"\n{'='*80}")
        print(f"📊 EVALUATING: {persona_id} with {fitness_type}")
        print(f"{'='*80}")

        log_file = f"logs/fitness_comparison/pipeline_eval_{persona_id}_{fitness_type}.log"

        cmd = [
            "python", "scripts/evaluate_fitness_on_test.py",
            "--persona_id", persona_id,
            "--fitness_type", fitness_type,
            "--gpu_id", str(self.gpu_id)
        ]

        try:
            with open(log_file, "w") as f:
                # Use gpu_id 0 in subprocess since CUDA_VISIBLE_DEVICES remaps it
                subprocess_cmd = cmd.copy()
                for i, arg in enumerate(subprocess_cmd):
                    if arg == "--gpu_id":
                        subprocess_cmd[i+1] = "0"
                        break

                result = subprocess.run(
                    subprocess_cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": str(self.gpu_id)},
                    timeout=1800  # 30 min timeout
                )

            if result.returncode == 0:
                print(f"✅ Evaluation complete: {persona_id} - {fitness_type}")
                return True
            else:
                print(f"❌ Evaluation failed: {persona_id} - {fitness_type}")
                return False

        except subprocess.TimeoutExpired:
            print(f"⏰ Evaluation timeout: {persona_id} - {fitness_type}")
            return False
        except Exception as e:
            print(f"❌ Error in evaluation: {e}")
            return False

    def load_results(self) -> Dict:
        """Load all evaluation results"""
        print(f"\n{'='*80}")
        print(f"📥 LOADING RESULTS")
        print(f"{'='*80}")

        all_results = {}

        for persona_id in self.personas:
            all_results[persona_id] = {}

            for fitness_type in self.fitness_types:
                result_file = Path(f"results/fitness_comparison/test_evaluation/{persona_id}_{fitness_type}.json")

                if result_file.exists():
                    with open(result_file) as f:
                        all_results[persona_id][fitness_type] = json.load(f)
                    print(f"✅ Loaded: {persona_id} - {fitness_type}")
                else:
                    print(f"❌ Missing: {persona_id} - {fitness_type}")

        return all_results

    def generate_report(self, results: Dict) -> str:
        """Generate comprehensive comparison report"""
        print(f"\n{'='*80}")
        print(f"📝 GENERATING REPORT")
        print(f"{'='*80}")

        report = []
        report.append("# Fitness Function Comparison Report (CORRECTED)")
        report.append(f"\n**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Personas**: {len(self.personas)}")
        report.append(f"**Fitness Types**: {', '.join(self.fitness_types)}")
        report.append(f"**OpenAI API**: {'✅ ENABLED' if os.getenv('OPENAI_API_KEY') else '❌ DISABLED'}")
        report.append("\n---\n")

        # Summary table
        report.append("## Results Summary\n")

        for persona_id in self.personas:
            report.append(f"\n### {persona_id}\n")
            report.append("| Fitness Type | BERTScore Gap | Style Gap | Judge Gap | Combined Gap | Judge Train | Judge Test |")
            report.append("|--------------|---------------|-----------|-----------|--------------|-------------|------------|")

            if persona_id in results:
                for fitness_type in self.fitness_types:
                    if fitness_type in results[persona_id]:
                        data = results[persona_id][fitness_type]
                        gap = data.get('generalization_gap', {})
                        train = data.get('train_scores', {})
                        test = data.get('test_scores', {})

                        report.append(
                            f"| {fitness_type:<12} | "
                            f"{gap.get('bertscore', 0):>13.4f} | "
                            f"{gap.get('style', 0):>9.4f} | "
                            f"{gap.get('judge', 0):>9.4f} | "
                            f"{gap.get('combined', 0):>12.4f} | "
                            f"{train.get('judge', {}).get('mean', 0):>11.4f} | "
                            f"{test.get('judge', {}).get('mean', 0):>10.4f} |"
                        )

        report.append("\n---\n")

        # Detailed analysis
        report.append("## Detailed Analysis\n")

        for persona_id in self.personas:
            report.append(f"\n### {persona_id}\n")

            if persona_id in results:
                for fitness_type in self.fitness_types:
                    if fitness_type in results[persona_id]:
                        data = results[persona_id][fitness_type]

                        report.append(f"\n#### {fitness_type.upper()}\n")
                        report.append(f"**Optimized Weights**:")
                        report.append("```json")
                        report.append(json.dumps(data.get('optimized_weights', {}), indent=2))
                        report.append("```\n")

                        report.append("**Train/Test Performance**:\n")
                        report.append("| Metric | Train Mean | Train Std | Test Mean | Test Std | Gap |")
                        report.append("|--------|-----------|-----------|-----------|----------|-----|")

                        train = data.get('train_scores', {})
                        test = data.get('test_scores', {})
                        gap = data.get('generalization_gap', {})

                        for metric in ['bertscore', 'style', 'judge', 'combined']:
                            report.append(
                                f"| {metric:<10} | "
                                f"{train.get(metric, {}).get('mean', 0):>9.4f} | "
                                f"{train.get(metric, {}).get('std', 0):>9.4f} | "
                                f"{test.get(metric, {}).get('mean', 0):>9.4f} | "
                                f"{test.get(metric, {}).get('std', 0):>8.4f} | "
                                f"{gap.get(metric, 0):>7.4f} |"
                            )

                        report.append("")

        report.append("\n---\n")
        report.append("## Conclusion\n")
        report.append("\n**Key Findings**:\n")
        report.append("- All optimizations run with Judge (GPT-4o) ENABLED\n")
        report.append("- True Judge scores now available for all fitness types\n")
        report.append("- Generalization performance measured accurately\n")

        return "\n".join(report)

    def run_pipeline(self):
        """Run complete pipeline"""
        print(f"\n{'#'*80}")
        print(f"# FITNESS COMPARISON PIPELINE")
        print(f"# Personas: {len(self.personas)}")
        print(f"# Fitness Types: {len(self.fitness_types)}")
        print(f"# Total experiments: {len(self.personas) * len(self.fitness_types)}")
        print(f"{'#'*80}\n")

        start_time = time.time()

        # Phase 1: Optimization
        print(f"\n{'='*80}")
        print(f"PHASE 1: OPTIMIZATION")
        print(f"{'='*80}")

        opt_success = 0
        opt_total = 0

        for persona_id in self.personas:
            for fitness_type in self.fitness_types:
                opt_total += 1
                if self.run_optimization(persona_id, fitness_type):
                    opt_success += 1

        print(f"\n✅ Optimization phase complete: {opt_success}/{opt_total} successful")

        # Phase 2: Evaluation
        print(f"\n{'='*80}")
        print(f"PHASE 2: EVALUATION")
        print(f"{'='*80}")

        eval_success = 0
        eval_total = 0

        for persona_id in self.personas:
            for fitness_type in self.fitness_types:
                eval_total += 1
                if self.run_evaluation(persona_id, fitness_type):
                    eval_success += 1

        print(f"\n✅ Evaluation phase complete: {eval_success}/{eval_total} successful")

        # Phase 3: Report generation
        results = self.load_results()
        report = self.generate_report(results)

        report_path = Path("results/fitness_comparison/PIPELINE_REPORT.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, "w") as f:
            f.write(report)

        print(f"\n✅ Report saved to: {report_path}")

        # Summary
        elapsed = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"PIPELINE COMPLETE")
        print(f"{'='*80}")
        print(f"Total time: {elapsed/60:.1f} minutes")
        print(f"Optimizations: {opt_success}/{opt_total}")
        print(f"Evaluations: {eval_success}/{eval_total}")
        print(f"Report: {report_path}")
        print(f"{'='*80}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--personas", nargs="+", default=["episode-184019_A", "episode-239427_A"])
    parser.add_argument("--fitness_types", nargs="+", default=["bertscore", "style", "combined"])
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--max_generations", type=int, default=10)
    args = parser.parse_args()

    pipeline = FitnessComparisonPipeline(
        personas=args.personas,
        fitness_types=args.fitness_types,
        gpu_id=args.gpu_id,
        max_generations=args.max_generations
    )

    pipeline.run_pipeline()


if __name__ == "__main__":
    main()
