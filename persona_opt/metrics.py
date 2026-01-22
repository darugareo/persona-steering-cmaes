"""
Metrics Module
Evaluation metrics for trait optimization experiments
"""

import json
import math
import re
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def wilson_score_interval(
    x: int,
    n: int,
    z: float = 1.96
) -> Tuple[float, float, float]:
    """
    Calculate Wilson score confidence interval

    Args:
        x: Number of successes
        n: Total trials
        z: Z-score for confidence level (1.96 for 95%)

    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound)
    """
    if n == 0:
        return (float('nan'), float('nan'), float('nan'))

    phat = x / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom

    return phat, center - half, center + half


def compute_win_rates(records: List[Dict]) -> Dict:
    """
    Compute win rates with confidence intervals

    Args:
        records: List of judgment records

    Returns:
        Dictionary with win rate statistics
    """
    n = len(records)
    if n == 0:
        return {
            "n_total": 0,
            "n_wins": 0,
            "n_ties": 0,
            "win_rate_tie05": float('nan'),
            "win_rate_excl_tie": float('nan'),
            "ci95_tie05": (float('nan'), float('nan')),
            "ci95_excl_tie": (float('nan'), float('nan'))
        }

    # Count preferences
    wins = 0
    ties = 0
    losses = 0

    for r in records:
        # Extract preference from various possible keys
        pref = (r.get('preference_gpt4o') or r.get('preference') or 'TIE').upper()

        if pref == 'B':
            wins += 1
        elif pref == 'TIE':
            ties += 1
        elif pref == 'A':
            losses += 1

    # Win rate with TIE counting as 0.5
    wins_with_half_tie = wins + 0.5 * ties
    wr_tie05, ci_lower_tie05, ci_upper_tie05 = wilson_score_interval(int(wins_with_half_tie), n)

    # Win rate excluding TIEs
    n_eff = wins + losses  # Effective sample size (no TIEs)
    if n_eff > 0:
        wr_excl_tie, ci_lower_excl, ci_upper_excl = wilson_score_interval(wins, n_eff)
    else:
        wr_excl_tie = float('nan')
        ci_lower_excl = ci_upper_excl = float('nan')

    return {
        "n_total": n,
        "n_wins": wins,
        "n_losses": losses,
        "n_ties": ties,
        "tie_rate": ties / n,
        "win_rate_tie05": wr_tie05,
        "win_rate_excl_tie": wr_excl_tie,
        "ci95_tie05": (ci_lower_tie05, ci_upper_tie05),
        "ci95_excl_tie": (ci_lower_excl, ci_upper_excl)
    }


def compute_content_consistency(records: List[Dict]) -> Dict:
    """
    Compute A/B swap consistency

    Args:
        records: List of judgment records

    Returns:
        Dictionary with consistency statistics
    """
    # Group by content pair
    pair_map = defaultdict(list)

    def content_hash(text: str) -> str:
        """Simple hash for content comparison"""
        import hashlib
        return hashlib.sha1((text or '').encode('utf-8')).hexdigest()

    for r in records:
        # Extract responses
        A = r.get('A') or r.get('A_text') or r.get('response_A') or ''
        B = r.get('B') or r.get('B_text') or r.get('response_B') or ''
        pref = (r.get('preference_gpt4o') or r.get('preference') or 'TIE').upper()

        # Create pair key (order-independent)
        hash_A = content_hash(A)
        hash_B = content_hash(B)
        key = '::'.join(sorted([hash_A, hash_B]))

        # Store which response won
        if pref == 'A':
            winner = hash_A
        elif pref == 'B':
            winner = hash_B
        else:
            winner = 'TIE'

        pair_map[key].append((hash_A, hash_B, winner))

    # Count consistent pairs
    pairs_considered = 0
    content_consistent = 0

    for evaluations in pair_map.values():
        # Only consider pairs evaluated multiple times
        if len(evaluations) < 2:
            continue

        # Extract winners (excluding TIEs for this analysis)
        winners = [winner for _, _, winner in evaluations if winner != 'TIE']

        if len(winners) < 2:
            continue

        pairs_considered += 1

        # Check if all non-TIE judgments agree
        if len(set(winners)) == 1:
            content_consistent += 1

    consistency_rate = content_consistent / pairs_considered if pairs_considered > 0 else float('nan')

    return {
        "pairs_considered": pairs_considered,
        "content_consistent": content_consistent,
        "consistency_rate": consistency_rate
    }


def compute_trait_alignment(records: List[Dict]) -> Dict:
    """
    Compute trait direction alignment

    Checks if generated responses move in the intended trait direction

    Args:
        records: List of judgment records with trait vectors

    Returns:
        Dictionary with alignment statistics
    """
    # Define keyword sets for trait detection
    hedge_words = {'maybe', 'might', 'could', 'perhaps', 'suggest', 'consider', 'possibly'}
    pos_words = {'good', 'great', 'helpful', 'clear', 'effective', 'excellent'}
    neg_words = {'bad', 'unclear', 'risky', 'wrong', 'poor', 'unsafe'}
    we_words = {'we', 'together', 'team', 'collaborate', 'our', 'us'}
    solo_words = {'alone', 'individually', 'solo', 'personal', 'myself'}
    you_words = {'you', 'your', 'yours'}
    impersonal_words = {'one', 'people', 'users', 'someone'}

    def score_text(text: str) -> Dict[str, float]:
        """Simple heuristic scoring of text for traits"""
        t = text.lower()
        tokens = re.findall(r'[a-z]+', t)
        n = max(len(tokens), 1)
        token_counts = Counter(tokens)

        scores = {
            'directness': 1.0 - sum(token_counts[w] for w in hedge_words) / n,
            'emotional_valence': (sum(token_counts[w] for w in pos_words) -
                                 sum(token_counts[w] for w in neg_words)) / n,
            'social_orientation': (sum(token_counts[w] for w in we_words) -
                                  sum(token_counts[w] for w in solo_words)) / n,
            'audience_focus': (sum(token_counts[w] for w in you_words) -
                              sum(token_counts[w] for w in impersonal_words)) / n,
        }
        return scores

    # Trait ID to name mapping
    trait_map = {
        'R1': 'directness',
        'R2': 'emotional_valence',
        'R3': 'social_orientation',
        'R4': 'audience_focus',
        'R5': 'risk_orientation'  # Not scored by this heuristic
    }

    aligned = 0
    checked = 0

    for r in records:
        # Get trait weights
        weights = r.get('weights') or r.get('traits') or r.get('w') or {}
        if not weights:
            continue

        # Get responses
        A = r.get('A') or r.get('response_A') or ''
        B = r.get('B') or r.get('response_B') or ''
        if not A or not B:
            continue

        # Score both responses
        score_A = score_text(A)
        score_B = score_text(B)

        # Check alignment for each trait
        is_aligned = True
        for trait_id, target_value in weights.items():
            if trait_id not in trait_map:
                continue

            trait_name = trait_map[trait_id]
            if trait_name not in score_A:
                continue

            # Skip near-zero targets
            if abs(target_value) < 0.05:
                continue

            # Check if B moved in the correct direction vs A
            diff = score_B[trait_name] - score_A[trait_name]
            target_sign = 1 if target_value > 0 else -1
            diff_sign = 1 if diff > 0 else (-1 if diff < 0 else 0)

            if diff_sign != target_sign and diff_sign != 0:
                is_aligned = False
                break

        checked += 1
        if is_aligned:
            aligned += 1

    alignment_rate = aligned / checked if checked > 0 else float('nan')

    return {
        "checked": checked,
        "aligned": aligned,
        "alignment_rate": alignment_rate
    }


def analyze_run(run_dir: Path, verbose: bool = True) -> Dict:
    """
    Analyze a complete optimization run

    Args:
        run_dir: Path to run directory (e.g., logs/run_20241125_123456)
        verbose: Print detailed output

    Returns:
        Dictionary with all metrics
    """
    # Load all records
    records = []
    for jsonl_file in sorted(run_dir.glob('gen*.jsonl')):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    # Flatten judgments if nested
                    if 'judgments' in record:
                        for j in record['judgments']:
                            records.append({**record, **j})
                    else:
                        records.append(record)

    if not records:
        if verbose:
            print(f"No records found in {run_dir}")
        return {}

    # Compute metrics
    win_rate_metrics = compute_win_rates(records)
    consistency_metrics = compute_content_consistency(records)
    alignment_metrics = compute_trait_alignment(records)

    # Load final results if available
    final_results_path = run_dir / 'final_results.json'
    if final_results_path.exists():
        with open(final_results_path, 'r', encoding='utf-8') as f:
            final_results = json.load(f)
    else:
        final_results = {}

    # Combine all metrics
    metrics = {
        "run_name": run_dir.name,
        "n_records": len(records),
        **win_rate_metrics,
        **consistency_metrics,
        **alignment_metrics,
        "final_results": final_results
    }

    # Print summary if verbose
    if verbose:
        print(f"\n{'='*60}")
        print(f"Analysis: {run_dir.name}")
        print(f"{'='*60}")
        print(f"\n📊 Win Rate Metrics:")
        print(f"  Total evaluations: {metrics['n_total']}")
        print(f"  Wins: {metrics['n_wins']}, Losses: {metrics['n_losses']}, Ties: {metrics['n_ties']}")
        print(f"  TIE rate: {metrics['tie_rate']:.3f}")
        print(f"\n  Win rate (TIE=0.5): {metrics['win_rate_tie05']:.3f}")
        print(f"    95% CI: [{metrics['ci95_tie05'][0]:.3f}, {metrics['ci95_tie05'][1]:.3f}]")
        print(f"\n  Win rate (excl TIE): {metrics['win_rate_excl_tie']:.3f}")
        print(f"    95% CI: [{metrics['ci95_excl_tie'][0]:.3f}, {metrics['ci95_excl_tie'][1]:.3f}]")

        print(f"\n🔄 Content Consistency:")
        print(f"  Pairs considered: {metrics['pairs_considered']}")
        print(f"  Consistent: {metrics['content_consistent']}")
        print(f"  Consistency rate: {metrics['consistency_rate']:.3f}")

        print(f"\n🎯 Trait Direction Alignment:")
        print(f"  Checked: {metrics['checked']}")
        print(f"  Aligned: {metrics['aligned']}")
        print(f"  Alignment rate: {metrics['alignment_rate']:.3f}")

        print(f"\n{'='*60}\n")

    return metrics


def main():
    """Analyze the most recent run"""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze optimization run metrics")
    parser.add_argument('--run_dir', type=str, help='Specific run directory to analyze')
    parser.add_argument('--logs_dir', type=str, default='logs', help='Logs directory')
    args = parser.parse_args()

    logs_path = Path(args.logs_dir)

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        # Find most recent run
        run_dirs = sorted(logs_path.glob('run_*'))
        if not run_dirs:
            print(f"No run directories found in {logs_path}")
            return
        run_dir = run_dirs[-1]

    metrics = analyze_run(run_dir, verbose=True)

    # Save metrics to file
    metrics_file = run_dir / 'metrics.json'
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Metrics saved to: {metrics_file}")


if __name__ == "__main__":
    main()
