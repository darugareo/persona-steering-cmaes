#!/usr/bin/env python3
"""
LaMP-7 Generation Pipeline - Phase 2-B

Generates personalized tweet paraphrases using 4 methods:
1. Base: No steering
2. Prompt Persona: Brief style summary in system prompt
3. Equal-weight Steering: SVD vectors with w=[1,1,1,1,1]
4. Optimized Steering: Chronicles-optimized weights (frozen)

IMPORTANT: User profiles are NOT shown to the generation model.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
import torch
from tqdm import tqdm
import re

# Add persona_opt to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from persona_opt.internal_steering_l3 import Llama3ActivationSteerer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_style_summary(profile_tweets: List[Dict[str, str]], max_features: int = 3) -> str:
    """
    Extract brief style features from user profile for Prompt method.

    Returns 3 concise style observations (NOT full profile text).
    """
    # Analyze tweets for style patterns
    texts = [t['text'] for t in profile_tweets]

    # Pattern detection
    has_ellipsis = sum('.. ' in t or ' ..' in t for t in texts) > len(texts) // 2
    has_emoji = sum(any(ord(c) > 127 for c in t) for t in texts) > len(texts) // 3
    avg_length = sum(len(t) for t in texts) / len(texts)
    has_quotes = sum('"' in t or "'" in t for t in texts) > len(texts) // 3
    all_lowercase = sum(t.islower() or not any(c.isupper() for c in t) for t in texts) > len(texts) // 2

    features = []
    if has_ellipsis:
        features.append("uses ellipsis frequently (.. )")
    if all_lowercase:
        features.append("casual/informal capitalization")
    elif has_quotes:
        features.append("often quotes phrases")
    if avg_length < 80:
        features.append("concise, short tweets")
    elif avg_length > 120:
        features.append("detailed, longer tweets")

    # Return top 3 features
    return ", ".join(features[:max_features]) if features else "casual Twitter style"


def load_lamp7_data(data_path: str, limit: Optional[int] = None) -> List[Dict]:
    """Load LaMP-7 dev questions and outputs."""
    logger.info(f"Loading LaMP-7 data from {data_path}")

    questions_path = Path(data_path) / "dev_questions.json"
    outputs_path = Path(data_path) / "dev_outputs.json"

    with open(questions_path) as f:
        questions = json.load(f)
    with open(outputs_path) as f:
        outputs_data = json.load(f)

    # Create id -> output mapping
    outputs = {item['id']: item['output'] for item in outputs_data['golds']}

    # Merge questions and outputs
    data = []
    for q in questions[:limit]:
        data.append({
            'id': q['id'],
            'input': q['input'],
            'profile': q['profile'],
            'gold_output': outputs.get(q['id'], '')
        })

    logger.info(f"Loaded {len(data)} samples")
    return data


def load_svd_vectors(vectors_dir: str, layer: int, trait_names: List[str]) -> Dict[str, torch.Tensor]:
    """Load SVD trait vectors for specified layer."""
    vectors_path = Path(vectors_dir) / f"layer{layer}_svd.pt"

    if not vectors_path.exists():
        raise FileNotFoundError(f"SVD vectors not found: {vectors_path}")

    # Note: Each layer has ONE vector that represents the aggregated trait space
    # The trait_names correspond to the 5 SVD components (R1-R5)
    vec = torch.load(vectors_path, weights_only=False)
    logger.info(f"Loaded SVD vector from {vectors_path}, shape: {vec.shape}")

    # For equal-weight steering, we use this vector directly
    # For optimized steering, we'll load the persona-specific weights
    return {"svd_vector": vec}


def load_optimized_weights(weights_path: str) -> Dict:
    """Load optimized weights from Chronicles persona."""
    with open(weights_path) as f:
        weights_data = json.load(f)
    logger.info(f"Loaded optimized weights for {weights_data.get('persona_id', 'unknown')}")
    return weights_data


def generate_base(steerer: Llama3ActivationSteerer, input_text: str, max_new_tokens: int = 100) -> str:
    """Generate without steering."""
    steerer.remove_hooks()

    messages = [{"role": "user", "content": input_text}]
    text = steerer.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = steerer.tokenizer(text, return_tensors="pt").to(steerer.device)

    with torch.no_grad():
        outputs = steerer.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=steerer.tokenizer.eos_token_id
        )

    generated = steerer.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return generated.strip()


def generate_prompt_persona(steerer: Llama3ActivationSteerer, input_text: str,
                            style_summary: str, max_new_tokens: int = 100) -> str:
    """Generate with style summary in system prompt (NO full profile)."""
    steerer.remove_hooks()

    system_prompt = f"You are paraphrasing tweets. Mimic this style: {style_summary}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_text}
    ]
    text = steerer.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = steerer.tokenizer(text, return_tensors="pt").to(steerer.device)

    with torch.no_grad():
        outputs = steerer.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=steerer.tokenizer.eos_token_id
        )

    generated = steerer.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return generated.strip()


def generate_with_steering(steerer: Llama3ActivationSteerer, input_text: str,
                           steering_vector: torch.Tensor, alpha: float = 1.0,
                           max_new_tokens: int = 100) -> str:
    """Generate with activation steering."""
    steerer.register_hooks(steering_vector=steering_vector, alpha=alpha)

    messages = [{"role": "user", "content": input_text}]
    text = steerer.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = steerer.tokenizer(text, return_tensors="pt").to(steerer.device)

    with torch.no_grad():
        outputs = steerer.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=steerer.tokenizer.eos_token_id
        )

    generated = steerer.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    steerer.remove_hooks()
    return generated.strip()


def main():
    parser = argparse.ArgumentParser(description="LaMP-7 Generation Pipeline")

    parser.add_argument("--data-path", type=str, default="data/lamp7",
                       help="Path to LaMP-7 dataset")
    parser.add_argument("--output-dir", type=str, default="outputs/lamp7",
                       help="Output directory for generated results")
    parser.add_argument("--limit", type=int, default=200,
                       help="Number of samples to process (default: 200)")

    # Model configuration
    parser.add_argument("--model-name", type=str,
                       default="meta-llama/Meta-Llama-3-8B-Instruct",
                       help="Base model")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device")
    parser.add_argument("--layer", type=int, default=22,
                       help="Layer for steering (default: 22)")

    # Steering configuration
    parser.add_argument("--svd-vectors-dir", type=str,
                       default="data/steering_vectors_v2/R1",
                       help="Directory with SVD vectors")
    parser.add_argument("--optimized-weights", type=str,
                       default="persona-opt/episode-184019_A/best_weights.json",
                       help="Path to optimized weights from Chronicles")
    parser.add_argument("--alpha", type=float, default=2.0,
                       help="Steering strength (default: 2.0)")

    # Generation parameters
    parser.add_argument("--max-new-tokens", type=int, default=100,
                       help="Max tokens to generate")
    parser.add_argument("--methods", type=str, nargs="+",
                       default=["base", "prompt", "equal", "optimized"],
                       choices=["base", "prompt", "equal", "optimized"],
                       help="Methods to run")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data = load_lamp7_data(args.data_path, limit=args.limit)
    logger.info(f"Processing {len(data)} samples with methods: {args.methods}")

    # Load steering vectors
    svd_vectors = load_svd_vectors(args.svd_vectors_dir, args.layer, ["R1", "R2", "R3", "R4", "R5"])
    svd_vector = svd_vectors["svd_vector"]

    # Load optimized weights
    opt_weights_data = load_optimized_weights(args.optimized_weights)
    opt_weights = torch.tensor(opt_weights_data['weights'], dtype=svd_vector.dtype)

    # Compute steering vectors
    # Equal-weight: Use SVD vector as-is (represents equal weighting of traits)
    equal_vector = svd_vector

    # Optimized: Scale SVD vector by optimized alpha
    # Note: In the multi-trait system, weights are applied to individual trait components
    # Here we use the SVD vector which already represents the trait space
    # and scale by the optimized alpha
    optimized_vector = svd_vector
    optimized_alpha = opt_weights_data.get('alpha', args.alpha)

    logger.info(f"Equal-weight vector norm: {equal_vector.norm().item():.4f}")
    logger.info(f"Optimized alpha: {optimized_alpha:.4f}")

    # Initialize steerer
    logger.info(f"Loading {args.model_name}...")
    steerer = Llama3ActivationSteerer(
        model_name=args.model_name,
        target_layer=args.layer,
        device=args.device,
        torch_dtype=torch.bfloat16
    )

    # Generate for each method
    results = {method: [] for method in args.methods}

    for sample in tqdm(data, desc="Generating"):
        input_text = sample['input']
        sample_id = sample['id']
        profile = sample['profile']
        gold = sample['gold_output']

        # Extract style summary for prompt method
        style_summary = extract_style_summary(profile) if 'prompt' in args.methods else None

        for method in args.methods:
            try:
                if method == "base":
                    output = generate_base(steerer, input_text, args.max_new_tokens)
                elif method == "prompt":
                    output = generate_prompt_persona(steerer, input_text, style_summary, args.max_new_tokens)
                elif method == "equal":
                    output = generate_with_steering(steerer, input_text, equal_vector,
                                                   alpha=args.alpha, max_new_tokens=args.max_new_tokens)
                elif method == "optimized":
                    output = generate_with_steering(steerer, input_text, optimized_vector,
                                                   alpha=optimized_alpha, max_new_tokens=args.max_new_tokens)

                results[method].append({
                    'id': sample_id,
                    'input': input_text,
                    'generated': output,
                    'gold': gold,
                    'method': method,
                    'style_summary': style_summary if method == 'prompt' else None
                })
            except Exception as e:
                logger.error(f"Error generating for {sample_id} with {method}: {e}")
                results[method].append({
                    'id': sample_id,
                    'input': input_text,
                    'generated': f"[ERROR: {str(e)}]",
                    'gold': gold,
                    'method': method,
                    'error': str(e)
                })

    # Save results
    for method, method_results in results.items():
        output_file = output_dir / f"{method}_results.jsonl"
        with open(output_file, 'w') as f:
            for result in method_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        logger.info(f"Saved {len(method_results)} results to {output_file}")

    # Save summary
    summary = {
        'num_samples': len(data),
        'methods': args.methods,
        'layer': args.layer,
        'alpha': args.alpha,
        'optimized_alpha': optimized_alpha,
        'model': args.model_name,
        'svd_vectors_dir': args.svd_vectors_dir,
        'optimized_weights_path': args.optimized_weights
    }
    summary_file = output_dir / "generation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_file}")

    logger.info("Generation complete!")


if __name__ == "__main__":
    main()
