#!/usr/bin/env python3
"""
Test prompt generation and response with new turn structure
"""
import torch
from pathlib import Path
import json
from scripts.fitness_comparison_optimizer import FitnessComparator

def test_generation():
    persona_id = "episode-184019_A"

    print("="*80)
    print("Testing Prompt Generation and Response")
    print("="*80)

    # Initialize comparator
    print("\nInitializing comparator...")
    comparator = FitnessComparator(
        persona_id=persona_id,
        fitness_type="bertscore",
        device="cuda:0"
    )

    # Test with zero weights (baseline)
    print("\n" + "="*80)
    print("TEST: Zero Weights (Baseline)")
    print("="*80)

    weights = {
        "R1": 0.0,
        "R2": 0.0,
        "R3": 0.0,
        "R4": 0.0,
        "R5": 0.0
    }

    # Test first 2 turns
    for i, turn in enumerate(comparator.train_turns[:2]):
        print(f"\n--- Turn {i+1} ---")

        context = turn["context"]
        input_text = turn["input"]
        ground_truth = turn["ground_truth"]

        # Build prompt
        prompt = f"""Continue this conversation naturally.

Conversation so far:
{context}

Partner: {input_text}

You:"""

        print(f"\n📝 Context:")
        print(f"   {context[:150]}...")
        print(f"\n💬 Partner says:")
        print(f"   {input_text}")
        print(f"\n✅ Ground truth:")
        print(f"   {ground_truth}")

        # Generate
        print(f"\n🤖 Generating response...")
        generated = comparator.generate_with_steering(prompt, weights)

        print(f"\n🔹 Generated:")
        print(f"   {generated}")

        # Calculate fitness
        bertscore = comparator.fitness_bertscore(generated, ground_truth)
        style = comparator.fitness_style(generated, ground_truth)

        print(f"\n📊 Scores:")
        print(f"   BERTScore: {bertscore:.4f}")
        print(f"   Style:     {style:.4f}")

    print("\n" + "="*80)
    print("Test complete!")
    print("="*80)

if __name__ == "__main__":
    test_generation()
