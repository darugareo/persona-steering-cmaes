"""
Test script for persona_judge_evaluator

Tests the evaluation wrapper with sample responses.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from persona_opt.persona_judge_evaluator import (
    evaluate_with_persona_judge,
    batch_evaluate,
    compute_aggregate_metrics
)


def test_single_evaluation():
    """Test single evaluation call"""
    print("=" * 60)
    print("Test 1: Single Evaluation")
    print("=" * 60)

    persona_id = "episode-184019_A"

    # Test prompt about sharing a personal experience
    prompt = "A friend asked about your weekend. What do you say?"

    # Response A: Generic, formal
    response_a = "I had a productive weekend. I completed several tasks and spent time with family. How was yours?"

    # Response B: Informal, personal anecdote (matches persona style)
    response_b = "Oh man, so I went hiking and totally got lost for like an hour! I was so confused, but eventually found my way back. It was kind of hilarious actually, haha."

    try:
        result = evaluate_with_persona_judge(
            persona_id=persona_id,
            prompt=prompt,
            response_a=response_a,
            response_b=response_b,
            trait_name="Communication Style Match",
            trait_direction="informal, anecdote-sharing style"
        )

        print("\nResult:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

        # Expected: B should win (informal, personal story, humor)
        assert result["winner"] in ["A", "B", "tie"], f"Invalid winner: {result['winner']}"
        assert 1 <= result["confidence"] <= 5, f"Invalid confidence: {result['confidence']}"
        print("\n✅ Single evaluation test passed")

    except Exception as e:
        print(f"\n❌ Single evaluation test failed: {e}")
        raise


def test_batch_evaluation():
    """Test batch evaluation"""
    print("\n" + "=" * 60)
    print("Test 2: Batch Evaluation")
    print("=" * 60)

    persona_id = "episode-239427_A"

    evaluations = [
        {
            "prompt": "Your neighbor mentioned they're struggling with a home repair. What's your response?",
            "response_a": "I suggest you hire a professional to handle it properly.",
            "response_b": "Oh no! That sounds frustrating. I actually had the same issue last month. Want me to come take a look? Maybe we can figure it out together.",
            "trait_name": "Social Support",
            "trait_direction": "empathetic, problem-sharing"
        },
        {
            "prompt": "Someone asks for your opinion on a community decision.",
            "response_a": "I think we should carefully consider all options before proceeding.",
            "response_b": "Hmm, I'm not totally sure. What do you think? I feel like we should hear from everyone first.",
            "trait_name": "Decision-Making Style",
            "trait_direction": "collaborative, other-focused"
        },
        {
            "prompt": "A friend shares good news with you.",
            "response_a": "Congratulations on your achievement.",
            "response_b": "Oh my gosh, that's amazing! I'm so happy for you! This is such great news!",
            "trait_name": "Emotional Expression",
            "trait_direction": "enthusiastic, informal"
        }
    ]

    try:
        results = batch_evaluate(
            persona_id=persona_id,
            evaluations=evaluations,
            model="gpt-4o-mini"
        )

        print(f"\nEvaluated {len(results)} pairs")

        # Compute aggregate metrics
        metrics = compute_aggregate_metrics(results)
        print("\nAggregate Metrics:")
        print(json.dumps(metrics, indent=2))

        # Show individual results
        print("\nIndividual Results:")
        for i, result in enumerate(results):
            print(f"\n{i+1}. Winner: {result['winner']}, Confidence: {result['confidence']}")
            print(f"   Fit A: {result['persona_fit_score_a']}, Fit B: {result['persona_fit_score_b']}")
            print(f"   Explanation: {result['explanation'][:100]}...")

        assert len(results) == len(evaluations), "Result count mismatch"
        print("\n✅ Batch evaluation test passed")

    except Exception as e:
        print(f"\n❌ Batch evaluation test failed: {e}")
        raise


def test_all_personas():
    """Test evaluation with all available personas"""
    print("\n" + "=" * 60)
    print("Test 3: All Personas")
    print("=" * 60)

    personas = ["episode-184019_A", "episode-239427_A", "episode-118328_B"]

    prompt = "A friend is feeling down. What do you say?"
    response_a = "I hope you feel better soon. Let me know if you need anything."
    response_b = "Aw, that sucks. Want to talk about it? I'm here if you need me."

    for persona_id in personas:
        print(f"\nTesting persona: {persona_id}")

        try:
            result = evaluate_with_persona_judge(
                persona_id=persona_id,
                prompt=prompt,
                response_a=response_a,
                response_b=response_b
            )

            print(f"  Winner: {result['winner']}")
            print(f"  Fit A: {result['persona_fit_score_a']}, Fit B: {result['persona_fit_score_b']}")
            print(f"  Confidence: {result['confidence']}")

        except FileNotFoundError as e:
            print(f"  ⚠️  Skipping (judge not found): {e}")
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            raise

    print("\n✅ All personas test completed")


def main():
    """Run all tests"""
    print("Testing Persona Judge Evaluator")
    print("=" * 60)

    # Check for API key
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  Warning: OPENAI_API_KEY not set in environment")
        print("Please set it in .env file or environment variables")
        print("\nSkipping tests that require API calls...")
        return

    try:
        test_single_evaluation()
        test_batch_evaluation()
        test_all_personas()

        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ Tests failed: {e}")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
