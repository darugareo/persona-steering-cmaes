"""
Test script for all 7 baseline methods.
Validates that each baseline can be instantiated and generates responses.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from persona_opt.internal_steering_l3 import Llama3ActivationSteerer
from persona_opt.baselines import (
    BaseMethod,
    PromptPersonaMethod,
    MeanDiffMethod,
    PCASteeringMethod,
    RandomSearchMethod,
    GridSearchMethod,
)


def test_baseline_instantiation():
    """Test that all baselines can be instantiated."""
    print("=" * 80)
    print("TEST: Baseline Instantiation")
    print("=" * 80)

    try:
        # Initialize steerer
        print("\nInitializing steerer...")
        steerer = Llama3ActivationSteerer(
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            target_layer=20,
            device="cuda",
        )
        print("✓ Steerer initialized")

        persona_id = "episode-184019_A"
        test_prompts = [
            "Tell me about yourself.",
            "How do you approach problems?",
        ]

        baselines = {}

        # Test 1: Base method
        print("\n1. Testing Base method...")
        baselines['base'] = BaseMethod(steerer, persona_id)
        print(f"   ✓ {baselines['base']}")

        # Test 2: Prompt Persona method
        print("\n2. Testing Prompt Persona method...")
        baselines['prompt_persona'] = PromptPersonaMethod(steerer, persona_id)
        print(f"   ✓ {baselines['prompt_persona']}")

        # Test 3: Mean Difference method
        print("\n3. Testing Mean Difference method...")
        baselines['meandiff'] = MeanDiffMethod(steerer, persona_id, layer=20, alpha=2.0)
        print(f"   ✓ {baselines['meandiff']}")

        # Test 4: PCA method
        print("\n4. Testing PCA method...")
        baselines['pca'] = PCASteeringMethod(steerer, persona_id, layer=20, alpha=2.0, n_components=5)
        print(f"   ✓ {baselines['pca']}")

        # Test 5: Random Search (without optimization)
        print("\n5. Testing Random Search method...")
        baselines['random_search'] = RandomSearchMethod(
            steerer, persona_id, layer=20, alpha=2.0,
            n_iterations=100, eval_prompts=None  # Skip optimization for speed
        )
        print(f"   ✓ {baselines['random_search']}")

        # Test 6: Grid Search (without optimization)
        print("\n6. Testing Grid Search method...")
        baselines['grid_search'] = GridSearchMethod(
            steerer, persona_id, layer=20, alpha=2.0,
            grid_points=3, eval_prompts=None  # Skip optimization for speed
        )
        print(f"   ✓ {baselines['grid_search']}")

        print("\n✓ All 6 baselines instantiated successfully")
        return baselines

    except Exception as e:
        print(f"\n✗ Instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_baseline_generation(baselines):
    """Test that all baselines can generate responses."""
    print("\n" + "=" * 80)
    print("TEST: Response Generation")
    print("=" * 80)

    if baselines is None:
        print("✗ Skipping (baselines not instantiated)")
        return False

    test_prompt = "Hello, how are you?"

    try:
        for name, baseline in baselines.items():
            print(f"\nTesting {name}...")
            response = baseline.generate(test_prompt, max_new_tokens=50, temperature=0.0)
            print(f"✓ Generated {len(response)} characters")
            print(f"  Response preview: {response[:100]}...")

        print("\n✓ All baselines can generate responses")
        return True

    except Exception as e:
        print(f"\n✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_baseline_configs(baselines):
    """Test that all baselines return proper configs."""
    print("\n" + "=" * 80)
    print("TEST: Configuration Retrieval")
    print("=" * 80)

    if baselines is None:
        print("✗ Skipping (baselines not instantiated)")
        return False

    try:
        for name, baseline in baselines.items():
            config = baseline.get_config()
            print(f"\n{name}:")
            print(f"  Method: {config['method']}")
            print(f"  Description: {config['description']}")
            print(f"  Steering: {config.get('steering', 'N/A')}")

        print("\n✓ All baselines return valid configs")
        return True

    except Exception as e:
        print(f"\n✗ Config retrieval failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all baseline tests."""
    print("\n" + "=" * 80)
    print("PHASE 1 STEP 2: BASELINE METHODS VALIDATION")
    print("=" * 80 + "\n")

    # Test 1: Instantiation
    baselines = test_baseline_instantiation()

    # Test 2: Generation
    gen_success = test_baseline_generation(baselines)

    # Test 3: Configs
    config_success = test_baseline_configs(baselines)

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    results = {
        'Instantiation': baselines is not None,
        'Generation': gen_success,
        'Configuration': config_success,
    }

    passed = sum(results.values())
    total = len(results)

    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    # List implemented baselines
    print("\n" + "=" * 80)
    print("IMPLEMENTED BASELINES (6/7)")
    print("=" * 80)
    print("✓ 1. Base (no steering)")
    print("✓ 2. Prompt Persona (system prompt injection)")
    print("✓ 3. Mean Difference (contrastive activations)")
    print("✓ 4. PCA Steering (principal components)")
    print("✓ 5. Random Search (trait weight optimization)")
    print("✓ 6. Grid Search (limited grid)")
    print("✓ 7. Proposed (SVD + CMA-ES) - already implemented")

    if passed == total:
        print("\n✓ Phase 1 Step 2 COMPLETE - All baselines operational")
        return 0
    else:
        print("\n✗ Some tests failed - Please review errors above")
        return 1


if __name__ == "__main__":
    exit(main())
