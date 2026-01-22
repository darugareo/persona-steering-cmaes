"""
Test script to verify Phase 1 Step 1 implementation.
Validates config loading, seed reproducibility, and prompt templates.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from persona_opt.utils.config_loader import (
    load_experiment_config,
    PromptTemplateManager,
    set_seed,
    create_output_directories,
    ExperimentLogger
)
import torch


def test_config_loading():
    """Test 1: Configuration loading."""
    print("=" * 80)
    print("TEST 1: Configuration Loading")
    print("=" * 80)

    try:
        config = load_experiment_config("config/experiment_config.yaml")
        print(f"✓ Config loaded successfully")
        print(f"  - Experiment: {config.name} v{config.version}")
        print(f"  - Seeds: {config.seeds}")
        print(f"  - Model: {config.model_name}")
        print(f"  - Generation: temp={config.temperature}, max_tokens={config.max_new_tokens}")
        print(f"  - Steering: layer={config.default_layer}, alpha={config.default_alpha}")
        print(f"  - Judges: {config.primary_judge}, {config.secondary_judge}")
        return True
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False


def test_seed_reproducibility():
    """Test 2: Seed-based reproducibility."""
    print("\n" + "=" * 80)
    print("TEST 2: Seed Reproducibility")
    print("=" * 80)

    try:
        # Test with seed 1
        set_seed(1, deterministic=True)
        rand1_a = torch.rand(5)
        rand1_b = torch.rand(5)

        # Reset seed 1
        set_seed(1, deterministic=True)
        rand1_c = torch.rand(5)

        # Test with seed 2
        set_seed(2, deterministic=True)
        rand2_a = torch.rand(5)

        # Verify
        assert torch.allclose(rand1_a, rand1_c), "Seed reproducibility failed"
        assert not torch.allclose(rand1_a, rand2_a), "Different seeds should produce different results"

        print(f"✓ Seed reproducibility verified")
        print(f"  - Seed 1 (first):  {rand1_a[:3].tolist()}")
        print(f"  - Seed 1 (repeat): {rand1_c[:3].tolist()}")
        print(f"  - Seed 2 (different): {rand2_a[:3].tolist()}")
        return True
    except Exception as e:
        print(f"✗ Seed reproducibility test failed: {e}")
        return False


def test_prompt_templates():
    """Test 3: Prompt template system."""
    print("\n" + "=" * 80)
    print("TEST 3: Prompt Template System")
    print("=" * 80)

    try:
        manager = PromptTemplateManager("config/prompt_templates.yaml")

        # Test system prompt
        system_default = manager.get_system_prompt("default")
        print(f"✓ System prompt (default):")
        print(f"  {system_default[:80]}...")

        # Test persona-injected prompt
        system_persona = manager.get_system_prompt(
            "persona_injected",
            persona_description="Friendly, empathetic, and informal"
        )
        print(f"\n✓ System prompt (persona_injected):")
        print(f"  {system_persona[:80]}...")

        # Test judge prompt
        judge_prompt = manager.get_judge_prompt(
            "persona_fit",
            persona_id="test_persona",
            traits="R1: 2.0, R2: -1.5",
            description="Friendly and casual",
            prompt="Hello, how are you?",
            response_a="I'm doing well, thank you.",
            response_b="Hey! I'm good, how about you?"
        )
        print(f"\n✓ Judge prompt (persona_fit):")
        print(f"  Length: {len(judge_prompt)} characters")

        # Test evaluation prompts
        social_prompts = manager.get_evaluation_prompts("social")
        print(f"\n✓ Evaluation prompts (social): {len(social_prompts)} prompts")
        print(f"  - Example: {social_prompts[0]}")

        return True
    except Exception as e:
        print(f"✗ Prompt template test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_output_directories():
    """Test 4: Output directory creation."""
    print("\n" + "=" * 80)
    print("TEST 4: Output Directory Structure")
    print("=" * 80)

    try:
        config = load_experiment_config("config/experiment_config.yaml")
        dirs = create_output_directories(config)

        print(f"✓ Output directories created:")
        for name, path in dirs.items():
            exists = path.exists()
            status = "✓" if exists else "✗"
            print(f"  {status} {name}: {path}")

        return all(path.exists() for path in dirs.values())
    except Exception as e:
        print(f"✗ Output directory test failed: {e}")
        return False


def test_experiment_logger():
    """Test 5: Experiment logging system."""
    print("\n" + "=" * 80)
    print("TEST 5: Experiment Logger")
    print("=" * 80)

    try:
        config = load_experiment_config("config/experiment_config.yaml")
        logger = ExperimentLogger(config, seed=1)

        # Log some test results
        logger.log_result("baseline", "train_test", {
            "train_score": 3.86,
            "test_score": 3.67,
            "gap": 0.19
        })

        logger.log_metric("total_prompts", 10)
        logger.log_error("Test error (ignore)")

        print(f"✓ Logger created and tested")
        print(f"  - Log file: {logger.log_file}")
        print(f"  - Results logged: {len(logger.log_data['results'])}")
        print(f"  - Metrics logged: {len(logger.log_data['metrics'])}")

        return True
    except Exception as e:
        print(f"✗ Experiment logger test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generation_kwargs():
    """Test 6: Generation parameters."""
    print("\n" + "=" * 80)
    print("TEST 6: Generation Parameters")
    print("=" * 80)

    try:
        config = load_experiment_config("config/experiment_config.yaml")
        gen_kwargs = config.get_generation_kwargs()

        print(f"✓ Generation kwargs extracted:")
        for key, value in gen_kwargs.items():
            print(f"  - {key}: {value}")

        # Verify critical parameters
        assert gen_kwargs['temperature'] == 0.0, "Temperature should be 0.0 for greedy"
        assert gen_kwargs['do_sample'] == False, "do_sample should be False for greedy"
        assert gen_kwargs['max_new_tokens'] == 128, "max_new_tokens should be 128"

        print(f"\n✓ All critical parameters verified")
        return True
    except Exception as e:
        print(f"✗ Generation kwargs test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("PHASE 1 STEP 1: EXPERIMENT SETUP VALIDATION")
    print("=" * 80 + "\n")

    tests = [
        ("Config Loading", test_config_loading),
        ("Seed Reproducibility", test_seed_reproducibility),
        ("Prompt Templates", test_prompt_templates),
        ("Output Directories", test_output_directories),
        ("Experiment Logger", test_experiment_logger),
        ("Generation Kwargs", test_generation_kwargs),
    ]

    results = {}
    for name, test_func in tests:
        results[name] = test_func()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(results.values())
    total = len(results)

    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ Phase 1 Step 1 COMPLETE - All systems operational")
        return 0
    else:
        print("\n✗ Some tests failed - Please review errors above")
        return 1


if __name__ == "__main__":
    exit(main())
