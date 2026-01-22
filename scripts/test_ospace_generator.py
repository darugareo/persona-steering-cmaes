#!/usr/bin/env python3
"""
Test O-space steering in generator
"""

from persona_opt.generator import PersonaGenerator
import sys

print("="*80)
print("O-SPACE GENERATOR TEST")
print("="*80)

# Initialize generator
gen = PersonaGenerator(model="mock")

# Test 1: Semantic traits (should auto-convert to O-space)
print("\nTest 1: Semantic traits → O-space steering")
semantic_traits = {
    "R1": 0.5,   # Self-other focus
    "R2": 0.3,   # Expressiveness
    "R3": -0.2,  # Assertiveness
    "R4": 0.4,   # Planning
    "R5": 0.6,   # Outlook
    "R8": 0.1    # Time orientation
}

prompt = "How do I learn Python?"
response = gen.generate(prompt, semantic_traits)

print(f"Input traits (semantic): {semantic_traits}")
print(f"\nGenerated response:\n{response[:500]}...")

# Test 2: Direct O-space traits
print("\n" + "="*80)
print("Test 2: Direct O-space traits")
ospace_traits = {
    "O1": 0.5,
    "O2": -0.3,
    "O3": 0.8,
    "O4": -0.1,
    "O5": 0.4,
    "O6": 0.2
}

response2 = gen.generate(prompt, ospace_traits)
print(f"Input traits (O-space): {ospace_traits}")
print(f"\nGenerated response:\n{response2[:500]}...")

print("\n" + "="*80)
print("✓ O-space steering test completed successfully!")
print("="*80)
