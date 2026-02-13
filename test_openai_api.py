#!/usr/bin/env python3
"""
Simple test script to verify OpenAI API configuration
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env (override existing environment variables)
load_dotenv(override=True)

try:
    from openai import OpenAI

    # Check if API key is loaded
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        exit(1)

    print(f"✓ API key loaded: {api_key[:20]}...{api_key[-4:]}")

    # Initialize client
    client = OpenAI(api_key=api_key)
    print("✓ OpenAI client initialized")

    # Make a simple test call
    print("\nTesting API connection...")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Say 'Hello from OpenAI API!' in one sentence."}
        ],
        max_tokens=50,
        temperature=0.7
    )

    result = response.choices[0].message.content
    print(f"✓ API test successful!")
    print(f"\nResponse: {result}")

except ImportError:
    print("❌ OpenAI package not installed. Run: pip install openai")
    exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

print("\n✓ All tests passed! OpenAI API is configured correctly.")
