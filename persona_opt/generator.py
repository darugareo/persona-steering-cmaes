"""
Generator Module
Converts trait vectors into persona-conditioned LLM responses
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai package not installed. Only mock mode available.")

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers/torch not installed. HuggingFace models unavailable.")

# Import gpt-oss wrapper for Python 3.12 bridge
try:
    from persona_opt.gpt_oss_wrapper import generate_with_gpt_oss
    GPT_OSS_WRAPPER_AVAILABLE = True
except ImportError:
    GPT_OSS_WRAPPER_AVAILABLE = False
    print("Warning: gpt_oss_wrapper not available.")

# Import O-space steering conversion
try:
    from persona_opt.steering_space_v4 import semantic_to_orthogonal, get_trait_names
    OSPACE_AVAILABLE = True
except ImportError:
    OSPACE_AVAILABLE = False
    print("Warning: steering_space_v4 not available. O-space steering disabled.")


class PersonaGenerator:
    """Generates responses based on trait vectors"""

    def __init__(
        self,
        traits_config_path: str = "persona_opt/traits_v2.json",
        api_key: Optional[str] = None,
        model: str = "mock",
        max_new_tokens: int = 300,
        temperature: float = 0.7
    ):
        """
        Initialize the generator with trait configuration

        Args:
            traits_config_path: Path to traits configuration JSON
            api_key: OpenAI API key (optional, will use env var if not provided)
            model: Model to use ("mock", "gpt-4o-mini", "gpt-4o", "gpt-oss-20b-local")
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        self.traits_config_path = Path(traits_config_path)
        self.traits_config = self._load_traits_config()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Determine backend based on model
        if model.startswith("hf/"):
            # HuggingFace model: hf/meta-llama/Llama-3.1-8B-Instruct
            self.backend = "huggingface"
            self.model = model.replace("hf/", "")
            self.tokenizer = None
            self.hf_model = None
            self.client = None
        elif model == "gpt-oss-20b-local" or model.startswith("ollama/"):
            self.backend = "ollama"
            # Extract model name (remove "ollama/" prefix if present)
            if model.startswith("ollama/"):
                self.model = model.replace("ollama/", "")
            elif model == "gpt-oss-20b-local":
                self.model = "gpt-oss:20b"
            # Initialize client for Ollama server (OpenAI-compatible)
            if OPENAI_AVAILABLE:
                # Support both OLLAMA_ and GPT_OSS_ environment variables
                base_url = os.environ.get("OLLAMA_BASE_URL") or os.environ.get("GPT_OSS_BASE_URL", "http://localhost:11434/v1")
                api_key = os.environ.get("OLLAMA_API_KEY") or os.environ.get("GPT_OSS_API_KEY", "ollama")
                self.client = OpenAI(
                    base_url=base_url,
                    api_key=api_key,
                )
            else:
                self.client = None
            self.tokenizer = None
            self.hf_model = None
        else:
            self.backend = "openai"
            self.model = model
            # Initialize OpenAI client (lazy initialization)
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            self.client = None
            self.tokenizer = None
            self.hf_model = None

    def _load_traits_config(self) -> dict:
        """Load trait specifications from JSON"""
        if not self.traits_config_path.exists():
            raise FileNotFoundError(f"Traits config not found: {self.traits_config_path}")

        with open(self.traits_config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _convert_to_ospace_if_needed(self, trait_dict: Dict[str, float]) -> Dict[str, float]:
        """
        Convert semantic traits to O-space steering if available

        Args:
            trait_dict: Semantic trait dictionary {R1: ..., R2: ..., etc}

        Returns:
            O-space dict {O1: ..., O2: ..., etc} or original if O-space unavailable
        """
        if not OSPACE_AVAILABLE:
            return trait_dict

        # Extract trait values in correct order (R1, R2, R3, R4, R5, R8)
        trait_names = get_trait_names()
        semantic_vector = [trait_dict.get(name, 0.0) for name in trait_names]

        # Convert to O-space
        o_vector = semantic_to_orthogonal(semantic_vector)

        # Return as O-space dictionary
        return {f"O{i+1}": val for i, val in enumerate(o_vector)}

    def _build_persona_prompt(self, trait_dict: Dict[str, float], use_ospace: bool = True) -> str:
        """
        Build persona steering instructions from trait vector

        Args:
            trait_dict: Dictionary mapping trait IDs (R1-R5, R8 or O1-O6) to values
            use_ospace: If True and available, convert semantic → O-space first

        Returns:
            Persona instruction string
        """
        # Convert to O-space if requested and available
        if use_ospace and not any(k.startswith("O") for k in trait_dict.keys()):
            trait_dict = self._convert_to_ospace_if_needed(trait_dict)

        # Check if we're using O-space or semantic space
        is_ospace = any(k.startswith("O") for k in trait_dict.keys())

        if is_ospace:
            # O-space steering: use simple numeric format
            steering_text = ", ".join([f"{k}={v:+.3f}" for k, v in sorted(trait_dict.items())])
            persona_prompt = f"""You are adopting an orthogonal persona vector style.
Follow these 6 orthogonal factors: {steering_text}

These factors describe stylistic directions in an optimized orthogonal trait space."""
            return persona_prompt

        # Semantic space: use human-readable rules
        trait_map = {t['id']: t for t in self.traits_config['traits']}

        instructions = []
        for trait_id, value in trait_dict.items():
            if trait_id not in trait_map:
                continue

            trait_spec = trait_map[trait_id]

            # Determine which rule to apply based on sign and magnitude
            if abs(value) < 0.05:  # Near-zero, skip
                continue
            elif value > 0:
                rule = trait_spec['generator_rule_pos']
                strength = "強く" if value > 0.5 else ""
            else:
                rule = trait_spec['generator_rule_neg']
                strength = "強く" if value < -0.5 else ""

            instructions.append(f"- {trait_spec['name']}: {strength}{rule}")

        if not instructions:
            return ""

        persona_prompt = "以下のペルソナ特性に従って応答してください：\n" + "\n".join(instructions)
        return persona_prompt

    def generate(
        self,
        prompt: str,
        traits: Dict[str, float]
    ) -> str:
        """
        Generate a response conditioned on trait vector

        Args:
            prompt: User prompt/query
            traits: Trait vector {R1: 0.5, R2: -0.3, ...}

        Returns:
            Generated response text
        """
        return self.generate_response(prompt, traits, model=self.model)

    def generate_response(
        self,
        prompt: str,
        trait_dict: Dict[str, float],
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> str:
        """
        Generate a response conditioned on trait vector

        Args:
            prompt: User prompt/query
            trait_dict: Trait vector {R1: 0.5, R2: -0.3, ...}
            model: LLM model to use (overrides instance model if provided)
            api_key: OpenAI API key (if using real model)

        Returns:
            Generated response text
        """
        # Use instance model if not overridden
        if model is None:
            model = self.model

        persona_instruction = self._build_persona_prompt(trait_dict)

        if model == "mock":
            # Mock implementation for testing
            return self._generate_mock_response(prompt, trait_dict, persona_instruction)
        else:
            # Real LLM generation using Responses API
            return self._generate_llm_response(prompt, persona_instruction, model, api_key)

    def _generate_mock_response(
        self,
        prompt: str,
        trait_dict: Dict[str, float],
        persona_instruction: str
    ) -> str:
        """
        Mock generator for testing (returns formatted string with traits)
        """
        trait_str = ", ".join([f"{k}={v:.2f}" for k, v in sorted(trait_dict.items())])

        response = f"""[MOCK Response with traits: {trait_str}]

Prompt: {prompt}

Persona: {persona_instruction if persona_instruction else '(neutral)'}

Generated Response: この応答は仮の出力です。実際のLLM生成では、上記のペルソナ指示に従った応答が生成されます。"""

        return response

    def _load_hf_model(self):
        """Load HuggingFace model and tokenizer (lazy loading)"""
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers/torch not installed. Run: pip install transformers torch")

        if self.tokenizer is None or self.hf_model is None:
            print(f"Loading HuggingFace model: {self.model}")

            # Get HuggingFace token and cache dir from environment
            hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
            # Use /data02 cache where models are actually stored
            cache_dir = os.getenv("HF_HOME") or "/data02/nakata/.cache/huggingface"

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model,
                cache_dir=cache_dir,
                token=hf_token
            )

            # Use bfloat16 for A100 (better than float16, no need for MXFP4)
            # trust_remote_code needed for gpt-oss custom architecture
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                self.model,
                torch_dtype=torch.bfloat16,  # A100 optimal
                device_map="auto",
                trust_remote_code=True,  # Required for gpt-oss
                cache_dir=cache_dir,
                token=hf_token
            )
            print(f"Model loaded successfully on {self.hf_model.device}")

    def _generate_llm_response(
        self,
        prompt: str,
        persona_instruction: str,
        model: str,
        api_key: Optional[str]
    ) -> str:
        """
        Real LLM generation using HuggingFace, OpenAI, or Ollama
        """
        # HuggingFace backend
        if self.backend == "huggingface":
            return self._generate_hf_response(prompt, persona_instruction)

        # Initialize client if needed (for OpenAI backend)
        if self.client is None and self.backend == "openai":
            if not OPENAI_AVAILABLE:
                raise RuntimeError("OpenAI package not installed. Run: pip install openai")

            if not self.api_key:
                raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key parameter")

            self.client = OpenAI(api_key=self.api_key)

        # Build system message
        if persona_instruction:
            system_content = f"""You are a helpful AI assistant. Respond to the user's query with the following persona characteristics:

{persona_instruction}

Be natural and helpful while maintaining these stylistic traits."""
        else:
            system_content = "You are a helpful AI assistant."

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt}
        ]

        # Call API (Ollama and OpenAI both use chat.completions)
        try:
            # Both Ollama and OpenAI use the same chat.completions API
            response = self.client.chat.completions.create(
                model=self.model if self.backend == "ollama" else model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_new_tokens
            )

            # Handle gpt-oss reasoning field (it outputs to 'reasoning' instead of 'content')
            message = response.choices[0].message
            generated_text = message.content

            # If content is empty but reasoning exists, use reasoning
            if (not generated_text or len(generated_text.strip()) == 0) and hasattr(message, 'reasoning'):
                generated_text = message.reasoning

            # If still empty, log warning
            if not generated_text or len(generated_text.strip()) == 0:
                print(f"Warning: Empty response from {self.backend} generator")
                generated_text = ""

            return generated_text

        except Exception as e:
            error_msg = f"Error calling LLM API: {e}"
            print(error_msg)
            return f"[Error generating response: {str(e)}]"

    def _generate_hf_response(self, prompt: str, persona_instruction: str) -> str:
        """
        Generate response using HuggingFace model

        Args:
            prompt: User prompt
            persona_instruction: Persona steering instruction

        Returns:
            Generated text
        """
        try:
            # Check if this is gpt-oss model (requires Python 3.12 wrapper)
            if "gpt-oss" in self.model.lower():
                if not GPT_OSS_WRAPPER_AVAILABLE:
                    raise RuntimeError("gpt_oss_wrapper not available. Check persona_opt/gpt_oss_wrapper.py")

                # Build chat messages
                if persona_instruction:
                    system_content = f"""You are a helpful AI assistant. Respond to the user's query with the following persona characteristics:

{persona_instruction}

Be natural and helpful while maintaining these stylistic traits."""
                else:
                    system_content = "You are a helpful AI assistant."

                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ]

                # Use wrapper to call Python 3.12 environment
                return generate_with_gpt_oss(
                    messages=messages,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature
                )

            # Otherwise, use regular HuggingFace Transformers loading
            # Lazy load model
            self._load_hf_model()

            # Build chat messages
            if persona_instruction:
                system_content = f"""You are a helpful AI assistant. Respond to the user's query with the following persona characteristics:

{persona_instruction}

Be natural and helpful while maintaining these stylistic traits."""
            else:
                system_content = "You are a helpful AI assistant."

            # Format for chat models (important: messages list, not plain text)
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ]

            # Apply chat template (returns tensor directly with return_tensors="pt")
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.hf_model.device)

            # Generate
            with torch.no_grad():
                outputs = self.hf_model.generate(
                    inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature if self.temperature > 0 else None,
                    do_sample=self.temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Decode only the generated part (skip the input prompt)
            generated_ids = outputs[0][inputs.shape[-1]:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            return generated_text.strip()

        except Exception as e:
            error_msg = f"Error generating HuggingFace response: {e}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return f"[Error: {str(e)}]"

    def _extract_text_from_responses(self, response) -> str:
        """
        Extract text from Responses API response object

        Args:
            response: Response object from responses.create()

        Returns:
            Generated text content
        """
        # Handle different possible response formats
        if hasattr(response, 'output'):
            # If output is a list of messages
            if isinstance(response.output, list):
                for msg in response.output:
                    if hasattr(msg, 'content'):
                        return msg.content
                    elif isinstance(msg, dict) and 'content' in msg:
                        return msg['content']
            # If output is a string
            elif isinstance(response.output, str):
                return response.output

        # Fallback: try to extract from choices (OpenAI-style)
        if hasattr(response, 'choices') and len(response.choices) > 0:
            choice = response.choices[0]
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                return choice.message.content

        # Last resort: convert to string
        return str(response)


def main():
    """Demo usage"""
    generator = PersonaGenerator()

    # Test with sample traits
    test_traits = {
        'R1': 0.8,   # directness: high
        'R2': -0.3,  # emotional_valence: slightly negative
        'R4': 0.6,   # audience_focus: high
    }

    test_prompt = "Pythonでファイルを読み込む方法を教えてください"

    response = generator.generate_response(test_prompt, test_traits, model="mock")
    print(response)


if __name__ == "__main__":
    main()
