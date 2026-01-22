"""
Judge Module
LLM-based A/B preference evaluation using GPT-4o
"""

import json
import random
import hashlib
import os
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

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


class LLMJudge:
    """A/B preference evaluation using LLM judge"""

    def __init__(
        self,
        policy_path: str = "policy/eval_policy_v2.md",
        model: str = "mock",
        api_key: Optional[str] = None,
        tie_threshold: float = 0.03
    ):
        """
        Initialize the judge

        Args:
            policy_path: Path to evaluation policy document
            model: Judge model ("mock", "gpt-4o", "gpt-4o-mini", "gpt-oss-20b-local")
            api_key: OpenAI API key (if using real model)
            tie_threshold: Margin threshold for TIE decisions
        """
        self.policy_path = Path(policy_path)
        self.tie_threshold = tie_threshold
        self.policy = self._load_policy()

        # Determine backend based on model
        if isinstance(model, str) and model.startswith("hf/"):
            # HuggingFace model
            self.backend = "huggingface"
            self.model = model.replace("hf/", "")
            self.client = None
            self.tokenizer = None
            self.hf_model = None
        elif model == "gpt-oss-20b-local" or (isinstance(model, str) and model.startswith("ollama/")):
            self.backend = "ollama"
            # Extract model name (remove "ollama/" prefix if present)
            if isinstance(model, str) and model.startswith("ollama/"):
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
        elif model == "mock":
            self.backend = "mock"
            self.model = model
            self.client = None
            self.tokenizer = None
            self.hf_model = None
        else:
            self.backend = "openai"
            self.model = model
            # Initialize OpenAI client if needed
            if not OPENAI_AVAILABLE:
                raise RuntimeError("OpenAI package not installed. Run: pip install openai")

            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key parameter")

            self.client = OpenAI(api_key=self.api_key)
            self.tokenizer = None
            self.hf_model = None

    def _load_policy(self) -> str:
        """Load evaluation policy"""
        if not self.policy_path.exists():
            return "Default evaluation policy: Compare responses on helpfulness, correctness, clarity."

        with open(self.policy_path, 'r', encoding='utf-8') as f:
            return f.read()

    def judge(
        self,
        user_prompt: str,
        response_A: str,
        response_B: str,
        trait_dict: Optional[Dict[str, float]] = None,
        swap_order: bool = False
    ) -> Dict:
        """
        Evaluate A vs B and return preference (alias for judge_AB)

        Args:
            user_prompt: Original user prompt
            response_A: Response A (typically baseline)
            response_B: Response B (trait-conditioned candidate)
            trait_dict: Trait vector used for B (for persona fit evaluation)
            swap_order: If True, present B before A (for consistency check)

        Returns:
            Dictionary with:
                - preference: "A" | "B" | "TIE"
                - confidence: float [0.5, 1.0]
                - reasoning: explanation text
                - margin: score difference (B - A)
        """
        return self.judge_AB(user_prompt, response_A, response_B, trait_dict, swap_order)

    def judge_AB(
        self,
        prompt: str,
        response_A: str,
        response_B: str,
        trait_dict: Optional[Dict[str, float]] = None,
        swap_order: bool = False
    ) -> Dict:
        """
        Evaluate A vs B and return preference

        Args:
            prompt: Original user prompt
            response_A: Response A (typically baseline)
            response_B: Response B (trait-conditioned candidate)
            trait_dict: Trait vector used for B (for persona fit evaluation)
            swap_order: If True, present B before A (for consistency check)

        Returns:
            Dictionary with:
                - preference: "A" | "B" | "TIE"
                - confidence: float [0.5, 1.0]
                - reasoning: explanation text
                - margin: score difference (B - A)
        """
        if self.backend == "mock":
            return self._judge_mock(prompt, response_A, response_B, trait_dict, swap_order)
        else:
            return self._judge_llm(prompt, response_A, response_B, trait_dict, swap_order)

    def _judge_mock(
        self,
        prompt: str,
        response_A: str,
        response_B: str,
        trait_dict: Optional[Dict[str, float]],
        swap_order: bool
    ) -> Dict:
        """
        Mock judge implementation for testing

        Uses simple heuristics:
        - Length difference
        - Trait signal strength
        - Random noise
        """
        # Deterministic hash for consistency
        content_hash = hashlib.sha256(
            (prompt + response_A + response_B).encode('utf-8')
        ).hexdigest()
        seed = int(content_hash[:8], 16)
        rng = random.Random(seed)

        # Score based on length (simple heuristic)
        len_A = len(response_A.split())
        len_B = len(response_B.split())

        # Prefer moderate length (not too short, not too long)
        ideal_len = 150
        score_A = 1.0 - abs(len_A - ideal_len) / ideal_len
        score_B = 1.0 - abs(len_B - ideal_len) / ideal_len

        # Add trait signal bonus for B
        if trait_dict:
            trait_strength = sum(abs(v) for v in trait_dict.values()) / len(trait_dict)
            score_B += 0.1 * trait_strength

        # Add random noise
        score_A += rng.uniform(-0.1, 0.1)
        score_B += rng.uniform(-0.1, 0.1)

        # Compute margin
        margin = score_B - score_A

        # Apply swap if requested
        if swap_order:
            margin = -margin

        # Determine preference based on threshold
        if margin > self.tie_threshold:
            preference = "B"
        elif margin < -self.tie_threshold:
            preference = "A"
        else:
            preference = "TIE"

        # Confidence based on margin magnitude
        confidence = 0.50 + min(0.49, abs(margin))

        reasoning = f"[MOCK Judge] Length A={len_A}, B={len_B}. Score A={score_A:.3f}, B={score_B:.3f}. Margin={margin:.3f}. Threshold={self.tie_threshold}"

        return {
            "preference": preference,
            "confidence": confidence,
            "margin": margin,
            "reasoning": reasoning,
            "scores": {"A": score_A, "B": score_B}
        }

    def _load_hf_model(self):
        """Load HuggingFace model and tokenizer (lazy loading)"""
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers/torch not installed. Run: pip install transformers torch")

        if self.tokenizer is None or self.hf_model is None:
            print(f"Loading HuggingFace judge model: {self.model}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model)

            # Use bfloat16 for A100 (better than float16)
            # trust_remote_code needed for gpt-oss custom architecture
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                self.model,
                torch_dtype=torch.bfloat16,  # A100 optimal
                device_map="auto",
                trust_remote_code=True,  # Required for gpt-oss
            )
            print(f"Judge model loaded successfully on {self.hf_model.device}")

    def _judge_llm(
        self,
        prompt: str,
        response_A: str,
        response_B: str,
        trait_dict: Optional[Dict[str, float]],
        swap_order: bool
    ) -> Dict:
        """
        Real LLM judge using HuggingFace, OpenAI, or Ollama
        """
        # HuggingFace backend
        if self.backend == "huggingface":
            return self._judge_hf(prompt, response_A, response_B, trait_dict, swap_order)
        # Swap responses if requested (for consistency checking)
        if swap_order:
            first, second = response_B, response_A
            first_label, second_label = "Response B", "Response A"
        else:
            first, second = response_A, response_B
            first_label, second_label = "Response A", "Response B"

        # Build evaluation prompt (optimized for Ollama to output JSON first)
        user_message = f"""Evaluate these two responses and output ONLY valid JSON.

User Prompt: {prompt}

{first_label}: {first}

{second_label}: {second}

Output a JSON object with: preference (A/B/TIE), confidence (0.5-1.0), reasoning (string), scores (object with A and B as floats).

Trait target: {trait_dict}

JSON output:"""

        system_message = f"You are an expert evaluator. Follow this policy:\n\n{self.policy}"

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

        # Call API (Ollama and OpenAI both use chat.completions)
        try:
            # Both Ollama and OpenAI use the same chat.completions API
            # Note: Ollama may not support response_format parameter
            if self.backend == "ollama":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=2000  # Increased for Ollama's reasoning field
                    # Ollama may not support response_format
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=500,
                    response_format={"type": "json_object"}
                )

            # Ollama/gpt-oss may put content in 'reasoning' field instead of 'content'
            message = response.choices[0].message
            result_text = message.content

            # Check if content is empty but reasoning field exists (Ollama/gpt-oss behavior)
            if (not result_text or len(result_text.strip()) == 0) and hasattr(message, 'reasoning'):
                reasoning_field = message.reasoning
                if reasoning_field:
                    # For gpt-oss, reasoning contains the actual content, try to use it
                    # Try to find JSON in the reasoning text
                    import re
                    json_match = re.search(r'\{[^}]*"preference"[^}]*\}', reasoning_field, re.DOTALL)
                    if json_match:
                        result_text = json_match.group(0)
                        print(f"Info: Extracted JSON from reasoning field")
                    else:
                        # No JSON found, construct response from reasoning content
                        print(f"Warning: No JSON in reasoning field. Using TIE fallback.")
                        result_text = '{"preference": "TIE", "confidence": 0.5, "reasoning": "No structured output from model", "scores": {"A": 0.5, "B": 0.5}}'
            elif not result_text or len(result_text.strip()) == 0:
                print(f"Warning: Empty response from {self.backend} judge")
                result_text = '{"preference": "TIE", "confidence": 0.5, "reasoning": "Empty response", "scores": {"A": 0.5, "B": 0.5}}'

            result = json.loads(result_text)

            # Validate and normalize
            preference = result.get("preference", "TIE").upper()
            if preference not in ["A", "B", "TIE"]:
                preference = "TIE"

            # Reverse swap if needed
            if swap_order and preference in ["A", "B"]:
                preference = "B" if preference == "A" else "A"

            confidence = float(result.get("confidence", 0.5))
            confidence = max(0.5, min(1.0, confidence))

            reasoning = result.get("reasoning", "No reasoning provided")
            scores = result.get("scores", {"A": 0.5, "B": 0.5})

            # Calculate margin
            margin = scores["B"] - scores["A"]

            return {
                "preference": preference,
                "confidence": confidence,
                "margin": margin,
                "reasoning": reasoning,
                "scores": scores,
                "raw_response": result_text
            }

        except Exception as e:
            print(f"Error calling LLM API: {e}")
            # Fallback to TIE on error
            return {
                "preference": "TIE",
                "confidence": 0.5,
                "margin": 0.0,
                "reasoning": f"Error during evaluation: {str(e)}",
                "scores": {"A": 0.5, "B": 0.5},
                "error": str(e)
            }

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

    def _judge_hf(
        self,
        prompt: str,
        response_A: str,
        response_B: str,
        trait_dict: Optional[Dict[str, float]],
        swap_order: bool
    ) -> Dict:
        """
        Judge using HuggingFace model

        Args:
            prompt: User prompt
            response_A: Response A
            response_B: Response B
            trait_dict: Trait dictionary
            swap_order: Whether to swap order

        Returns:
            Dictionary with preference, confidence, margin, reasoning, scores
        """
        try:
            # Swap responses if requested
            if swap_order:
                first, second = response_B, response_A
                first_label, second_label = "Response B", "Response A"
            else:
                first, second = response_A, response_B
                first_label, second_label = "Response A", "Response B"

            # Build evaluation prompt (simplified for better JSON output)
            user_message = f"""Evaluate these two responses and output ONLY valid JSON.

User Prompt: {prompt}

{first_label}: {first}

{second_label}: {second}

Output a JSON object with: preference (A/B/TIE), confidence (0.5-1.0), reasoning (string), scores (object with A and B as floats 0-1).

Trait target: {trait_dict}

JSON output:"""

            system_message = f"You are an expert evaluator. Output only valid JSON."

            # Format for chat models (important: messages list, not plain text)
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]

            # Check if this is gpt-oss model (requires Python 3.12 wrapper)
            if "gpt-oss" in self.model.lower():
                if not GPT_OSS_WRAPPER_AVAILABLE:
                    raise RuntimeError("gpt_oss_wrapper not available. Check persona_opt/gpt_oss_wrapper.py")

                # Use wrapper to call Python 3.12 environment
                result_text = generate_with_gpt_oss(
                    messages=messages,
                    max_new_tokens=1000,
                    temperature=0.3
                )
            else:
                # Regular HuggingFace model loading
                # Lazy load model
                self._load_hf_model()

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
                        max_new_tokens=1000,
                        temperature=0.3,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

                # Decode only the generated part (skip the input prompt)
                generated_ids = outputs[0][inputs.shape[-1]:]
                result_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{[^}]*"preference"[^}]*"scores"[^}]*\}', result_text, re.DOTALL)
            if json_match:
                result_text = json_match.group(0)
            else:
                # Try to find any JSON object
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    result_text = json_match.group(0)

            result = json.loads(result_text)

            # Validate and normalize
            preference = result.get("preference", "TIE").upper()
            if preference not in ["A", "B", "TIE"]:
                preference = "TIE"

            # Reverse swap if needed
            if swap_order and preference in ["A", "B"]:
                preference = "B" if preference == "A" else "A"

            confidence = float(result.get("confidence", 0.5))
            confidence = max(0.5, min(1.0, confidence))

            reasoning = result.get("reasoning", "No reasoning provided")
            scores = result.get("scores", {"A": 0.5, "B": 0.5})

            # Calculate margin
            margin = scores.get("B", 0.5) - scores.get("A", 0.5)

            return {
                "preference": preference,
                "confidence": confidence,
                "margin": margin,
                "reasoning": reasoning,
                "scores": scores,
                "raw_response": result_text
            }

        except Exception as e:
            print(f"Error in HuggingFace judge: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to TIE on error
            return {
                "preference": "TIE",
                "confidence": 0.5,
                "margin": 0.0,
                "reasoning": f"Error during HF evaluation: {str(e)}",
                "scores": {"A": 0.5, "B": 0.5},
                "error": str(e)
            }

    def evaluate_batch(
        self,
        prompts: list,
        responses_A: list,
        responses_B: list,
        trait_dict: Optional[Dict[str, float]] = None
    ) -> Tuple[float, float, list]:
        """
        Evaluate a batch of prompt-response pairs

        Args:
            prompts: List of prompts
            responses_A: List of A responses
            responses_B: List of B responses
            trait_dict: Trait vector for persona fit

        Returns:
            Tuple of (win_rate, tie_rate, judgments)
            - win_rate: Fraction of B wins
            - tie_rate: Fraction of TIEs
            - judgments: List of judgment dicts
        """
        judgments = []
        wins = 0
        ties = 0

        for prompt, resp_A, resp_B in zip(prompts, responses_A, responses_B):
            result = self.judge_AB(prompt, resp_A, resp_B, trait_dict)
            judgments.append(result)

            if result['preference'] == 'B':
                wins += 1
            elif result['preference'] == 'TIE':
                ties += 1

        n = len(judgments)
        win_rate = wins / n if n > 0 else 0.0
        tie_rate = ties / n if n > 0 else 0.0

        return win_rate, tie_rate, judgments


def main():
    """Demo usage"""
    judge = LLMJudge(model="mock")

    # Test case
    prompt = "Pythonでファイルを読み込む方法を教えてください"

    response_A = """Pythonでファイルを読み込むには、open()関数を使います。

with open('file.txt', 'r') as f:
    content = f.read()

これで変数contentにファイルの内容が格納されます。"""

    response_B = """open()関数でファイルを読めます。以下のコードを使ってください：

with open('file.txt', 'r', encoding='utf-8') as f:
    content = f.read()

エンコーディング指定を忘れずに。日本語ファイルの場合は'utf-8'が推奨です。エラーハンドリングも追加すると安全です。"""

    result = judge.judge_AB(prompt, response_A, response_B)

    print("=== Judge Result ===")
    print(f"Preference: {result['preference']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Margin: {result['margin']:.3f}")
    print(f"Reasoning: {result['reasoning']}")


if __name__ == "__main__":
    main()
