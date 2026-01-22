"""
Baseline 2: Prompt Persona Method
Injects persona description into system prompt.
Classic approach for persona steering without activation manipulation.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from persona_opt.utils.config_loader import PromptTemplateManager
from persona_opt.evaluation.utils import load_persona_profile


class PromptPersonaMethod:
    """
    Prompt-based persona steering using system prompt injection.
    """

    def __init__(
        self,
        steerer,
        persona_id: str,
        persona_profile: Optional[Dict] = None,
        templates_path: str = "config/prompt_templates.yaml",
        **kwargs
    ):
        """
        Args:
            steerer: Llama3ActivationSteerer instance
            persona_id: Persona identifier
            persona_profile: Preloaded persona profile (optional)
            templates_path: Path to prompt templates
            **kwargs: Additional configuration
        """
        self.steerer = steerer
        self.persona_id = persona_id
        self.method_name = "prompt_persona"

        # Load persona profile
        if persona_profile is None:
            persona_profile = load_persona_profile(persona_id)
        self.persona_profile = persona_profile

        # Load template manager
        self.template_manager = PromptTemplateManager(templates_path)

        # Build persona description
        self.persona_description = self._build_persona_description()

    def _build_persona_description(self) -> str:
        """Build natural language persona description."""
        # Extract key traits from profile
        traits = []

        # Communication style
        if 'communication_style' in self.persona_profile:
            style = self.persona_profile['communication_style']
            if style.get('formality') == 'informal':
                traits.append("informal and casual")
            elif style.get('formality') == 'formal':
                traits.append("formal and professional")

            if style.get('humor') == 'high':
                traits.append("humorous and playful")

            if style.get('empathy') == 'high':
                traits.append("empathetic and supportive")

        # Big Five traits (if available)
        if 'big_five' in self.persona_profile:
            big5 = self.persona_profile['big_five']
            if big5.get('extraversion', 0) > 0.5:
                traits.append("outgoing and sociable")
            if big5.get('openness', 0) > 0.5:
                traits.append("open to new experiences")
            if big5.get('conscientiousness', 0) > 0.5:
                traits.append("organized and detail-oriented")

        # Fallback: use generic description
        if not traits:
            traits = [
                "friendly and helpful",
                "clear and concise",
                "respectful and professional"
            ]

        # Format as natural description
        description = "You are " + ", ".join(traits[:3]) + "."
        return description

    def _format_prompt_with_persona(self, prompt: str) -> str:
        """
        Format prompt with persona context.

        For Llama-3-8B-Instruct, we inject persona into the system message.
        """
        # Create messages with persona system prompt
        messages = [
            {
                "role": "system",
                "content": f"You are a helpful AI assistant. {self.persona_description}"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Apply chat template
        formatted_prompt = self.steerer.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return formatted_prompt

    def generate(self, prompt: str, **generation_kwargs) -> str:
        """
        Generate response with persona-injected prompt.

        Args:
            prompt: Input prompt
            **generation_kwargs: Generation parameters

        Returns:
            Generated response
        """
        # Remove any activation steering
        self.steerer.remove_hooks()

        # Format prompt with persona
        formatted_prompt = self._format_prompt_with_persona(prompt)

        # Tokenize manually since we've already applied chat template
        inputs = self.steerer.tokenizer(
            formatted_prompt,
            return_tensors="pt"
        ).to(self.steerer.device)

        # Generate
        with torch.no_grad():
            if generation_kwargs.get('temperature', 0.0) == 0.0:
                outputs = self.steerer.model.generate(
                    **inputs,
                    max_new_tokens=generation_kwargs.get('max_new_tokens', 128),
                    do_sample=False,
                    pad_token_id=self.steerer.tokenizer.eos_token_id,
                )
            else:
                outputs = self.steerer.model.generate(
                    **inputs,
                    max_new_tokens=generation_kwargs.get('max_new_tokens', 128),
                    temperature=generation_kwargs.get('temperature', 1.0),
                    do_sample=True,
                    top_p=generation_kwargs.get('top_p', 1.0),
                    pad_token_id=self.steerer.tokenizer.eos_token_id,
                )

        # Decode
        response = self.steerer.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response

    def batch_generate(self, prompts: List[str], **generation_kwargs) -> List[str]:
        """
        Generate multiple responses with persona-injected prompts.

        Args:
            prompts: List of input prompts
            **generation_kwargs: Generation parameters

        Returns:
            List of generated responses
        """
        responses = []
        for prompt in prompts:
            response = self.generate(prompt, **generation_kwargs)
            responses.append(response)

        return responses

    def get_config(self) -> Dict[str, Any]:
        """Get method configuration."""
        return {
            'method': self.method_name,
            'persona_id': self.persona_id,
            'description': 'System prompt persona injection',
            'persona_description': self.persona_description,
            'steering': 'prompt_based',
        }

    def __repr__(self):
        return f"PromptPersonaMethod(persona_id={self.persona_id})"


# Import torch at module level for generate method
import torch
