"""
Google Gemini API wrapper for lm-evaluation-harness
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model


eval_logger = logging.getLogger(__name__)


def gemini_completion(
    client,
    model: str,
    prompt: str,
    max_tokens: Optional[int] = None,
    temperature: float = 0.0,
    stop: Optional[List[str]] = None,
    thinking_budget: Optional[int] = None,
) -> str:
    """
    Wrapper for Google Gemini API completion.

    Args:
        client: google.genai.Client instance
        model: Model name (e.g., "gemini-2.5-pro")
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        stop: Stop sequences (not directly supported by Gemini)
        thinking_budget: Thinking budget for models that support it

    Returns:
        Generated text response
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError(
            "Please install the Google Generative AI package: "
            "pip install google-generativeai"
        )

    # Build generation config
    config_kwargs = {}
    if temperature is not None:
        config_kwargs["temperature"] = temperature
    if max_tokens is not None:
        config_kwargs["max_output_tokens"] = max_tokens

    # Add thinking config if budget specified
    if thinking_budget is not None:
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_budget=thinking_budget
        )

    config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

    # Generate response
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )

    # Extract text from response
    text = response.text

    # Handle stop sequences manually if provided
    if stop:
        for stop_seq in stop:
            if stop_seq in text:
                text = text[: text.index(stop_seq)]
                break

    return text


@register_model("gemini", "google-gemini")
class GeminiLM(LM):
    """
    Google Gemini API wrapper for language model evaluation.
    
    Example usage:
        lm_eval --model gemini \
            --model_args model=gemini-2.5-pro \
            --tasks hellaswag
            
        # With limited thinking budget and higher concurrency:
        lm_eval --model gemini \
            --model_args model=gemini-2.5-pro,thinking_budget=1024,num_concurrent=50 \
            --tasks hellaswag
    """

    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        truncate: bool = False,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        thinking_budget: Optional[int] = -1,
        num_concurrent: int = 10,
        **kwargs,
    ):
        """
        Initialize Gemini model wrapper.

        Args:
            model: Model name (default: "gemini-2.5-pro")
            truncate: Whether to truncate input
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)
            thinking_budget: Thinking budget for supported models (-1 = unlimited, default)
            num_concurrent: Maximum number of concurrent API calls (default: 10)
            **kwargs: Additional arguments
        """
        super().__init__()

        self.model = model
        self.truncate = truncate
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.thinking_budget = thinking_budget
        self.num_concurrent = num_concurrent
        self.kwargs = kwargs

        # Initialize client
        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "Please install the Google Generative AI package: "
                "pip install google-generativeai"
            )

        # Check for API key
        if "GEMINI_API_KEY" not in os.environ:
            raise ValueError(
                "GEMINI_API_KEY environment variable not set. "
                "Please set it to your Google AI API key."
            )

        self.client = genai.Client()
        eval_logger.info(f"Initialized Gemini model: {self.model}")
        if self.thinking_budget:
            eval_logger.info(f"Using thinking budget: {self.thinking_budget}")
        eval_logger.info(f"Max concurrent requests: {self.num_concurrent}")

    @property
    def eot_token_id(self):
        # Gemini doesn't expose token IDs
        return None

    @property
    def max_length(self):
        # Gemini models have different context lengths
        # This is a conservative default
        return 32768

    @property
    def max_gen_toks(self):
        return self.max_tokens or 8192

    @property
    def batch_size(self):
        # Gemini API doesn't support batching
        return 1

    @property
    def device(self):
        # API model, no device
        return "api"

    def tok_encode(self, string: str) -> List[int]:
        # Gemini doesn't provide tokenization
        # Return dummy tokens based on approximate token count
        # (rough estimate: ~4 chars per token)
        return list(range(len(string) // 4))

    def tok_decode(self, tokens: List[int]) -> str:
        # Not supported
        raise NotImplementedError("Gemini API does not support token decoding")

    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError(
            "Gemini API does not support loglikelihood computation"
        )

    def generate_until(self, requests, disable_tqdm: bool = False) -> List[str]:
        """
        Generate text until stop sequences are encountered.

        Args:
            requests: List of generation requests
            disable_tqdm: Whether to disable progress bar

        Returns:
            List of generated texts
        """
        if not requests:
            return []

        # Prepare all requests
        request_data = []
        for request in requests:
            prompt = request.args[0]
            gen_kwargs = request.args[1]

            request_data.append(
                {
                    "prompt": prompt,
                    "until": gen_kwargs.get("until", []),
                    "max_tokens": gen_kwargs.get("max_gen_toks", self.max_gen_toks),
                    "temperature": gen_kwargs.get("temperature", self.temperature),
                }
            )

        # Process requests concurrently
        results = [None] * len(request_data)

        def process_request(idx, data):
            try:
                response = gemini_completion(
                    self.client,
                    self.model,
                    data["prompt"],
                    max_tokens=data["max_tokens"],
                    temperature=data["temperature"],
                    stop=data["until"],
                    thinking_budget=self.thinking_budget,
                )
                return idx, response
            except Exception as e:
                eval_logger.error(f"Error generating response for request {idx}: {e}")
                return idx, ""

        # Use ThreadPoolExecutor for concurrent requests
        with ThreadPoolExecutor(max_workers=self.num_concurrent) as executor:
            # Submit all tasks
            futures = [
                executor.submit(process_request, idx, data)
                for idx, data in enumerate(request_data)
            ]

            # Process completed tasks with progress bar
            with tqdm(
                total=len(futures), disable=disable_tqdm, desc="Generating"
            ) as pbar:
                for future in as_completed(futures):
                    idx, response = future.result()
                    results[idx] = response
                    pbar.update(1)

        return results

    def _model_call(self, inps):
        # Not used since we override generate_until
        raise NotImplementedError()

    def _model_generate(self, context, max_length, eos_token_id):
        # Not used since we override generate_until
        raise NotImplementedError()

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError(
            "Gemini API does not support loglikelihood computation"
        )

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError(
            "Gemini API does not support loglikelihood computation"
        )
