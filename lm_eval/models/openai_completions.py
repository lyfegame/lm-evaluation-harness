import logging
import os
from functools import cached_property
from operator import itemgetter
from typing import Any, Dict, List, Optional, Tuple, Union

from lm_eval.api.registry import register_model
from lm_eval.models.api_models import TemplateAPI
from lm_eval.models.utils import handle_stop_sequences


eval_logger = logging.getLogger(__name__)


@register_model("local-completions")
class LocalCompletionsAPI(TemplateAPI):
    def __init__(
        self,
        base_url: str = None,
        tokenizer_backend: str = "huggingface",
        **kwargs,
    ):
        super().__init__(
            base_url=base_url, tokenizer_backend=tokenizer_backend, **kwargs
        )

    def _create_payload(
        self,
        messages: Union[List[List[int]], List[dict], List[str], str],
        generate=False,
        gen_kwargs: Optional[dict] = None,
        seed: int = 1234,
        eos=None,
        **kwargs,
    ) -> dict:
        if generate:
            gen_kwargs.pop("do_sample", False)
            if "max_tokens" in gen_kwargs:
                max_tokens = gen_kwargs.pop("max_tokens")
            else:
                max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
            temperature = gen_kwargs.pop("temperature", 0)
            stop = handle_stop_sequences(gen_kwargs.pop("until", None), eos)
            return {
                "prompt": messages,
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": stop,
                "seed": seed,
                **gen_kwargs,
            }
        else:
            return {
                "model": self.model,
                "prompt": messages,
                "temperature": 0,
                "max_tokens": 1,
                "logprobs": 1,
                "seed": seed,
                "echo": True,
            }

    @staticmethod
    def parse_logprobs(
        outputs: Union[Dict, List[Dict]],
        tokens: List[List[int]] = None,
        ctxlens: List[int] = None,
        **kwargs,
    ) -> List[Tuple[float, bool]]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            for choice, ctxlen in zip(
                sorted(out["choices"], key=itemgetter("index")), ctxlens
            ):
                assert ctxlen > 0, "Context length must be greater than 0"
                logprobs = sum(choice["logprobs"]["token_logprobs"][ctxlen:-1])
                tokens_logprobs = choice["logprobs"]["token_logprobs"][ctxlen:-1]
                top_logprobs = choice["logprobs"]["top_logprobs"][ctxlen:-1]
                is_greedy = True
                for tok, top in zip(tokens_logprobs, top_logprobs):
                    if tok != max(top.values()):
                        is_greedy = False
                        break
                res.append((logprobs, is_greedy))
        return res

    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            tmp = [None] * len(out["choices"])
            for choices in out["choices"]:
                tmp[choices["index"]] = choices["text"]
            res = res + tmp
        return res

    @property
    def api_key(self):
        return os.environ.get("OPENAI_API_KEY", "")


@register_model("local-chat-completions")
class LocalChatCompletion(LocalCompletionsAPI):
    def __init__(
        self,
        base_url: str = None,
        tokenizer_backend: str = None,
        tokenized_requests: bool = False,
        **kwargs,
    ):
        eval_logger.warning(
            "chat-completions endpoint requires the `--apply_chat_template` flag."
        )
        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            tokenized_requests=tokenized_requests,
            **kwargs,
        )
        if self._batch_size > 1:
            eval_logger.warning(
                "Chat completions does not support batching. Defaulting to batch size 1."
            )
            self._batch_size = 1

    def _create_payload(
        self,
        messages: List[Dict],
        generate=False,
        gen_kwargs: dict = None,
        seed=1234,
        eos=None,
        **kwargs,
    ) -> dict:
        assert type(messages) is not str, (
            "chat-completions require the --apply_chat_template flag."
        )
        gen_kwargs.pop("do_sample", False)
        if "max_tokens" in gen_kwargs:
            max_tokens = gen_kwargs.pop("max_tokens")
        else:
            max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
        temperature = gen_kwargs.pop("temperature", 0)
        stop = handle_stop_sequences(gen_kwargs.pop("until", None), eos)
        if not isinstance(stop, (list, tuple)):
            stop = [stop]
        return {
            "messages": messages,
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop[:4],
            "seed": seed,
            **gen_kwargs,
        }

    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            tmp = [None] * len(out["choices"])
            for choices in out["choices"]:
                tmp[choices["index"]] = choices["message"]["content"]
            res = res + tmp
        return res

    def tok_encode(
        self,
        string: Union[str, Any],
        left_truncate_len=None,
        add_special_tokens=None,
        **kwargs,
    ) -> Union[List[str], List[int], Any]:
        return string

    def loglikelihood(self, requests, **kwargs):
        raise NotImplementedError(
            "Loglikelihood is not supported for chat completions. Consider using the completions API instead."
        )


@register_model(
    "openai-completions",
)
class OpenAICompletionsAPI(LocalCompletionsAPI):
    def __init__(
        self,
        base_url="https://api.openai.com/v1/completions",
        tokenizer_backend="tiktoken",
        **kwargs,
    ):
        super().__init__(
            base_url=base_url, tokenizer_backend=tokenizer_backend, **kwargs
        )

    @cached_property
    def api_key(self):
        """Override this property to return the API key for the API request."""
        key = os.environ.get("OPENAI_API_KEY", None)
        if key is None:
            raise ValueError(
                "API key not found. Please set the `OPENAI_API_KEY` environment variable."
            )
        return key

    def loglikelihood(self, requests, **kwargs):
        assert self.model in [
            "babbage-002",
            "davinci-002",
        ], (
            f"Prompt loglikelihoods are only supported by OpenAI's API for {['babbage-002', 'davinci-002']}."
        )
        return super().loglikelihood(requests, **kwargs)

    def chat_template(self, chat_template: Union[bool, str] = False) -> Optional[str]:
        return ""


@register_model("openai-chat-completions")
class OpenAIChatCompletion(LocalChatCompletion):
    def __init__(
        self,
        base_url="https://api.openai.com/v1/chat/completions",
        tokenizer_backend=None,
        tokenized_requests=False,
        **kwargs,
    ):
        if "o1" in kwargs.get("model", ""):
            eval_logger.warning(
                "o1 models do not support `stop` and only support temperature=1"
            )
        
        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            tokenized_requests=tokenized_requests,
            **kwargs,
        )

    @cached_property
    def api_key(self):
        """Override this property to return the API key for the API request."""
        key = os.environ.get("OPENAI_API_KEY", None)
        if key is None:
            raise ValueError(
                "API key not found. Please set the `OPENAI_API_KEY` environment variable."
            )
        return key

    def loglikelihood(self, requests, **kwargs):
        raise NotImplementedError(
            "Loglikelihood (and therefore `multiple_choice`-type tasks) is not supported for chat completions as OpenAI does not provide prompt logprobs. See https://github.com/EleutherAI/lm-evaluation-harness/issues/942#issuecomment-1777836312 or https://github.com/EleutherAI/lm-evaluation-harness/issues/1196 for more background on this limitation."
        )

    def _create_payload(
        self,
        messages: List[Dict],
        generate=False,
        gen_kwargs: dict = None,
        seed=1234,
        eos="<|endoftext|>",
        **kwargs,
    ) -> dict:
        assert type(messages) is not str, (
            "chat-completions require the --apply_chat_template flag."
        )
        gen_kwargs.pop("do_sample", False)
        if "max_tokens" in gen_kwargs:
            max_tokens = gen_kwargs.pop("max_tokens")
        else:
            max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
        temperature = gen_kwargs.pop("temperature", 0)
        stop = handle_stop_sequences(gen_kwargs.pop("until", ["<|endoftext|>"]), eos)
        if not isinstance(stop, (list, tuple)):
            stop = [stop]
        output = {
            "messages": messages,
            "model": self.model,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop[:4],
            "seed": seed,
            **gen_kwargs,
        }
        if "o1" in self.model:
            output.pop("stop")
            output["temperature"] = 1
        elif "o3" in self.model:
            output.pop("temperature")
        return output


@register_model("gpt-5", "openai-gpt5-responses", "o1", "o1-preview", "o1-mini", "openai-responses")
class OpenAIResponses(LocalChatCompletion):
    """
    Model class for OpenAI models using the /v1/responses endpoint.
    This includes o1, o1-preview, o1-mini, and future reasoning models.
    
    API structure:
    - Endpoint: /v1/responses 
    - Input format: 'messages' field (same as chat)
    - Supports 'reasoning_effort' parameter for thinking budget
    """
    
    def __init__(
        self,
        base_url="https://api.openai.com/v1/responses",
        tokenizer_backend=None,
        tokenized_requests=False,
        reasoning_effort=None,  # Thinking budget parameter
        model=None,  # Model name parameter
        **kwargs,
    ):
        if reasoning_effort:
            eval_logger.info(f"Using reasoning_effort: {reasoning_effort}")
        
        # Store reasoning_effort as an instance variable
        self.reasoning_effort = reasoning_effort
        
        # If no model specified, default to gpt-5
        if model is None and "pretrained" not in kwargs:
            model = "gpt-5"
        
        super().__init__(
            base_url=base_url,
            model=model,
            tokenizer_backend=tokenizer_backend,
            tokenized_requests=tokenized_requests,
            **kwargs,
        )

    @cached_property
    def api_key(self):
        """Override this property to return the API key for the API request."""
        key = os.environ.get("OPENAI_API_KEY", None)
        if key is None:
            raise ValueError(
                "API key not found. Please set the `OPENAI_API_KEY` environment variable."
            )
        return key

    def _create_payload(
        self,
        messages: List[Dict],
        generate=False,
        gen_kwargs: dict = None,
        seed=1234,
        eos="<|endoftext|>",
        **kwargs,
    ) -> dict:
        """
        Create payload for OpenAI responses endpoint.
        Based on https://platform.openai.com/docs/api-reference/responses/create
        """
        gen_kwargs = gen_kwargs or {}
        
        # Extract parameters
        max_output_tokens = None
        if "max_output_tokens" in gen_kwargs:
            max_output_tokens = gen_kwargs.pop("max_output_tokens")
        elif "max_tokens" in gen_kwargs:
            max_output_tokens = gen_kwargs.pop("max_tokens")
        elif "max_gen_toks" in gen_kwargs:
            max_output_tokens = gen_kwargs.pop("max_gen_toks")
        
        # Remove unsupported parameters
        gen_kwargs.pop("do_sample", False)
        gen_kwargs.pop("temperature", None)
        gen_kwargs.pop("top_p", None)
        gen_kwargs.pop("top_k", None)
        gen_kwargs.pop("until", None)
        gen_kwargs.pop("stop", None)
        
        # Extract reasoning_effort
        reasoning_effort = None
        if "reasoning_effort" in gen_kwargs:
            reasoning_effort = gen_kwargs.pop("reasoning_effort")
        elif hasattr(self, 'reasoning_effort') and self.reasoning_effort:
            reasoning_effort = self.reasoning_effort
        
        # Build the payload according to API spec
        # The Responses API uses 'input' not 'messages'
        # Convert messages to a single input string
        if isinstance(messages, list) and messages:
            if isinstance(messages[0], dict):
                # Chat format - extract content from last message
                input_text = messages[-1].get("content", "")
            else:
                input_text = str(messages)
        else:
            input_text = str(messages)
            
        # Ensure model is a string, not a dict
        if not isinstance(self.model, str):
            eval_logger.warning(f"Model is not a string: {type(self.model)} = {self.model}")
            model_name = "gpt-5"
        else:
            model_name = self.model
        
        output = {
            "model": model_name,
            "input": input_text,
        }
        
        # Add optional parameters
        if max_output_tokens is not None:
            output["max_output_tokens"] = max_output_tokens
            
        # Reasoning parameter uses nested structure: reasoning.effort
        if reasoning_effort is not None:
            output["reasoning"] = {"effort": reasoning_effort}
        
        # Log any remaining unsupported parameters
        if gen_kwargs:
            eval_logger.warning(
                f"Responses API does not support these parameters, ignoring: {list(gen_kwargs.keys())}"
            )
        
        return output
    
    @staticmethod  
    def parse_generations(outputs: Union[Dict, List], **kwargs) -> List[str]:
        """
        Parse responses from the OpenAI Responses API.
        Returns only the message content, discarding reasoning.
        """
        if not isinstance(outputs, list):
            outputs = [outputs]
        
        res = []
        for output in outputs:
            # GPT-5 response format: {"output": [{type: "reasoning", ...}, {type: "message", content: [...]}], ...}
            if isinstance(output, dict) and "output" in output:
                output_list = output["output"]
                # Find and extract the message content from the output list
                for item in output_list:
                    if isinstance(item, dict) and item.get("type") == "message" and "content" in item:
                        content = item["content"]
                        # Content is a list of objects with type: "output_text"
                        if isinstance(content, list):
                            for content_item in content:
                                if isinstance(content_item, dict) and content_item.get("type") == "output_text":
                                    text = content_item.get("text", "")
                                    res.append(text)
                                    break
                            else:
                                res.append("")
                        else:
                            res.append(str(content))
                        break
                else:
                    res.append("")
            # Fallback for other response formats
            elif isinstance(output, dict) and "message" in output:
                message = output["message"]
                if isinstance(message, dict) and "content" in message:
                    res.append(message["content"])
                else:
                    res.append("")
            else:
                res.append("")
        
        return res
