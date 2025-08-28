"""
Custom task class that supports per-document system prompts.
This allows each example in the dataset to have its own system prompt.
"""

from lm_eval.api.task import ConfigurableTask
from typing import Union, List, Dict, Optional, Callable
import copy


class SystemPromptTask(ConfigurableTask):
    """
    A custom task class that extracts system prompts from each document.
    
    This class looks for a 'system_prompt' field in each document and uses it
    as the system prompt for that specific example. If no system_prompt field
    is present, it falls back to the global system_instruction or no system prompt.
    """
    
    VERSION = 0
    
    def fewshot_context(
        self,
        doc: dict,
        num_fewshot: int,
        system_instruction: Optional[str] = None,
        apply_chat_template: bool = False,
        fewshot_as_multiturn: bool = False,
        chat_template: Optional[Callable] = None,
        gen_prefix: Optional[str] = None,
    ) -> Union[str, List[str]]:
        """
        Override fewshot_context to use per-document system prompts.
        
        This method checks if the document has a 'system_prompt' field and uses it
        as the system instruction for this specific example.
        """
        
        # Extract per-document system prompt if it exists
        doc_system_prompt = doc.get('system_prompt', None)
        
        # Use document-specific system prompt if available, otherwise fall back to global
        if doc_system_prompt is not None:
            effective_system_instruction = doc_system_prompt
        else:
            effective_system_instruction = system_instruction
        
        # Call the parent class's fewshot_context with the effective system instruction
        return super().fewshot_context(
            doc=doc,
            num_fewshot=num_fewshot,
            system_instruction=effective_system_instruction,
            apply_chat_template=apply_chat_template,
            fewshot_as_multiturn=fewshot_as_multiturn,
            chat_template=chat_template,
            gen_prefix=gen_prefix,
        )
    
    def doc_to_text(self, doc):
        """
        Extract the instruction/question from the document.
        """
        # Support multiple possible field names for the instruction, only use instruction for now
        for field in ['instruction']:
            if field in doc:
                return doc[field]
        
        # If no standard field is found, raise an error
        raise ValueError(f"No instruction field found in document. Available fields: {list(doc.keys())}")
    
    def doc_to_target(self, doc):
        """
        Extract the target response from the document.
        """
        # Support multiple possible field names for the response, only use response for now
        for field in ['response']:
            if field in doc:
                return doc[field]
        
        # If no standard field is found, raise an error
        raise ValueError(f"No response field found in document. Available fields: {list(doc.keys())}")
