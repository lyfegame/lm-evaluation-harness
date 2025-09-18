#!/usr/bin/env python3
"""
Hugging Face Remote Inference LeaderAgent for SWE-bench tasks.
Uses Hugging Face Inference API instead of loading models locally.
"""
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List

try:
    from huggingface_hub import InferenceClient
except ImportError:
    print("Please install huggingface_hub: pip install huggingface_hub")
    exit(1)

class HuggingFaceRemoteAgent:
    """
    A Hugging Face-based LeaderAgent that uses remote inference API.
    This avoids loading models locally, saving memory and resources.
    """
    
    def __init__(self, model_name: str, working_dir: str, max_iterations: int = 3):
        self.model_name = model_name
        self.working_dir = Path(working_dir)
        self.max_iterations = max_iterations
        self.conversation: List[Dict[str, str]] = []
        
        # Create working directory
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"HuggingFaceRemoteAgent initialized:")
        print(f"  Model: {model_name}")
        print(f"  Working dir: {working_dir}")
        print(f"  Max iterations: {max_iterations}")
        print(f"  Using Hugging Face Inference API (remote)")
        
        # Load environment variables for authentication
        from dotenv import load_dotenv
        load_dotenv()
        
        # Get Hugging Face token
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
        if not self.hf_token:
            print("Warning: No Hugging Face token found. Some models may not be accessible.")
            print("Set HUGGINGFACE_TOKEN or HF_TOKEN environment variable.")
        
        # Initialize HF Inference Client
        self.client = InferenceClient(model=model_name, token=self.hf_token)
        
        print(f"Model: {model_name}")
        print("Ready to use Hugging Face remote inference!")
    
    def _call_hf_api(self, prompt: str, max_new_tokens: int = 2048) -> str:
        """Call Hugging Face Inference API using chat completion."""
        try:
            # Use chat completion format since that works better with most models
            messages = [
                {"role": "system", "content": "You are an expert software engineer who generates git patches to fix bugs."},
                {"role": "user", "content": prompt}
            ]
            
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=0.1,
            )
            
            # Extract the response
            choice = completion.choices[0]
            msg = choice.message if hasattr(choice, "message") else choice
            content = getattr(msg, "content", None)
            
            return content or ""
                
        except Exception as e:
            print(f"HF API call failed: {e}")
            return ""
    
    def solve_task(self, problem_statement: str) -> Dict[str, Any]:
        """
        Solve a SWE-bench task using remote inference.
        """
        print(f"Solving task with remote inference...")
        
        # Create a focused prompt for patch generation
        prompt = f"""You are an expert software engineer. Generate a patch to fix the following issue:

{problem_statement}

Generate a proper git patch that fixes the issue. The patch should:
1. Be in unified diff format
2. Include proper file paths
3. Show the exact changes needed
4. Be minimal and focused

Patch:"""
        
        # Generate patch using remote inference
        start_time = time.time()
        patch = self._call_hf_api(prompt, max_new_tokens=2048)
        duration = time.time() - start_time
        
        print(f"Remote inference completed in {duration:.2f} seconds")
        
        # Save conversation and results
        self.conversation.append({
            "role": "user",
            "content": problem_statement
        })
        self.conversation.append({
            "role": "assistant", 
            "content": patch
        })
        
        # Save conversation to file
        conversation_file = self.working_dir / "conversation.json"
        with open(conversation_file, 'w') as f:
            json.dump(self.conversation, f, indent=2)
        
        # Save patch to file
        patch_file = self.working_dir / "solution.patch"
        with open(patch_file, 'w') as f:
            f.write(patch)
        
        # Save agent output log
        log_file = self.working_dir / "leader_agent_output.log"
        with open(log_file, 'w') as f:
            f.write(f"HuggingFaceRemoteAgent Log\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Duration: {duration:.2f} seconds\n")
            f.write(f"Patch Length: {len(patch)} characters\n")
            f.write(f"API Endpoint: hf_inference:{self.model_name}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        return {
            "patch": patch,
            "conversation": self.conversation,
            "duration": duration,
            "model_name": self.model_name,
            "api_endpoint": f"hf_inference:{self.model_name}",
            "patch_file": str(patch_file),
            "conversation_file": str(conversation_file),
            "log_file": str(log_file)
        }
