#!/usr/bin/env python3
"""
Hugging Face LeaderAgent for SWE-bench tasks.
Uses Hugging Face transformers directly instead of API calls.
"""
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
except ImportError:
    print("Please install transformers and torch: pip install transformers torch")
    exit(1)

class HuggingFaceLeaderAgent:
    """
    A Hugging Face-based LeaderAgent that generates proper patches for SWE-bench tasks.
    """
    
    def __init__(self, model_name: str, working_dir: str, max_iterations: int = 3, device: str = "auto"):
        self.model_name = model_name
        self.working_dir = Path(working_dir)
        self.max_iterations = max_iterations
        self.conversation: List[Dict[str, str]] = []
        
        # Create working directory
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"HuggingFaceLeaderAgent initialized:")
        print(f"  Model: {model_name}")
        print(f"  Working dir: {working_dir}")
        print(f"  Max iterations: {max_iterations}")
        print(f"  Using local Hugging Face transformers")
        
        # Load environment variables for authentication
        from dotenv import load_dotenv
        load_dotenv()
        
        # Get Hugging Face token
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
        if not self.hf_token:
            print("Warning: No Hugging Face token found. Some models may not be accessible.")
        
        # Initialize model and tokenizer
        self._load_model(device)
        
        print("Ready to use local Hugging Face model!")
    
    def _load_model(self, device: str = "auto"):
        """Load the model and tokenizer locally."""
        try:
            print(f"Loading model {self.model_name}...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=self.hf_token,
                trust_remote_code=True
            )
            
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with memory optimization
            if device == "auto":
                # Use CPU for large models to avoid memory issues
                device_map = "cpu"
                torch_dtype = torch.float32
            else:
                device_map = device
                torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
                
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=self.hf_token,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                offload_folder="./offload"  # Offload to disk if needed
            )
            
            print(f"Model loaded successfully on device: {self.model.device}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
    
    def call_model(self, messages: List[Dict[str, str]]) -> str:
        """Call the Hugging Face model locally."""
        try:
            # Convert messages to a single prompt
            prompt = self._messages_to_prompt(messages)
            
            # Tokenize the input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode the response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input prompt from the response
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            print(f"Error calling model: {e}")
            return ""
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to a single prompt string."""
        prompt = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"
        
        prompt += "Assistant:"
        return prompt
    
    def extract_patch_from_response(self, response: str) -> str:
        """Extract patch content from model response."""
        # Look for patch content between ```diff and ``` or ```patch and ```
        import re
        
        # Try to find patch in code blocks
        patch_patterns = [
            r'```(?:diff|patch)\n(.*?)\n```',
            r'```\n(.*?)\n```',
            r'--- a/.*?\n(?:\+.*?\n)*'
        ]
        
        for pattern in patch_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                patch_content = matches[0].strip()
                if patch_content.startswith('--- a/'):
                    return patch_content
        
        # If no code block found, look for patch content directly
        lines = response.split('\n')
        patch_lines = []
        in_patch = False
        
        for line in lines:
            if line.startswith('--- a/'):
                in_patch = True
                patch_lines.append(line)
            elif in_patch:
                if line.startswith('+++ b/') or line.startswith('@@') or line.startswith('-') or line.startswith('+') or line.startswith(' '):
                    patch_lines.append(line)
                elif line.strip() == '':
                    patch_lines.append(line)
                else:
                    break
        
        if patch_lines:
            return '\n'.join(patch_lines)
        
        return ""
    
    def solve_task(self, prompt: str) -> Dict[str, Any]:
        """Solve the SWE-bench task by generating a proper patch."""
        start_time = time.time()
        print(f"Starting patch generation for SWE-bench task...")
        
        # System message focused on patch generation
        system_message = """You are an expert software engineer solving a SWE-bench task. 
Your task is to generate a unified diff patch that fixes a specific bug in a codebase.

CRITICAL REQUIREMENTS:
1. Generate ONLY a unified diff patch, no explanations or additional text
2. The patch must start with "---" and "+++" lines
3. Include proper file paths (e.g., "--- a/django/db/models/constraints.py")
4. Show the exact lines that need to be changed
5. Use proper diff format with context lines

Example format:
--- a/django/db/models/constraints.py
+++ b/django/db/models/constraints.py
@@ -123,7 +123,7 @@ class CheckConstraint:
     def __init__(self, check, name):
         self.check = check
         self.name = name
-        # Old buggy code here
+        # Fixed code here

Generate the patch that fixes the described bug."""
        
        self.conversation.append({"role": "system", "content": system_message})
        self.conversation.append({"role": "user", "content": prompt})
        
        for iteration in range(self.max_iterations):
            print(f"Iteration {iteration + 1}/{self.max_iterations}")
            
            # Call the model
            response = self.call_model(self.conversation)
            if not response:
                print("No response from model")
                continue
            
            # Extract patch from response
            patch_content = self.extract_patch_from_response(response)
            
            if patch_content:
                print(f"Generated patch ({len(patch_content)} characters)")
                
                # Save patch to file
                patch_file = self.working_dir / "solution.patch"
                with open(patch_file, 'w') as f:
                    f.write(patch_content)
                
                # Save conversation
                conversation_file = self.working_dir / "conversation.json"
                with open(conversation_file, 'w') as f:
                    json.dump(self.conversation, f, indent=2)
                
                duration = time.time() - start_time
                
                return {
                    "status": "completed",
                    "duration": duration,
                    "iterations": iteration + 1,
                    "conversation_path": str(conversation_file),
                    "patch_path": str(patch_file),
                    "patch": patch_content
                }
            else:
                print("No valid patch found in response")
                # Add the response to conversation for next iteration
                self.conversation.append({"role": "assistant", "content": response})
                self.conversation.append({"role": "user", "content": "Please generate a proper unified diff patch. The response should contain only the patch content starting with '--- a/' and ending with the last line of the patch."})
        
        # If we get here, all iterations failed
        duration = time.time() - start_time
        return {
            "status": "failed",
            "duration": duration,
            "iterations": self.max_iterations,
            "error": "Failed to generate valid patch after all iterations"
        }

def main():
    """Main entry point for the HuggingFaceLeaderAgent script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hugging Face LeaderAgent for SWE-bench tasks")
    parser.add_argument("--model-name", required=True, help="Hugging Face model name (e.g., microsoft/DialoGPT-medium)")
    parser.add_argument("--working-dir", required=True, help="Working directory")
    parser.add_argument("--max-iterations", type=int, default=3, help="Maximum iterations")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--prompt", help="Task prompt (if not provided, will read from stdin)")
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = HuggingFaceLeaderAgent(
        model_name=args.model_name,
        working_dir=args.working_dir,
        max_iterations=args.max_iterations,
        device=args.device
    )
    
    # Get prompt
    if args.prompt:
        prompt = args.prompt
    else:
        prompt = input("Enter the task prompt: ")
    
    # Solve the task
    result = agent.solve_task(prompt)
    
    # Print results
    print(f"\n=== Task Results ===")
    print(f"Status: {result['status']}")
    print(f"Duration: {result['duration']:.2f}s")
    print(f"Iterations: {result['iterations']}")
    
    if result['status'] == 'completed':
        print(f"Patch generated: {result['patch_path']}")
        print(f"Patch size: {len(result['patch'])} characters")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
