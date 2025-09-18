#!/usr/bin/env python3
"""
Clean LeaderAgent for SWE-bench tasks.
Focuses on generating proper patches that can be evaluated by the SWE-bench harness.
"""
import os
import json
import time
import requests
from pathlib import Path
from typing import Dict, Any, List

class CleanLeaderAgent:
    """
    A clean, focused LeaderAgent that generates proper patches for SWE-bench tasks.
    """
    
    def __init__(self, model_endpoint: str, working_dir: str, max_iterations: int = 3):
        self.model_endpoint = model_endpoint
        self.working_dir = Path(working_dir)
        self.max_iterations = max_iterations
        self.conversation: List[Dict[str, str]] = []
        
        # Create working directory
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"CleanLeaderAgent initialized:")
        print(f"  Model endpoint: {model_endpoint}")
        print(f"  Working dir: {working_dir}")
        print(f"  Max iterations: {max_iterations}")
    
    def call_model(self, messages: List[Dict[str, str]]) -> str:
        """Call the model endpoint with the given messages."""
        try:
            import ssl
            import urllib3
            
            # Disable SSL warnings for testing
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            
            response = requests.post(
                self.model_endpoint,
                json={
                    "model": "gpt-4",  # or whatever model you're using
                    "messages": messages,
                    "temperature": 0.1,
                    "max_tokens": 4000
                },
                timeout=120,
                verify=False  # Disable SSL verification for testing
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error calling model: {e}")
            return ""
    
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
    """Main entry point for the CleanLeaderAgent script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean LeaderAgent for SWE-bench tasks")
    parser.add_argument("--model-endpoint", required=True, help="Model endpoint URL")
    parser.add_argument("--working-dir", required=True, help="Working directory")
    parser.add_argument("--max-iterations", type=int, default=3, help="Maximum iterations")
    parser.add_argument("--prompt", help="Task prompt (if not provided, will read from stdin)")
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = CleanLeaderAgent(
        model_endpoint=args.model_endpoint,
        working_dir=args.working_dir,
        max_iterations=args.max_iterations
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
