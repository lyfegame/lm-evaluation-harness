#!/usr/bin/env python3
"""
Results analysis script for SWE-bench evaluations.
Analyzes all results stored externally for easy review.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime

def analyze_results(results_dir: str = "./results"):
    """
    Analyze all evaluation results stored in the results directory.
    """
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    print(f"🔍 Analyzing results in: {results_dir}")
    print("=" * 60)
    
    # Find all summary files
    summary_files = list(results_path.glob("*/summary.json"))
    
    if not summary_files:
        print("❌ No summary files found")
        return
    
    print(f"📊 Found {len(summary_files)} evaluation results")
    print()
    
    # Load all results
    results = []
    for summary_file in summary_files:
        try:
            with open(summary_file, 'r') as f:
                result = json.load(f)
                result['result_dir'] = str(summary_file.parent)
                results.append(result)
        except Exception as e:
            print(f"⚠️  Error loading {summary_file}: {e}")
    
    if not results:
        print("❌ No valid results found")
        return
    
    # Create summary table
    print("📋 EVALUATION SUMMARY")
    print("-" * 60)
    print(f"{'Task ID':<25} {'Model':<20} {'Solve Rate':<12} {'Duration':<10}")
    print("-" * 60)
    
    total_solve_rate = 0
    total_duration = 0
    
    for result in results:
        task_id = result.get('task_id', 'Unknown')[:24]
        model_name = result.get('model_name', 'Unknown').split('/')[-1][:19]
        solve_rate = result.get('solve_rate', 0.0)
        duration = result.get('duration_seconds', 0.0)
        
        print(f"{task_id:<25} {model_name:<20} {solve_rate:<12.2%} {duration:<10.1f}s")
        
        total_solve_rate += solve_rate
        total_duration += duration
    
    print("-" * 60)
    print(f"{'AVERAGE':<25} {'':<20} {total_solve_rate/len(results):<12.2%} {total_duration/len(results):<10.1f}s")
    print()
    
    # Detailed analysis
    print("📈 DETAILED ANALYSIS")
    print("-" * 60)
    
    # Success rate by model
    model_stats = {}
    for result in results:
        model = result.get('model_name', 'Unknown')
        if model not in model_stats:
            model_stats[model] = {'total': 0, 'solved': 0, 'duration': 0}
        
        model_stats[model]['total'] += 1
        if result.get('solve_rate', 0) > 0:
            model_stats[model]['solved'] += 1
        model_stats[model]['duration'] += result.get('duration_seconds', 0)
    
    print("🤖 Model Performance:")
    for model, stats in model_stats.items():
        success_rate = stats['solved'] / stats['total'] if stats['total'] > 0 else 0
        avg_duration = stats['duration'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {model.split('/')[-1]:<20} {success_rate:<8.2%} ({stats['solved']}/{stats['total']}) avg: {avg_duration:.1f}s")
    
    print()
    
    # Task difficulty analysis
    task_stats = {}
    for result in results:
        task = result.get('task_id', 'Unknown')
        if task not in task_stats:
            task_stats[task] = {'total': 0, 'solved': 0}
        
        task_stats[task]['total'] += 1
        if result.get('solve_rate', 0) > 0:
            task_stats[task]['solved'] += 1
    
    print("🎯 Task Difficulty:")
    for task, stats in task_stats.items():
        success_rate = stats['solved'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {task:<30} {success_rate:<8.2%} ({stats['solved']}/{stats['total']})")
    
    print()
    
    # Export to CSV
    if results:
        df = pd.DataFrame(results)
        csv_file = results_path / "evaluation_summary.csv"
        df.to_csv(csv_file, index=False)
        print(f"📄 Results exported to: {csv_file}")
    
    # Show recent results
    print("\n🕒 RECENT RESULTS")
    print("-" * 60)
    
    # Sort by timestamp
    recent_results = sorted(results, key=lambda x: x.get('timestamp', ''), reverse=True)[:5]
    
    for result in recent_results:
        timestamp = result.get('timestamp', 'Unknown')
        task_id = result.get('task_id', 'Unknown')
        solve_rate = result.get('solve_rate', 0.0)
        duration = result.get('duration_seconds', 0.0)
        
        print(f"  {timestamp[:19]} | {task_id:<25} | {solve_rate:<8.2%} | {duration:.1f}s")
    
    print()
    print("✅ Analysis complete!")

def show_individual_result(result_dir: str):
    """
    Show detailed information for a specific result.
    """
    result_path = Path(result_dir)
    
    if not result_path.exists():
        print(f"❌ Result directory not found: {result_dir}")
        return
    
    summary_file = result_path / "summary.json"
    if not summary_file.exists():
        print(f"❌ Summary file not found: {summary_file}")
        return
    
    with open(summary_file, 'r') as f:
        result = json.load(f)
    
    print(f"📋 DETAILED RESULT: {result.get('task_id', 'Unknown')}")
    print("=" * 60)
    print(f"Model: {result.get('model_name', 'Unknown')}")
    print(f"Solve Rate: {result.get('solve_rate', 0.0):.2%}")
    print(f"Duration: {result.get('duration_seconds', 0.0):.1f} seconds")
    print(f"Timestamp: {result.get('timestamp', 'Unknown')}")
    print(f"Method: {result.get('evaluation_method', 'Unknown')}")
    print()
    
    # Show files in result directory
    print("📁 Files in result directory:")
    for file_path in sorted(result_path.iterdir()):
        if file_path.is_file():
            size = file_path.stat().st_size
            print(f"  {file_path.name:<30} {size:>8} bytes")
    
    # Show patch content if available
    patch_file = result_path / "predictions.jsonl"
    if patch_file.exists():
        print("\n🔍 Patch Content:")
        with open(patch_file, 'r') as f:
            prediction = json.load(f)
            patch_content = prediction.get('model_patch', '')
            print(f"Patch size: {len(patch_content)} characters")
            if len(patch_content) < 1000:
                print("Patch content:")
                print(patch_content)
            else:
                print("Patch content (first 500 chars):")
                print(patch_content[:500] + "...")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Show individual result
        show_individual_result(sys.argv[1])
    else:
        # Analyze all results
        analyze_results()
