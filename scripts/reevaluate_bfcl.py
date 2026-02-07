#!/usr/bin/env python3
"""
Re-evaluate existing BFCL results with corrected evaluation logic.
This script reads trace.json files and re-computes match status without the lenient fallback.
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def normalize_value(v):
    """Normalize a value for comparison - handle whitespace variations."""
    if isinstance(v, str):
        # Normalize whitespace around punctuation
        v = re.sub(r'\s*,\s*', ',', v)  # Remove spaces around commas
        v = v.strip().lower()
    return v


def check_predicted_vs_expected(predicted_calls: list, ground_truth: list) -> tuple[bool, float]:
    """
    Check predicted function calls against ground truth.
    
    Predicted format: [{'name': 'func', 'arguments': {'arg': value}}]
    Ground truth format: [{'func': {'arg': [value]}}]
    """
    if not ground_truth:
        return False, 0.0
    
    if not predicted_calls:
        return False, 0.0
    
    total_expected = len(ground_truth)
    matched = 0
    partial_scores = []
    
    for gt in ground_truth:
        if not isinstance(gt, dict):
            continue
        
        for expected_func_name, expected_args in gt.items():
            expected_args = expected_args or {}
            best_score = 0.0
            func_matched = False
            
            # Find matching predicted call
            for pred in predicted_calls:
                if not isinstance(pred, dict):
                    continue
                
                # Get predicted function name (could be 'name' or 'function')
                pred_name = pred.get('name') or pred.get('function') or ''
                pred_args = pred.get('arguments') or pred.get('parameters') or {}
                
                if pred_name.lower() != expected_func_name.lower():
                    continue
                
                # Function name matches, now check arguments
                if not expected_args:
                    func_matched = True
                    best_score = 1.0
                    break
                
                arg_matches = 0
                for k, expected_vals in expected_args.items():
                    actual_val = pred_args.get(k)
                    
                    # Ground truth values are in lists
                    if isinstance(expected_vals, list):
                        for ev in expected_vals:
                            # Normalize both for comparison
                            if normalize_value(actual_val) == normalize_value(ev):
                                arg_matches += 1
                                break
                            # Also try string comparison
                            if str(actual_val).lower().strip() == str(ev).lower().strip():
                                arg_matches += 1
                                break
                    else:
                        if normalize_value(actual_val) == normalize_value(expected_vals):
                            arg_matches += 1
                
                score = arg_matches / len(expected_args)
                if score > best_score:
                    best_score = score
                if arg_matches == len(expected_args):
                    func_matched = True
                    best_score = 1.0
                    break
            
            if func_matched:
                matched += 1
            partial_scores.append(best_score)
    
    success = matched == total_expected
    avg_score = sum(partial_scores) / len(partial_scores) if partial_scores else 0.0
    
    return success, avg_score


def reevaluate_run(run_dir: Path):
    """Re-evaluate all results in a run directory."""
    results = {}
    
    for agent_dir in run_dir.iterdir():
        if not agent_dir.is_dir() or agent_dir.name == "archived":
            continue
        
        agent_name = agent_dir.name
        results[agent_name] = {"correct": 0, "total": 0, "old_correct": 0}
        
        # Process each trace
        for trace_dir in agent_dir.iterdir():
            if not trace_dir.is_dir():
                continue
            
            trace_file = trace_dir / "trace.json"
            if not trace_file.exists():
                continue
            
            with open(trace_file) as f:
                trace = json.load(f)
            
            results[agent_name]["total"] += 1
            
            # Track old result
            if trace.get("match"):
                results[agent_name]["old_correct"] += 1
            
            # Get ground truth from correct field
            correct_str = trace.get("correct", "[]")
            try:
                if isinstance(correct_str, str):
                    # Handle Python-style dict strings
                    import ast
                    ground_truth = ast.literal_eval(correct_str)
                else:
                    ground_truth = correct_str
            except:
                try:
                    ground_truth = json.loads(correct_str.replace("'", '"'))
                except:
                    ground_truth = []
            
            # Get predicted calls from predicted field
            predicted_str = trace.get("predicted", "[]")
            try:
                if isinstance(predicted_str, str):
                    import ast
                    predicted_calls = ast.literal_eval(predicted_str)
                else:
                    predicted_calls = predicted_str
            except:
                try:
                    predicted_calls = json.loads(predicted_str.replace("'", '"'))
                except:
                    predicted_calls = []
            
            # Re-evaluate using predicted vs ground truth
            new_match, new_score = check_predicted_vs_expected(predicted_calls, ground_truth)
            
            if new_match:
                results[agent_name]["correct"] += 1
            
            # Update trace file
            trace["match_original"] = trace.get("match")
            trace["confidence_original"] = trace.get("confidence")
            trace["match"] = new_match
            trace["confidence"] = new_score
            
            with open(trace_file, 'w') as f:
                json.dump(trace, f, indent=4)
        
        # Update results.jsonl
        results_file = agent_dir / "results.jsonl"
        if results_file.exists():
            updated_lines = []
            with open(results_file) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        task_id = r.get("task_id")
                        
                        # Find corresponding trace
                        trace_file = agent_dir / task_id / "trace.json"
                        if trace_file.exists():
                            with open(trace_file) as tf:
                                trace = json.load(tf)
                            r["success_original"] = r.get("success")
                            r["success"] = trace.get("match", False)
                        
                        updated_lines.append(json.dumps(r))
                    except:
                        updated_lines.append(line.strip())
            
            with open(results_file, 'w') as f:
                f.write('\n'.join(updated_lines) + '\n')
        
        # Update summary file
        summary_file = agent_dir / "summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                summary = json.load(f)
            
            summary["success_rate_original"] = summary.get("success_rate")
            total = results[agent_name]["total"]
            correct = results[agent_name]["correct"]
            summary["success_rate"] = correct / total if total > 0 else 0
            summary["reevaluated"] = True
            summary["reevaluated_at"] = datetime.now().isoformat()
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=4)
    
    return results


def main():
    print("=" * 70)
    print("🔄 RE-EVALUATING BFCL RESULTS (Strict Mode)")
    print("=" * 70)
    
    bfcl_dir = Path("results/bfcl")
    all_results = defaultdict(lambda: defaultdict(dict))
    
    for run_dir in sorted(bfcl_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name == "archived":
            continue
        
        print(f"\n📁 Processing: {run_dir.name}")
        results = reevaluate_run(run_dir)
        
        for agent, stats in results.items():
            if stats["total"] > 0:
                old_rate = stats["old_correct"] / stats["total"] * 100
                new_rate = stats["correct"] / stats["total"] * 100
                diff = new_rate - old_rate
                
                print(f"   {agent}: {old_rate:.0f}% → {new_rate:.0f}% ({diff:+.0f}%)")
                
                # Extract model from run_dir name
                model = run_dir.name.split("_")[0]
                all_results[model][agent] = {
                    "old": old_rate,
                    "new": new_rate,
                    "n": stats["total"]
                }
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 CORRECTED BFCL LEADERBOARD")
    print("=" * 70)
    print(f"\n{'Model':<18} {'OneShot':>12} {'Sequential':>12} {'Concurrent':>12} {'GroupChat':>12}")
    print("-" * 70)
    
    for model in sorted(all_results.keys()):
        agents = all_results[model]
        
        def fmt(agent):
            if agent in agents and agents[agent]["n"] >= 50:
                old = agents[agent]["old"]
                new = agents[agent]["new"]
                diff = new - old
                if diff != 0:
                    return f"{new:.0f}% ({diff:+.0f})"
                return f"{new:.0f}%"
            return "-"
        
        print(f"{model:<18} {fmt('OneShotAgent'):>12} {fmt('SequentialAgent'):>12} {fmt('ConcurrentAgent'):>12} {fmt('GroupChatAgent'):>12}")
    
    print("\n✅ Re-evaluation complete! Trace files updated with match_original and match fields.")


if __name__ == "__main__":
    main()
