#!/usr/bin/env python3
"""
Generate statistical analysis and compile LaTeX results report.

Usage:
    python scripts/generate_results_report.py [--compile]
    
Options:
    --compile: Attempt to compile the LaTeX document to PDF (requires pdflatex)
"""

import json
import glob
import argparse
import subprocess
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


def wilson_ci(successes: int, n: int, confidence: float = 0.95) -> tuple:
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return 0, 0, 0
    p = successes / n
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4*n)) / n) / denominator
    return p, max(0, center - margin), min(1, center + margin)


def chi_square_test(p1: float, n1: int, p2: float, n2: int) -> float:
    """Chi-square test for two proportions."""
    successes1 = int(p1 * n1)
    successes2 = int(p2 * n2)
    table = [[successes1, n1 - successes1], [successes2, n2 - successes2]]
    try:
        chi2, p_value, dof, expected = stats.chi2_contingency(table)
        return p_value
    except:
        return 1.0


def load_results(results_dir: Path) -> dict:
    """Load all benchmark results from results directory."""
    results = defaultdict(dict)
    
    # Criticality - find latest run
    crit_dirs = sorted(glob.glob(str(results_dir / 'criticality' / '*' / 'OneShotAgent' / 'summary.json')))
    for f in crit_dirs:
        d = json.load(open(f))
        model = d['model']
        results[model]['criticality'] = {
            'accuracy': d.get('accuracy', 0),
            'num_tasks': d.get('num_tasks', 100),
            'avg_confidence': d.get('avg_confidence', 0),
            'avg_latency': d.get('avg_latency', 0),
        }
    
    # Episodic Memory
    epi_dirs = sorted(glob.glob(str(results_dir / 'episodic_memory' / '*' / 'OneShotAgent' / 'summary.json')))
    for f in epi_dirs:
        d = json.load(open(f))
        model = d['model']
        results[model]['episodic_memory'] = {
            'simple_recall_score': d.get('simple_recall_score', 0),
            'chronological_awareness_score': d.get('chronological_awareness_score', 0),
            'num_questions': d.get('num_questions', 100),
            'avg_latency': d.get('avg_latency', 0),
        }
    
    # Recall
    recall_dirs = sorted(glob.glob(str(results_dir / 'recall' / '*' / 'OneShotAgent' / 'summary.json')))
    for f in recall_dirs:
        d = json.load(open(f))
        model = d['model']
        results[model]['recall'] = {
            'accuracy': d.get('accuracy', 0),
            'num_tasks': d.get('num_tasks', 100),
            'avg_latency': d.get('avg_latency', 0),
        }
    
    return dict(results)


def generate_statistical_report(results: dict) -> str:
    """Generate comprehensive statistical analysis report."""
    report = []
    report.append("=" * 80)
    report.append("STATISTICAL ANALYSIS REPORT")
    report.append("=" * 80)
    
    # Criticality analysis
    report.append("\n1. CRITICALITY BENCHMARK")
    report.append("-" * 60)
    report.append(f"{'Model':<25} {'Accuracy':<12} {'95% CI':<25} {'n':<6}")
    
    crit_data = []
    for model in sorted(results.keys()):
        if 'criticality' in results[model]:
            acc = results[model]['criticality']['accuracy']
            n = results[model]['criticality']['num_tasks']
            successes = int(acc * n)
            p, ci_low, ci_high = wilson_ci(successes, n)
            crit_data.append((model, acc, ci_low, ci_high, n))
            report.append(f"{model:<25} {acc*100:>6.1f}%      [{ci_low*100:.1f}%, {ci_high*100:.1f}%]     {n}")
    
    # Episodic Memory analysis
    report.append("\n2. EPISODIC MEMORY BENCHMARK")
    report.append("-" * 60)
    report.append(f"{'Model':<25} {'F1 Score':<12} {'95% CI':<25} {'n':<6}")
    
    epi_data = []
    for model in sorted(results.keys()):
        if 'episodic_memory' in results[model]:
            score = results[model]['episodic_memory']['simple_recall_score']
            n = results[model]['episodic_memory']['num_questions']
            successes = int(score * n)
            p, ci_low, ci_high = wilson_ci(successes, n)
            epi_data.append((model, score, ci_low, ci_high, n))
            report.append(f"{model:<25} {score*100:>6.1f}%      [{ci_low*100:.1f}%, {ci_high*100:.1f}%]     {n}")
    
    # Recall analysis
    report.append("\n3. RECALL BENCHMARK")
    report.append("-" * 60)
    report.append(f"{'Model':<25} {'Accuracy':<12} {'95% CI':<25} {'n':<6}")
    
    recall_data = []
    for model in sorted(results.keys()):
        if 'recall' in results[model]:
            acc = results[model]['recall']['accuracy']
            n = results[model]['recall']['num_tasks']
            successes = int(acc * n)
            p, ci_low, ci_high = wilson_ci(successes, n)
            recall_data.append((model, acc, ci_low, ci_high, n))
            report.append(f"{model:<25} {acc*100:>6.1f}%      [{ci_low*100:.1f}%, {ci_high*100:.1f}%]     {n}")
    
    # Pairwise significance tests
    report.append("\n" + "=" * 80)
    report.append("PAIRWISE SIGNIFICANCE TESTS")
    report.append("=" * 80)
    
    # Criticality top 3 vs rest
    report.append("\nCriticality: Significant differences (p < 0.05)")
    sorted_crit = sorted(crit_data, key=lambda x: -x[1])
    for i, (m1, p1, _, _, n1) in enumerate(sorted_crit[:3]):
        for m2, p2, _, _, n2 in sorted_crit[3:]:
            pval = chi_square_test(p1, n1, p2, n2)
            if pval < 0.05:
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*"
                report.append(f"  {m1} vs {m2}: p={pval:.4f} {sig}")
    
    # Episodic Memory top 3 vs rest
    report.append("\nEpisodic Memory: Significant differences (p < 0.05)")
    sorted_epi = sorted([d for d in epi_data if d[1] > 0], key=lambda x: -x[1])
    for i, (m1, p1, _, _, n1) in enumerate(sorted_epi[:3]):
        for m2, p2, _, _, n2 in sorted_epi[3:]:
            pval = chi_square_test(p1, n1, p2, n2)
            if pval < 0.05:
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*"
                report.append(f"  {m1} vs {m2}: p={pval:.4f} {sig}")
    
    # Recall top vs rest
    report.append("\nRecall: Significant differences (p < 0.05)")
    sorted_recall = sorted([d for d in recall_data if d[1] > 0], key=lambda x: -x[1])
    for i, (m1, p1, _, _, n1) in enumerate(sorted_recall[:2]):
        for m2, p2, _, _, n2 in sorted_recall[2:]:
            pval = chi_square_test(p1, n1, p2, n2)
            if pval < 0.05:
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*"
                report.append(f"  {m1} vs {m2}: p={pval:.4f} {sig}")
    
    report.append("\n" + "=" * 80)
    report.append("KEY FINDINGS")
    report.append("=" * 80)
    
    # Find winners
    crit_winner = sorted_crit[0] if sorted_crit else None
    epi_winner = sorted_epi[0] if sorted_epi else None
    recall_winner = sorted_recall[0] if sorted_recall else None
    
    if crit_winner:
        report.append(f"\n• Criticality Winner: {crit_winner[0]} ({crit_winner[1]*100:.1f}%)")
    if epi_winner:
        report.append(f"• Episodic Memory Winner: {epi_winner[0]} ({epi_winner[1]*100:.1f}%)")
    if recall_winner:
        report.append(f"• Recall Winner: {recall_winner[0]} ({recall_winner[1]*100:.1f}%)")
    
    report.append("\n" + "=" * 80)
    
    return "\n".join(report)


def compile_latex(tex_file: Path) -> bool:
    """Attempt to compile LaTeX to PDF."""
    try:
        result = subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', str(tex_file)],
            cwd=tex_file.parent,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"✅ PDF compiled: {tex_file.with_suffix('.pdf')}")
            return True
        else:
            print(f"⚠️ pdflatex errors (may still produce PDF):")
            print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
            return tex_file.with_suffix('.pdf').exists()
    except FileNotFoundError:
        print("⚠️ pdflatex not found. Install TeX Live or MacTeX to compile.")
        return False


def main():
    parser = argparse.ArgumentParser(description='Generate benchmark results report')
    parser.add_argument('--compile', action='store_true', help='Compile LaTeX to PDF')
    args = parser.parse_args()
    
    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    results_dir = project_root / 'results'
    
    # Load results
    print("Loading benchmark results...")
    results = load_results(results_dir)
    
    if not results:
        print("❌ No results found in", results_dir)
        return
    
    print(f"✅ Found results for {len(results)} models")
    
    # Save consolidated results
    output_json = results_dir / 'all_benchmark_results.json'
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Saved consolidated results to {output_json}")
    
    # Generate statistical report
    report = generate_statistical_report(results)
    print("\n" + report)
    
    # Save report
    report_txt = results_dir / 'statistical_analysis.txt'
    with open(report_txt, 'w') as f:
        f.write(report)
    print(f"\n✅ Saved statistical report to {report_txt}")
    
    # Compile LaTeX if requested
    tex_file = results_dir / 'benchmark_results.tex'
    if args.compile and tex_file.exists():
        print("\nCompiling LaTeX...")
        compile_latex(tex_file)


if __name__ == '__main__':
    main()
