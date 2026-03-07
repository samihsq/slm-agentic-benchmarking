#!/usr/bin/env python3
"""
Test script for Criticality v2 sequence scoring mode.

Tests all 9 SLM models using local GGUF inference via llama-cpp-python.
"""

import argparse
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmarks.skills.criticality.v2.runner import CriticalityV2Runner

# Base paths
OLLAMA_BLOBS = "/Users/samih_sully/.ollama/models/blobs"
MODELS_DIR = str(Path(__file__).parent.parent / "models")

# All 9 models with their GGUF paths
MODEL_PATHS = {
    # --- Ollama blobs (standard llama.cpp compatible) ---
    "Falcon-H1-90M": f"{OLLAMA_BLOBS}/sha256-c763e52b3902e5834d602b46c779ae54fa7b82cd94a0e893d818b804e081bdfd",
    "Qwen3-0.6B": f"{OLLAMA_BLOBS}/sha256-7f4030143c1c477224c5434f8272c662a8b042079a0a584f0a27a1684fe2e1fa",
    "Gemma3-1B": f"{OLLAMA_BLOBS}/sha256-7cd4618c1faf8b7233c6c906dac1694b6a47684b37b8895d470ac688520b9c01",
    "DASD-4B": f"{OLLAMA_BLOBS}/sha256-11e95b9455d84dff393e43d161c238ab65ded003bc85c3f29b628e2461c8879d",
    "Phi4-Mini-Reasoning": f"{OLLAMA_BLOBS}/sha256-f4dd2368e6c32725dc1c5c5548ae9ee2724d6a79052952eb50b65e26288022c4",
    # --- ggml-org GGUFs (downloaded from HuggingFace) ---
    "Gemma3-4B": f"{MODELS_DIR}/gemma-3-4b-it-Q4_K_M.gguf",
    "Gemma3n-E2B": f"{MODELS_DIR}/gemma-3n-E2B-it-Q8_0.gguf",
    "Gemma3n-E4B": f"{MODELS_DIR}/gemma-3n-E4B-it-Q8_0.gguf",
    "GPT-OSS-20B": f"{MODELS_DIR}/gpt-oss-20b-mxfp4.gguf",
}


def test_model(model_name: str, model_path: str, limit: int = 5):
    """Test a single model with sequence scoring."""
    print(f"\n{'='*70}")
    print(f"Testing {model_name} in sequence mode")
    print(f"{'='*70}\n")
    
    try:
        runner = CriticalityV2Runner(
            model=model_name,
            model_path=model_path,
            verbose=True,
            concurrency=1,
        )
        
        result = runner.run(limit=limit, save_results=True)
        
        metrics = result["metrics"]
        print(f"\n{model_name} SUCCESS")
        print(f"   Accuracy: {metrics['top1_accuracy']*100:.1f}%")
        print(f"   Rank Correlation: {metrics['rank_correlation']:.4f}")
        print(f"   Calibration Error: {metrics['calibration_error']:.4f}")
        print(f"   Logprob extractions: {result['num_tasks']}/{limit}")
        
        return True
    except Exception as e:
        print(f"\n{model_name} FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Criticality v2 sequence scoring")
    parser.add_argument("--models", type=str, default="all",
                        help="Comma-separated model names or 'all' (default: all)")
    parser.add_argument("--limit", type=int, default=5,
                        help="Number of tasks per model (default: 5)")
    args = parser.parse_args()

    print("Criticality v2 Sequence Scoring Test")
    print("=" * 70)
    
    # Check llama-cpp-python
    try:
        import llama_cpp
        print("llama-cpp-python is installed")
    except ImportError:
        print("llama-cpp-python is NOT installed")
        print("\nInstall with:")
        print('  CMAKE_ARGS="-DGGML_METAL=on" pip install git+https://github.com/abetlen/llama-cpp-python.git')
        sys.exit(1)
    
    # Select models
    if args.models.lower() == "all":
        selected = list(MODEL_PATHS.keys())
    else:
        selected = [m.strip() for m in args.models.split(",")]
        for m in selected:
            if m not in MODEL_PATHS:
                print(f"Unknown model: {m}")
                print(f"Available: {', '.join(MODEL_PATHS.keys())}")
                sys.exit(1)
    
    print(f"Models: {', '.join(selected)}")
    print(f"Limit: {args.limit} tasks each")
    
    # Test models
    results = {}
    for model_name in selected:
        model_path = MODEL_PATHS[model_name]
        if not Path(model_path).exists():
            print(f"\n{model_name}: GGUF not found at {model_path}, skipping")
            results[model_name] = False
            continue
        results[model_name] = test_model(model_name, model_path, limit=args.limit)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for model, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"  {model}: {status}")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n{passed}/{total} models passed")
    
    if passed == total:
        print("All tests passed! Sequence scoring is working.")
    else:
        print("Some tests failed. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
