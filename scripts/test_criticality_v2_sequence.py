#!/usr/bin/env python3
"""
Test script for Criticality v2 sequence scoring mode.

Run this on the remote machine (10.27.102.240) to verify the implementation.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmarks.skills.criticality.v2.runner import CriticalityV2Runner

# Model paths on remote machine
MODEL_PATHS = {
    "DASD-4B": "/Users/samih_sully/.ollama/models/blobs/sha256-11e95b9455d84dff393e43d161c238ab65ded003bc85c3f29b628e2461c8879d",
    "Qwen3-0.6B": "/Users/samih_sully/.ollama/models/blobs/sha256-7f4030143c1c477224c5434f8272c662a8b042079a0a584f0a27a1684fe2e1fa",
    "Falcon-H1-90M": "/Users/samih_sully/.ollama/models/blobs/sha256-c763e52b3902e5834d602b46c779ae54fa7b82cd94a0e893d818b804e081bdfd",
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
        print(f"\n✅ {model_name} SUCCESS")
        print(f"   Accuracy: {metrics['top1_accuracy']*100:.1f}%")
        print(f"   Rank Correlation: {metrics['rank_correlation']:.4f}")
        print(f"   Calibration Error: {metrics['calibration_error']:.4f}")
        print(f"   Logprob extractions: {result['num_tasks']}/{limit}")
        
        return True
    except Exception as e:
        print(f"\n❌ {model_name} FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("Criticality v2 Sequence Scoring Test")
    print("=" * 70)
    
    # Check llama-cpp-python
    try:
        import llama_cpp
        print("✓ llama-cpp-python is installed")
    except ImportError:
        print("❌ llama-cpp-python is NOT installed")
        print("\nInstall with:")
        print("  CMAKE_ARGS=\"-DGGML_METAL=on\" pip install llama-cpp-python")
        sys.exit(1)
    
    # Test models
    results = {}
    
    # Test DASD-4B (should work - was working in API mode)
    results["DASD-4B"] = test_model("DASD-4B", MODEL_PATHS["DASD-4B"], limit=5)
    
    # Test Qwen3-0.6B (the real test - previously failed in API mode)
    results["Qwen3-0.6B"] = test_model("Qwen3-0.6B", MODEL_PATHS["Qwen3-0.6B"], limit=5)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for model, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {model}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All tests passed! Sequence scoring is working.")
    else:
        print("\n⚠️  Some tests failed. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
