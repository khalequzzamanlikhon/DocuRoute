"""Entry point to run DocuRoute evaluation with Hermes-path isolation.

Usage (from project root):
    python run_eval.py --version hybrid_rrf_rerank
    python run_eval.py --version hybrid_rrf
    python run_eval.py --version baseline_vector_only

Note: --version automatically sets the RETRIEVAL_MODE env var so you don't
need to export it separately.
"""
import os
import sys

# Remove Hermes-agent venv paths that shadow the project's .venv
sys.path = [p for p in sys.path if "hermes" not in p or ".venv" in p]

import argparse
from evaluation.run_eval import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    os.environ["RETRIEVAL_MODE"] = args.version
    run(args.version)
