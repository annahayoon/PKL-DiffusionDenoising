#!/usr/bin/env python3
"""
Main entry point for running pkl_dg.evaluation as a module.

This allows running: python -m pkl_dg.evaluation
"""

import os
import sys
from pathlib import Path

def main():
    """Main entry point that handles config path resolution."""
    # Change to project root directory so Hydra can find configs
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent  # pkl_dg/evaluation/__main__.py -> project root
    
    # Change working directory to project root
    original_cwd = os.getcwd()
    os.chdir(project_root)
    
    try:
        # Now import and run the evaluation main function
        from .evaluation import main as eval_main
        eval_main()
    finally:
        # Restore original working directory
        os.chdir(original_cwd)

if __name__ == "__main__":
    main()
