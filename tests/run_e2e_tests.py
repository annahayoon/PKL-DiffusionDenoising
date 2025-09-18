#!/usr/bin/env python3
"""
Comprehensive end-to-end test runner for PKL-Guided Diffusion pipeline.

This script provides a unified interface for running all E2E tests with
various configuration options, reporting, and integration with CI/CD systems.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tempfile
import shutil

import yaml


class E2ETestRunner:
    """Main test runner for end-to-end pipeline tests."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.test_results = {}
        self.start_time = time.time()
    
    def setup_test_environment(self, temp_dir: Optional[Path] = None) -> Path:
        """Set up temporary test environment."""
        if temp_dir is None:
            temp_dir = Path(tempfile.mkdtemp(prefix="pkl_e2e_tests_"))
        
        # Create required directories
        directories = [
            "data/train/classless",
            "data/val/classless", 
            "checkpoints",
            "outputs",
            "logs"
        ]
        
        for dir_path in directories:
            (temp_dir / dir_path).mkdir(parents=True, exist_ok=True)
        
        print(f"Test environment created at: {temp_dir}")
        return temp_dir
    
    def run_test_suite(
        self, 
        test_suite: str, 
        verbose: bool = False,
        timeout: int = 600,
        coverage: bool = True
    ) -> Tuple[bool, str, Dict]:
        """Run a specific test suite."""
        print(f"\n{'='*60}")
        print(f"Running {test_suite} tests...")
        print(f"{'='*60}")
        
        # Map test suite names to pytest commands
        test_commands = {
            "data-pipeline": [
                "tests/test_e2e_pipeline.py::TestDataPipeline",
            ],
            "training-pipeline": [
                "tests/test_e2e_pipeline.py::TestTrainingPipeline",
            ],
            "inference-pipeline": [
                "tests/test_e2e_pipeline.py::TestInferencePipeline",
            ],
            "evaluation-pipeline": [
                "tests/test_e2e_pipeline.py::TestEvaluationPipeline",
            ],
            "performance-tests": [
                "tests/test_e2e_performance_profiling.py::TestPerformanceProfiling",
            ],
            "integration-workflows": [
                "tests/test_e2e_pipeline.py::TestIntegrationWorkflows",
            ],
            "all-e2e": [
                "tests/test_e2e_pipeline.py",
            ],
            "benchmarks": [
                "tests/test_e2e_performance_profiling.py::TestPerformanceBenchmarks",
            ]
        }
        
        if test_suite not in test_commands:
            return False, f"Unknown test suite: {test_suite}", {}
        
        # Build pytest command
        cmd = ["python", "-m", "pytest"]
        cmd.extend(test_commands[test_suite])
        
        # Add common options
        cmd.extend([
            "-v" if verbose else "-q",
            "--tb=short",
            f"--timeout={timeout}",
            "--disable-warnings"
        ])
        
        # Add coverage options
        if coverage:
            cmd.extend([
                "--cov=pkl_dg",
                "--cov-report=term-missing",
                "--cov-report=xml",
                f"--cov-report=html:htmlcov/{test_suite}"
            ])
        
        # Run the test
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout + 60  # Extra buffer for pytest overhead
            )
            
            success = result.returncode == 0
            execution_time = time.time() - start_time
            
            output = result.stdout + result.stderr
            
            # Parse test results
            test_info = {
                "execution_time": execution_time,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
            if success:
                print(f"✅ {test_suite} tests PASSED ({execution_time:.1f}s)")
            else:
                print(f"❌ {test_suite} tests FAILED ({execution_time:.1f}s)")
                if verbose:
                    print(f"Error output:\n{result.stderr}")
            
            return success, output, test_info
            
        except subprocess.TimeoutExpired:
            print(f"⏰ {test_suite} tests TIMED OUT after {timeout}s")
            return False, f"Test suite timed out after {timeout}s", {}
        
        except Exception as e:
            print(f"💥 {test_suite} tests CRASHED: {e}")
            return False, f"Test suite crashed: {e}", {}
    
    def run_performance_benchmarks(self, output_file: Optional[Path] = None) -> bool:
        """Run performance benchmarks with detailed reporting."""
        print(f"\n{'='*60}")
        print("Running Performance Benchmarks...")
        print(f"{'='*60}")
        
        cmd = [
            "python", "-m", "pytest",
            "tests/test_e2e_performance_profiling.py",
            "-v", "--tb=short",
            "--benchmark-only",
            "--timeout=1800"
        ]
        
        if output_file:
            cmd.extend([f"--benchmark-json={output_file}"])
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=2000
            )
            
            success = result.returncode == 0
            
            if success:
                print("✅ Performance benchmarks COMPLETED")
            else:
                print("❌ Performance benchmarks FAILED")
                print(f"Error: {result.stderr}")
            
            return success
            
        except subprocess.TimeoutExpired:
            print("⏰ Performance benchmarks TIMED OUT")
            return False
        except Exception as e:
            print(f"💥 Performance benchmarks CRASHED: {e}")
            return False
    
    def generate_test_report(self, output_file: Path) -> None:
        """Generate comprehensive test report."""
        total_time = time.time() - self.start_time
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_execution_time": total_time,
            "test_results": self.test_results,
            "summary": {
                "total_suites": len(self.test_results),
                "passed_suites": sum(1 for r in self.test_results.values() if r["success"]),
                "failed_suites": sum(1 for r in self.test_results.values() if not r["success"])
            }
        }
        
        # Write JSON report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print(f"\n{'='*60}")
        print("TEST EXECUTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total execution time: {total_time:.1f}s")
        print(f"Test suites run: {report['summary']['total_suites']}")
        print(f"Passed: {report['summary']['passed_suites']}")
        print(f"Failed: {report['summary']['failed_suites']}")
        
        if report['summary']['failed_suites'] > 0:
            print("\nFailed test suites:")
            for suite_name, result in self.test_results.items():
                if not result["success"]:
                    print(f"  - {suite_name}: {result.get('error', 'Unknown error')}")
        
        print(f"\nDetailed report saved to: {output_file}")
    
    def run_all_tests(
        self,
        test_suites: List[str],
        verbose: bool = False,
        coverage: bool = True,
        benchmarks: bool = False,
        output_dir: Optional[Path] = None
    ) -> bool:
        """Run all specified test suites."""
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Starting E2E test execution for PKL-Guided Diffusion")
        print(f"Test suites: {', '.join(test_suites)}")
        
        # Set up test environment
        temp_dir = self.setup_test_environment()
        
        try:
            # Run each test suite
            overall_success = True
            
            for suite in test_suites:
                success, output, info = self.run_test_suite(
                    suite, 
                    verbose=verbose, 
                    coverage=coverage
                )
                
                self.test_results[suite] = {
                    "success": success,
                    "output": output[:1000] if len(output) > 1000 else output,  # Truncate long output
                    "info": info
                }
                
                if not success:
                    overall_success = False
                    self.test_results[suite]["error"] = output
            
            # Run benchmarks if requested
            if benchmarks:
                benchmark_file = output_dir / "benchmark_results.json" if output_dir else None
                benchmark_success = self.run_performance_benchmarks(benchmark_file)
                
                self.test_results["benchmarks"] = {
                    "success": benchmark_success
                }
                
                if not benchmark_success:
                    overall_success = False
            
            # Generate report
            if output_dir:
                report_file = output_dir / "test_report.json"
                self.generate_test_report(report_file)
            
            return overall_success
            
        finally:
            # Cleanup temporary directory
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run end-to-end tests for PKL-Guided Diffusion pipeline"
    )
    
    parser.add_argument(
        "--suites",
        nargs="+",
        choices=[
            "data-pipeline",
            "training-pipeline", 
            "inference-pipeline",
            "evaluation-pipeline",
            "performance-tests",
            "integration-workflows",
            "all-e2e",
            "benchmarks"
        ],
        default=["all-e2e"],
        help="Test suites to run"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Disable coverage reporting"
    )
    
    parser.add_argument(
        "--benchmarks",
        action="store_true",
        help="Run performance benchmarks"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for reports"
    )
    
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Project root directory"
    )
    
    args = parser.parse_args()
    
    # Validate project root
    if not (args.project_root / "pkl_dg").exists():
        print(f"Error: Invalid project root {args.project_root}")
        print("Expected to find pkl_dg package directory")
        sys.exit(1)
    
    # Create test runner
    runner = E2ETestRunner(args.project_root)
    
    # Run tests
    success = runner.run_all_tests(
        test_suites=args.suites,
        verbose=args.verbose,
        coverage=not args.no_coverage,
        benchmarks=args.benchmarks,
        output_dir=args.output_dir
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
