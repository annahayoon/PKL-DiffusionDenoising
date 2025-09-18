# Makefile for PKL-Guided Diffusion E2E Testing
# Provides convenient commands for running different test suites

.PHONY: help test-all test-e2e test-fast test-slow test-gpu test-benchmarks
.PHONY: test-data test-training test-inference test-evaluation test-integration
.PHONY: test-performance test-memory test-coverage clean-test setup-test-env
.PHONY: lint format install-dev install-test-deps

# Default target
help:
	@echo "PKL-Guided Diffusion E2E Testing Commands"
	@echo "========================================"
	@echo ""
	@echo "Test Suites:"
	@echo "  test-all          - Run all E2E tests"
	@echo "  test-e2e          - Run core E2E pipeline tests"
	@echo "  test-fast         - Run fast tests only"
	@echo "  test-slow         - Run slow/comprehensive tests"
	@echo "  test-gpu          - Run GPU-specific tests"
	@echo "  test-benchmarks   - Run performance benchmarks"
	@echo ""
	@echo "Individual Components:"
	@echo "  test-data         - Test data pipeline"
	@echo "  test-training     - Test training pipeline"
	@echo "  test-inference    - Test inference pipeline"
	@echo "  test-evaluation   - Test evaluation pipeline"
	@echo "  test-integration  - Test integration workflows"
	@echo "  test-performance  - Test performance profiling"
	@echo "  test-memory       - Test memory usage"
	@echo ""
	@echo "Development:"
	@echo "  test-coverage     - Generate coverage report"
	@echo "  setup-test-env    - Set up test environment"
	@echo "  clean-test        - Clean test artifacts"
	@echo "  lint              - Run code linting"
	@echo "  format            - Format code"
	@echo "  install-dev       - Install development dependencies"
	@echo "  install-test-deps - Install test dependencies"

# Environment variables
PYTHON ?= python3
PYTEST_OPTS ?= -v --tb=short
COVERAGE_OPTS ?= --cov=pkl_dg --cov-report=html --cov-report=term-missing
TEST_TIMEOUT ?= 600
NUM_WORKERS ?= auto

# Main test targets
test-all: install-test-deps
	@echo "🧪 Running all E2E tests..."
	$(PYTHON) scripts/run_e2e_tests.py --suites all-e2e performance-tests --verbose --benchmarks

test-e2e: install-test-deps
	@echo "🧪 Running E2E pipeline tests..."
	$(PYTHON) -m pytest tests/test_e2e_pipeline.py $(PYTEST_OPTS) --timeout=$(TEST_TIMEOUT)

test-fast: install-test-deps
	@echo "⚡ Running fast tests..."
	$(PYTHON) -m pytest tests/test_e2e_pipeline.py -m "not slow" $(PYTEST_OPTS) --timeout=300

test-slow: install-test-deps
	@echo "🐌 Running slow/comprehensive tests..."
	$(PYTHON) -m pytest tests/test_e2e_pipeline.py -m "slow" $(PYTEST_OPTS) --timeout=1200

test-gpu: install-test-deps
	@echo "🚀 Running GPU tests..."
	USE_GPU_TESTS=true $(PYTHON) -m pytest tests/ -m "gpu" $(PYTEST_OPTS) --timeout=$(TEST_TIMEOUT)

test-benchmarks: install-test-deps
	@echo "📊 Running performance benchmarks..."
	$(PYTHON) -m pytest tests/test_e2e_performance_profiling.py::TestPerformanceBenchmarks \
		--benchmark-only --benchmark-json=benchmark_results.json $(PYTEST_OPTS)

# Individual component tests
test-data: install-test-deps
	@echo "📁 Testing data pipeline..."
	$(PYTHON) scripts/run_e2e_tests.py --suites data-pipeline --verbose

test-training: install-test-deps
	@echo "🏋️ Testing training pipeline..."
	$(PYTHON) scripts/run_e2e_tests.py --suites training-pipeline --verbose

test-inference: install-test-deps
	@echo "🔮 Testing inference pipeline..."
	$(PYTHON) scripts/run_e2e_tests.py --suites inference-pipeline --verbose

test-evaluation: install-test-deps
	@echo "📏 Testing evaluation pipeline..."
	$(PYTHON) scripts/run_e2e_tests.py --suites evaluation-pipeline --verbose

test-integration: install-test-deps
	@echo "🔗 Testing integration workflows..."
	$(PYTHON) scripts/run_e2e_tests.py --suites integration-workflows --verbose

test-performance: install-test-deps
	@echo "⚡ Testing performance profiling..."
	$(PYTHON) -m pytest tests/test_e2e_performance_profiling.py::TestPerformanceProfiling \
		$(PYTEST_OPTS) --timeout=1800

test-memory: install-test-deps
	@echo "🧠 Testing memory usage..."
	$(PYTHON) -m pytest tests/test_e2e_performance_profiling.py -k "memory" \
		$(PYTEST_OPTS) --timeout=900

# Coverage and reporting
test-coverage: install-test-deps
	@echo "📊 Generating test coverage report..."
	$(PYTHON) -m pytest tests/test_e2e_pipeline.py $(COVERAGE_OPTS) $(PYTEST_OPTS)
	@echo "📄 Coverage report generated in htmlcov/"

# Development utilities
setup-test-env:
	@echo "🔧 Setting up test environment..."
	mkdir -p data/train/classless data/val/classless checkpoints outputs logs
	mkdir -p htmlcov benchmark_results
	@echo "✅ Test environment ready"

clean-test:
	@echo "🧹 Cleaning test artifacts..."
	rm -rf htmlcov/ .coverage coverage.xml benchmark_results.json
	rm -rf .pytest_cache/ __pycache__/ */__pycache__/ */*/__pycache__/
	rm -rf test_results/ temp_test_*
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	@echo "✅ Test artifacts cleaned"

lint:
	@echo "🔍 Running code linting..."
	$(PYTHON) -m flake8 pkl_dg/ tests/ scripts/ --max-line-length=100 --ignore=E203,W503
	$(PYTHON) -m pylint pkl_dg/ --disable=C0103,R0903,R0913,W0613

format:
	@echo "🎨 Formatting code..."
	$(PYTHON) -m black pkl_dg/ tests/ scripts/ --line-length=100
	$(PYTHON) -m isort pkl_dg/ tests/ scripts/ --profile=black

install-dev:
	@echo "📦 Installing development dependencies..."
	$(PYTHON) -m pip install -e .
	$(PYTHON) -m pip install black isort flake8 pylint pre-commit
	pre-commit install

install-test-deps:
	@echo "📦 Installing test dependencies..."
	$(PYTHON) -m pip install -e .
	$(PYTHON) -m pip install pytest pytest-cov pytest-xdist pytest-timeout pytest-benchmark
	$(PYTHON) -m pip install psutil tifffile Pillow

# Continuous Integration targets
ci-test-fast:
	@echo "🏗️ Running CI fast tests..."
	$(PYTHON) -m pytest tests/test_e2e_pipeline.py::TestDataPipeline \
		tests/test_e2e_pipeline.py::TestEvaluationPipeline \
		$(PYTEST_OPTS) --timeout=300 $(COVERAGE_OPTS)

ci-test-training:
	@echo "🏗️ Running CI training tests..."
	$(PYTHON) -m pytest tests/test_e2e_pipeline.py::TestTrainingPipeline \
		$(PYTEST_OPTS) --timeout=600 $(COVERAGE_OPTS)

ci-test-inference:
	@echo "🏗️ Running CI inference tests..."
	$(PYTHON) -m pytest tests/test_e2e_pipeline.py::TestInferencePipeline \
		$(PYTEST_OPTS) --timeout=600 $(COVERAGE_OPTS)

ci-test-integration:
	@echo "🏗️ Running CI integration tests..."
	$(PYTHON) -m pytest tests/test_e2e_pipeline.py::TestIntegrationWorkflows \
		$(PYTEST_OPTS) --timeout=900 $(COVERAGE_OPTS)

# Performance monitoring
monitor-performance:
	@echo "📈 Monitoring performance trends..."
	$(PYTHON) scripts/run_e2e_tests.py --suites benchmarks --output-dir performance_reports
	@echo "📊 Performance report saved to performance_reports/"

# Stress testing
stress-test-memory:
	@echo "💪 Running memory stress tests..."
	$(PYTHON) -m pytest tests/test_e2e_performance_profiling.py -k "memory or scaling" \
		$(PYTEST_OPTS) --timeout=1800

stress-test-concurrent:
	@echo "💪 Running concurrent stress tests..."
	$(PYTHON) -m pytest tests/test_e2e_pipeline.py -n $(NUM_WORKERS) \
		$(PYTEST_OPTS) --timeout=1200

# Documentation generation
docs-test-results:
	@echo "📚 Generating test documentation..."
	$(PYTHON) -m pytest tests/ --collect-only --quiet | grep "test_" > test_inventory.txt
	@echo "📄 Test inventory saved to test_inventory.txt"

# Debugging helpers
debug-test-env:
	@echo "🐛 Debugging test environment..."
	@echo "Python version: $$($(PYTHON) --version)"
	@echo "PyTorch version: $$($(PYTHON) -c 'import torch; print(torch.__version__)')"
	@echo "CUDA available: $$($(PYTHON) -c 'import torch; print(torch.cuda.is_available())')"
	@echo "GPU count: $$($(PYTHON) -c 'import torch; print(torch.cuda.device_count())')"
	@echo "Working directory: $$(pwd)"
	@echo "Test data directories:"
	@ls -la data/ 2>/dev/null || echo "  No data directory found"

# Help for specific test categories
help-performance:
	@echo "Performance Testing Help"
	@echo "======================="
	@echo "test-benchmarks    - Standard performance benchmarks"
	@echo "test-performance   - Detailed performance profiling"
	@echo "test-memory        - Memory usage analysis"
	@echo "stress-test-memory - Memory stress testing"
	@echo "monitor-performance- Performance trend monitoring"

help-ci:
	@echo "Continuous Integration Help"
	@echo "=========================="
	@echo "ci-test-fast       - Fast tests for CI"
	@echo "ci-test-training   - Training pipeline for CI"
	@echo "ci-test-inference  - Inference pipeline for CI"
	@echo "ci-test-integration- Integration tests for CI"
