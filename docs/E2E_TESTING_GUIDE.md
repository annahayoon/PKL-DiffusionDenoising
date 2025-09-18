# End-to-End Pipeline Testing Guide

This document provides comprehensive guidance for running and maintaining the end-to-end (E2E) testing framework for the PKL-Guided Diffusion system.

## Overview

The E2E testing framework validates the complete workflows of the PKL-Guided Diffusion pipeline, from data synthesis through training, inference, and evaluation. It ensures system reliability, performance, and correctness across all components.

## Test Architecture

### Test Categories

1. **Data Pipeline Tests** (`TestDataPipeline`)
   - Synthetic data creation and validation
   - Dataset loading and transformation
   - Data consistency and format verification

2. **Training Pipeline Tests** (`TestTrainingPipeline`)
   - Complete training workflow validation
   - Configuration handling and checkpoint creation
   - Different training configurations (EMA, schedules, etc.)

3. **Inference Pipeline Tests** (`TestInferencePipeline`)
   - End-to-end inference with pretrained models
   - Multiple guidance strategy validation
   - Input/output format verification

4. **Evaluation Pipeline Tests** (`TestEvaluationPipeline`)
   - Metrics computation (PSNR, SSIM, FRC, SAR, Hausdorff)
   - Robustness testing framework
   - Baseline method comparisons

5. **Performance Tests** (`TestPerformanceProfiling`)
   - Memory usage profiling and scaling
   - Execution time benchmarking
   - Resource utilization analysis
   - Memory leak detection

6. **Integration Workflow Tests** (`TestIntegrationWorkflows`)
   - Complete end-to-end workflows
   - Cross-component integration validation
   - Configuration validation and error handling

## Quick Start

### Prerequisites

```bash
# Install test dependencies
pip install -e .
pip install pytest pytest-cov pytest-xdist pytest-timeout pytest-benchmark psutil
```

### Running Tests

#### Using Make (Recommended)

```bash
# Run all E2E tests
make test-all

# Run specific test suites
make test-data          # Data pipeline tests
make test-training      # Training pipeline tests  
make test-inference     # Inference pipeline tests
make test-evaluation    # Evaluation pipeline tests
make test-integration   # Integration workflow tests
make test-performance   # Performance profiling tests

# Run fast tests only (< 5 minutes)
make test-fast

# Run comprehensive tests (may take 30+ minutes)
make test-slow

# Generate coverage report
make test-coverage
```

#### Using the Test Runner Script

```bash
# Run all E2E tests with detailed reporting
python scripts/run_e2e_tests.py --suites all-e2e --verbose --output-dir test_results

# Run specific test suites
python scripts/run_e2e_tests.py --suites data-pipeline training-pipeline --verbose

# Run with performance benchmarks
python scripts/run_e2e_tests.py --suites all-e2e --benchmarks --output-dir results
```

#### Using pytest Directly

```bash
# Run all E2E pipeline tests
pytest tests/test_e2e_pipeline.py -v

# Run specific test class
pytest tests/test_e2e_pipeline.py::TestDataPipeline -v

# Run performance tests
pytest tests/test_e2e_performance_profiling.py -v

# Run with coverage
pytest tests/test_e2e_pipeline.py --cov=pkl_dg --cov-report=html
```

## Test Configuration

### Environment Variables

- `USE_GPU_TESTS=true` - Enable GPU-specific tests
- `SKIP_SLOW_TESTS=true` - Skip slow-running tests
- `TEST_TIMEOUT=600` - Set test timeout in seconds

### Configuration Files

Tests use minimal configurations optimized for fast execution:

```python
# Minimal config for fast testing
minimal_config = {
    "model": {"sample_size": 32, "block_out_channels": [32, 32, 64]},
    "training": {"num_timesteps": 50, "max_epochs": 1, "batch_size": 2},
    "inference": {"ddim_steps": 5}
}

# Standard config for realistic testing  
standard_config = {
    "model": {"sample_size": 64, "block_out_channels": [64, 128, 256]},
    "training": {"num_timesteps": 200, "max_epochs": 2, "batch_size": 4},
    "inference": {"ddim_steps": 20}
}
```

## Test Data

### Synthetic Data Generation

Tests automatically generate synthetic data:

- **Training Images**: Synthetic microscopy-like images with Gaussian spots
- **Measurements**: Poisson-noisy measurements simulating WF microscopy
- **Ground Truth**: Clean images for evaluation

### Data Locations

```
temp_workspace/
├── data/
│   ├── train/classless/     # Training images
│   ├── val/classless/       # Validation images
│   └── synth/              # Synthesized pairs
├── checkpoints/            # Model checkpoints
├── inference_input/        # Test measurements
├── inference_output/       # Reconstruction results
└── logs/                  # Training logs
```

## Performance Testing

### Memory Profiling

```python
# Memory usage scaling test
def test_inference_memory_scaling():
    # Tests memory usage with different image sizes
    # Validates memory scaling behavior
    # Detects memory leaks
```

### Benchmarking

```bash
# Run performance benchmarks
make test-benchmarks

# Monitor performance trends
make monitor-performance

# Stress test memory usage
make stress-test-memory
```

### Performance Metrics

- **Execution Time**: End-to-end inference timing
- **Memory Usage**: Peak and average memory consumption
- **GPU Utilization**: GPU memory and compute usage
- **Scalability**: Performance vs. image size/batch size

## Continuous Integration

### GitHub Actions Workflow

The E2E tests are integrated with GitHub Actions:

```yaml
# .github/workflows/e2e-pipeline-tests.yml
- Matrix testing across Python versions (3.8-3.11)
- Parallel test execution by suite
- Coverage reporting with Codecov
- Performance benchmark tracking
- GPU testing on self-hosted runners
```

### CI Test Suites

```bash
# Fast CI tests (< 5 minutes)
make ci-test-fast

# Training pipeline CI tests
make ci-test-training  

# Inference pipeline CI tests
make ci-test-inference

# Integration workflow CI tests
make ci-test-integration
```

## Debugging and Troubleshooting

### Common Issues

1. **Memory Errors**
   ```bash
   # Check system memory
   make debug-test-env
   
   # Run memory-specific tests
   make test-memory
   ```

2. **Timeout Errors**
   ```bash
   # Increase timeout
   TEST_TIMEOUT=1200 make test-training
   
   # Run with minimal config
   pytest tests/test_e2e_pipeline.py::TestTrainingPipeline::test_minimal_training_run
   ```

3. **GPU Issues**
   ```bash
   # Check GPU availability
   python -c "import torch; print(torch.cuda.is_available())"
   
   # Run CPU-only tests
   pytest tests/ -m "not gpu"
   ```

### Debugging Tools

```bash
# Verbose test output
pytest tests/test_e2e_pipeline.py -v -s

# Debug specific test
pytest tests/test_e2e_pipeline.py::TestDataPipeline::test_synthetic_data_creation -v -s --tb=long

# Profile test performance
pytest tests/test_e2e_performance_profiling.py --profile

# Check test environment
make debug-test-env
```

## Test Development Guidelines

### Writing New E2E Tests

1. **Follow the Pattern**
   ```python
   class TestNewComponent(E2EPipelineTestBase):
       def test_component_functionality(self, temp_workspace, minimal_config):
           # Setup test data
           # Run component
           # Validate results
           # Check error conditions
   ```

2. **Use Fixtures**
   ```python
   def test_with_synthetic_data(self, synthetic_image_generator, temp_workspace):
       images = synthetic_image_generator(temp_workspace / "data", num_images=5)
       # Test with generated images
   ```

3. **Performance Considerations**
   - Use minimal configurations for fast tests
   - Mark slow tests with `@pytest.mark.slow`
   - Include memory cleanup in teardown

### Test Validation

```python
# Standard validation patterns
assert result.shape == expected_shape
assert torch.isfinite(result).all()
assert torch.all(result >= 0)  # For intensity domain
assert execution_time < max_acceptable_time
```

## Maintenance

### Regular Tasks

1. **Update Benchmarks**
   ```bash
   # Run monthly performance benchmarks
   make test-benchmarks
   ```

2. **Check Coverage**
   ```bash
   # Generate coverage reports
   make test-coverage
   ```

3. **Validate CI**
   ```bash
   # Test CI configurations locally
   make ci-test-fast
   ```

### Monitoring

- **Performance Trends**: Track benchmark results over time
- **Test Reliability**: Monitor test failure rates
- **Resource Usage**: Monitor CI resource consumption

## Advanced Usage

### Custom Test Configurations

```python
# Create custom config for specific testing needs
custom_config = minimal_config.copy()
custom_config.model.sample_size = 128
custom_config.training.use_ema = True

# Use in tests
def test_custom_scenario(self, temp_workspace):
    config = create_custom_config(temp_workspace)
    # Run test with custom config
```

### Parallel Testing

```bash
# Run tests in parallel
pytest tests/test_e2e_pipeline.py -n auto

# Distributed testing
pytest tests/ -n 4 --dist=worksteal
```

### Test Data Management

```python
# Custom test data generation
def create_specific_test_case():
    # Generate data for specific edge case
    # Return test measurement and expected result
```

## Integration with Development Workflow

### Pre-commit Hooks

```bash
# Install pre-commit hooks
make install-dev

# Run E2E tests before commit
make test-fast
```

### Release Testing

```bash
# Full test suite before release
make test-all

# Performance regression testing
make test-benchmarks

# Memory leak testing
make test-memory
```

## Support and Resources

### Documentation
- [Software Architecture](SOFTWARE_ARCHITECTURE.md) - System design overview
- [Implementation Plan](IMPLEMENTATION_PLAN.md) - Development roadmap
- [Quickstart Guide](QUICKSTART.md) - Getting started

### Debugging Resources
- Test logs in `logs/` directory
- Coverage reports in `htmlcov/`
- Performance reports in `benchmark_results.json`

### Getting Help

1. Check test logs for detailed error information
2. Run `make debug-test-env` to verify environment
3. Use verbose mode (`-v`) for detailed test output
4. Check GitHub Actions for CI-specific issues

---

This comprehensive E2E testing framework ensures the reliability and performance of the PKL-Guided Diffusion system across all major workflows and use cases.
