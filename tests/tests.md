# PKL-Guided Diffusion: End-to-End Testing Framework

## Overview

This document describes the comprehensive end-to-end (E2E) testing framework for the PKL-Guided Diffusion system. The framework validates complete workflows from data synthesis through training, inference, and evaluation, ensuring system reliability, performance, and correctness across all components.

## Test Architecture

### Core Components

The E2E testing framework consists of several interconnected components:

1. **Pipeline Tests** - Validate complete workflows
2. **Performance Tests** - Profile memory usage and execution time
3. **Integration Tests** - Test cross-component interactions
4. **Automation Framework** - CI/CD integration and test orchestration

### Test Categories

#### 1. Data Pipeline Tests (`TestDataPipeline`)
- **Synthetic Data Creation**: Validates generation of microscopy-like training data
- **Dataset Loading**: Tests `SynthesisDataset` with different configurations
- **Transform Consistency**: Validates reversible intensity ↔ model domain transforms
- **Format Verification**: Ensures proper data formats and shapes

```python
def test_synthetic_data_creation(self, temp_workspace, minimal_config):
    # Creates training/validation datasets
    # Tests forward model application
    # Validates data loading pipeline
```

#### 2. Training Pipeline Tests (`TestTrainingPipeline`)
- **Minimal Training**: Fast validation with reduced parameters
- **Configuration Handling**: Tests different training options (EMA, schedules, batch sizes)
- **Checkpoint Creation**: Validates model saving and loading
- **Error Handling**: Tests invalid configurations

```python
def test_minimal_training_run(self, temp_workspace, minimal_config):
    # Runs complete training workflow
    # Validates checkpoint creation
    # Tests model state preservation
```

#### 3. Inference Pipeline Tests (`TestInferencePipeline`)
- **Pretrained Model Inference**: Tests loading and using trained models
- **Guidance Strategy Comparison**: Validates PKL, L2, and Anscombe guidance
- **Input/Output Validation**: Ensures proper format handling
- **Configuration Flexibility**: Tests different inference parameters

```python
def test_inference_with_different_guidance_strategies(self, temp_workspace, minimal_config):
    # Tests PKL, L2, and Anscombe guidance
    # Validates different reconstruction results
    # Ensures proper error handling
```

#### 4. Evaluation Pipeline Tests (`TestEvaluationPipeline`)
- **Metrics Computation**: Tests PSNR, SSIM, FRC, SAR, Hausdorff distance
- **Robustness Framework**: PSF mismatch and alignment error testing
- **Baseline Comparisons**: Integration with Richardson-Lucy and other methods
- **Statistical Validation**: Ensures proper metric calculation

```python
def test_metrics_computation(self):
    # Tests all evaluation metrics
    # Validates numerical accuracy
    # Ensures proper error handling
```

#### 5. Performance Pipeline Tests (`TestPerformancePipeline`)
- **Memory Efficiency**: Validates memory usage and cleanup
- **Batch Processing**: Tests scalability with different batch sizes
- **Resource Monitoring**: CPU/GPU utilization tracking
- **Execution Time**: Performance benchmarking

```python
def test_memory_efficiency(self, temp_workspace, minimal_config):
    # Tests memory usage patterns
    # Validates cleanup after inference
    # Ensures no memory leaks
```

#### 6. Integration Workflow Tests (`TestIntegrationWorkflows`)
- **End-to-End Workflows**: Complete data → training → inference → evaluation
- **Cross-Component Integration**: Tests component interactions
- **Configuration Validation**: Ensures proper config handling
- **Error Recovery**: Tests graceful failure handling

```python
def test_full_training_to_inference_workflow(self, temp_workspace, minimal_config):
    # Complete workflow validation
    # Tests all pipeline stages
    # Validates final results
```

## Performance Profiling Framework

### Memory Profiling (`MemoryProfiler`)

Advanced memory monitoring with background thread tracking:

```python
class MemoryProfiler:
    def __init__(self, device: str = "cpu"):
        # Supports both CPU and GPU memory tracking
        
    def __enter__(self):
        # Starts background monitoring
        # Records baseline memory usage
        
    def get_metrics(self) -> Dict[str, float]:
        # Returns peak, average, and delta memory usage
```

### Performance Test Categories

#### 1. Memory Scaling Tests
- **Image Size Scaling**: Tests memory usage vs. image dimensions
- **Batch Size Impact**: Validates memory scaling with batch processing
- **Memory Leak Detection**: Monitors for memory accumulation over time

#### 2. Execution Time Benchmarks
- **Guidance Strategy Performance**: Compares PKL vs L2 vs Anscombe timing
- **DDIM Steps Tradeoff**: Analyzes quality vs speed with different step counts
- **Device Performance**: CPU vs GPU comparison when available

#### 3. Scalability Analysis
- **Concurrent Processing**: Tests parallel inference capabilities
- **Resource Utilization**: Monitors CPU/GPU/memory usage patterns
- **Bottleneck Identification**: Identifies performance limiting factors

## Test Configuration System

### Configuration Levels

#### 1. Minimal Configuration (Fast Testing)
```yaml
# Optimized for speed (~30 seconds per test)
model:
  sample_size: 32
  block_out_channels: [32, 32, 64]
training:
  num_timesteps: 50
  max_epochs: 1
  batch_size: 2
inference:
  ddim_steps: 5
```

#### 2. Standard Configuration (Realistic Testing)
```yaml
# Balanced for accuracy (~2-5 minutes per test)
model:
  sample_size: 64
  block_out_channels: [64, 128, 256]
training:
  num_timesteps: 200
  max_epochs: 2
  batch_size: 4
inference:
  ddim_steps: 20
```

### Fixture System

#### Core Fixtures
- `temp_workspace`: Isolated test environment with cleanup
- `minimal_test_config`: Fast testing configuration
- `standard_test_config`: Realistic testing configuration
- `synthetic_image_generator`: Creates microscopy-like test images
- `measurement_generator`: Creates noisy measurement data
- `performance_monitor`: Tracks execution metrics

#### Example Usage
```python
def test_example(self, temp_workspace, minimal_config, synthetic_image_generator):
    # Create test data
    images = synthetic_image_generator(temp_workspace / "data", num_images=5)
    
    # Run test with minimal config
    result = run_pipeline_component(minimal_config)
    
    # Validate results
    assert result.shape == expected_shape
```

## Automation Framework

### Test Runner Script (`scripts/run_e2e_tests.py`)

Unified interface for running all E2E tests:

```bash
# Run all E2E tests with reporting
python scripts/run_e2e_tests.py --suites all-e2e --verbose --output-dir results

# Run specific test suites
python scripts/run_e2e_tests.py --suites data-pipeline training-pipeline

# Run with performance benchmarks
python scripts/run_e2e_tests.py --suites all-e2e --benchmarks
```

#### Features
- **Suite Selection**: Choose specific test categories
- **Parallel Execution**: Run tests concurrently when possible
- **Comprehensive Reporting**: JSON reports with detailed metrics
- **Performance Integration**: Automatic benchmark collection
- **CI/CD Ready**: Designed for automated environments

### Makefile Integration

Convenient commands for common testing scenarios:

```bash
# Core test commands
make test-all          # Run complete test suite
make test-fast         # Quick validation (< 5 minutes)
make test-slow         # Comprehensive tests (30+ minutes)

# Component-specific tests
make test-data         # Data pipeline validation
make test-training     # Training workflow tests
make test-inference    # Inference pipeline tests
make test-evaluation   # Metrics and evaluation tests
make test-integration  # End-to-end workflow tests

# Performance analysis
make test-performance  # Performance profiling
make test-memory      # Memory usage analysis
make test-benchmarks  # Standardized benchmarks

# Development utilities
make test-coverage    # Generate coverage reports
make setup-test-env   # Prepare test environment
make clean-test      # Clean test artifacts
```

### GitHub Actions CI/CD

Automated testing with matrix execution:

```yaml
# .github/workflows/e2e-pipeline-tests.yml
strategy:
  matrix:
    python-version: [3.8, 3.9, "3.10", "3.11"]
    test-suite: [
      "data-pipeline",
      "training-pipeline", 
      "inference-pipeline",
      "evaluation-pipeline",
      "integration-workflows"
    ]
```

#### CI Features
- **Matrix Testing**: Multiple Python versions and test suites
- **Parallel Execution**: Faster CI runs with concurrent testing
- **Coverage Reporting**: Integration with Codecov
- **Performance Tracking**: Benchmark regression detection
- **GPU Testing**: Support for self-hosted GPU runners
- **Automated Reporting**: Test result summaries in PR comments

## Test Data Management

### Synthetic Data Generation

Automated creation of realistic test data:

#### Training Images
```python
def create_synthetic_images(output_dir, num_images=5, image_size=32):
    # Creates microscopy-like images with:
    # - Gaussian spots (simulating cells/structures)
    # - Realistic background and noise
    # - Proper intensity ranges
    # - Multiple formats (PNG for training, TIFF for measurements)
```

#### Measurement Data
```python
def create_test_measurements(output_dir, num_images=3, noise_level=0.1):
    # Creates noisy measurements simulating WF microscopy:
    # - Poisson noise (realistic for photon counting)
    # - Gaussian background
    # - Proper intensity scaling
```

### Data Pipeline Validation

#### Format Consistency
- **Input Validation**: Ensures proper image formats and dimensions
- **Transform Verification**: Validates intensity ↔ model domain conversions
- **Batch Processing**: Tests data loading with different batch sizes
- **Memory Efficiency**: Monitors data loading memory usage

#### Reproducibility
- **Fixed Seeds**: Consistent synthetic data generation
- **Deterministic Processing**: Reproducible test results
- **Version Control**: Test data generation scripts under version control

## Performance Benchmarking

### Standard Benchmarks

#### Inference Benchmark
```python
def test_standard_inference_benchmark():
    # Standardized test configuration:
    # - 256x256 images (reduced to 64x64 for CI)
    # - 50 DDIM steps
    # - PKL guidance
    # - Batch size 1
    
    # Measures:
    # - Total execution time
    # - Peak memory usage
    # - Average memory usage
```

#### Memory Scaling Benchmark
```python
def test_inference_memory_scaling():
    # Tests memory usage across image sizes: 32x32, 64x64, 128x128
    # Validates linear/quadratic scaling behavior
    # Detects memory leaks and inefficiencies
```

#### Guidance Strategy Comparison
```python
def test_guidance_strategy_performance():
    # Compares PKL vs L2 vs Anscombe:
    # - Execution time
    # - Memory usage
    # - Result quality (variance as proxy)
```

### Performance Monitoring

#### Continuous Tracking
- **Benchmark History**: Track performance trends over time
- **Regression Detection**: Alert on significant performance degradation
- **Resource Usage**: Monitor CI resource consumption
- **Optimization Validation**: Verify performance improvements

#### Reporting
- **JSON Reports**: Machine-readable performance data
- **HTML Dashboards**: Visual performance trend analysis
- **CI Integration**: Performance metrics in pull request comments

## Error Handling and Validation

### Robust Error Detection

#### Configuration Validation
```python
def test_configuration_validation_workflow():
    # Tests invalid configurations:
    # - Missing required parameters
    # - Invalid model architectures
    # - Incompatible training settings
    # - Unknown guidance types
```

#### Resource Limits
```python
# Automatic detection and handling:
skip_if_memory_limited()    # Skip tests on low-memory systems
skip_if_no_gpu()           # Skip GPU tests when unavailable
skip_if_slow_tests_disabled()  # Respect CI time limits
```

#### Graceful Degradation
- **Timeout Protection**: Prevents hanging tests
- **Memory Limit Handling**: Graceful failure on resource exhaustion
- **Device Fallbacks**: Automatic CPU fallback when GPU unavailable
- **Configuration Adaptation**: Reduced parameters for constrained environments

### Validation Patterns

#### Standard Assertions
```python
# Common validation patterns used throughout tests:
assert result.shape == expected_shape
assert torch.isfinite(result).all()
assert torch.all(result >= 0)  # For intensity domain
assert execution_time < max_acceptable_time
assert memory_usage < memory_limit
```

#### Statistical Validation
```python
# For metrics and evaluation:
assert 0 <= ssim_value <= 1
assert psnr_value > minimum_acceptable_psnr
assert frc_resolution > 0
```

## Development Integration

### Pre-commit Testing

```bash
# Fast validation before commits
make test-fast

# Specific component validation
make test-data  # If modifying data pipeline
make test-training  # If modifying training code
```

### Development Workflow

#### 1. Feature Development
```bash
# During development
make test-fast  # Quick validation

# Before commit
make test-coverage  # Ensure test coverage
```

#### 2. Performance Validation
```bash
# After optimization changes
make test-performance

# Benchmark comparison
make test-benchmarks
```

#### 3. Release Preparation
```bash
# Comprehensive validation
make test-all

# Performance regression check
make monitor-performance
```

### Debugging Support

#### Verbose Testing
```bash
# Detailed test output
pytest tests/test_e2e_pipeline.py -v -s

# Debug specific test
pytest tests/test_e2e_pipeline.py::TestDataPipeline::test_synthetic_data_creation -v -s --tb=long
```

#### Environment Debugging
```bash
# Check test environment
make debug-test-env

# Outputs:
# - Python version
# - PyTorch version  
# - CUDA availability
# - GPU count
# - Working directory
# - Test data directories
```

## Maintenance and Monitoring

### Regular Maintenance Tasks

#### Weekly
- **Performance Benchmarks**: Run comprehensive performance analysis
- **Coverage Review**: Ensure test coverage remains high
- **CI Health Check**: Monitor CI success rates and execution times

#### Monthly
- **Benchmark Comparison**: Compare performance trends
- **Test Reliability**: Analyze test failure patterns
- **Resource Usage**: Monitor CI resource consumption

#### Before Releases
- **Full Test Suite**: Run complete E2E test suite
- **Performance Regression**: Validate no performance degradation
- **Documentation Update**: Ensure test documentation is current

### Monitoring Dashboards

#### Test Metrics
- **Test Success Rate**: Percentage of passing tests over time
- **Execution Time Trends**: Monitor test execution duration
- **Coverage Metrics**: Track code coverage percentages
- **Failure Analysis**: Categorize and track test failures

#### Performance Metrics
- **Inference Time**: Track inference performance over time
- **Memory Usage**: Monitor memory consumption trends
- **Resource Utilization**: CPU/GPU usage patterns
- **Scalability**: Performance vs. problem size relationships

## Advanced Usage

### Custom Test Scenarios

#### Creating Specialized Tests
```python
class TestCustomScenario(E2EPipelineTestBase):
    def test_edge_case_handling(self, temp_workspace):
        # Create custom configuration for specific edge case
        custom_config = self.create_custom_config()
        
        # Generate specialized test data
        test_data = self.create_edge_case_data()
        
        # Run pipeline with custom scenario
        result = run_pipeline(custom_config, test_data)
        
        # Validate edge case handling
        assert validate_edge_case_result(result)
```

#### Parameterized Testing
```python
@pytest.mark.parametrize("guidance_type", ["pkl", "l2", "anscombe"])
@pytest.mark.parametrize("image_size", [32, 64, 128])
def test_guidance_scaling(self, guidance_type, image_size):
    # Test all combinations of guidance types and image sizes
    # Validates consistent behavior across parameters
```

### Integration with External Tools

#### Memory Profiling
```python
# Integration with memory_profiler
@profile
def test_memory_intensive_operation():
    # Detailed line-by-line memory profiling
    # Identifies memory hotspots
```

#### Performance Profiling
```python
# Integration with cProfile
def test_with_profiling():
    import cProfile
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run test operation
    result = run_operation()
    
    profiler.disable()
    profiler.dump_stats('profile_output.prof')
```

## Troubleshooting Guide

### Common Issues

#### 1. Memory Errors
```bash
# Symptoms: OOM errors, slow tests
# Solutions:
make debug-test-env  # Check available memory
TEST_TIMEOUT=1200 make test-training  # Increase timeout
pytest tests/ -k "not memory_intensive"  # Skip memory tests
```

#### 2. Timeout Errors
```bash
# Symptoms: Tests hanging or timing out
# Solutions:
pytest tests/test_e2e_pipeline.py::TestTrainingPipeline::test_minimal_training_run  # Test specific component
make test-fast  # Use minimal configurations
```

#### 3. GPU Issues
```bash
# Symptoms: CUDA errors, GPU unavailable
# Solutions:
python -c "import torch; print(torch.cuda.is_available())"  # Check GPU
pytest tests/ -m "not gpu"  # Skip GPU tests
USE_GPU_TESTS=false make test-all  # Disable GPU testing
```

#### 4. Import Errors
```bash
# Symptoms: Module not found errors
# Solutions:
pip install -e .  # Reinstall package in development mode
make install-test-deps  # Install test dependencies
```

### Debugging Strategies

#### 1. Isolate the Problem
```bash
# Test individual components
make test-data
make test-training  
make test-inference

# Test specific functions
pytest tests/test_e2e_pipeline.py::TestDataPipeline::test_synthetic_data_creation -v
```

#### 2. Increase Verbosity
```bash
# Get detailed output
pytest tests/ -v -s --tb=long

# Show all print statements
pytest tests/ -s
```

#### 3. Check Environment
```bash
# Validate test environment
make debug-test-env

# Check dependencies
pip list | grep -E "(torch|pytest|numpy)"
```

## Future Enhancements

### Planned Improvements

#### 1. Advanced Performance Analysis
- **GPU Memory Profiling**: Detailed GPU memory usage tracking
- **Multi-GPU Testing**: Validation across multiple GPU configurations
- **Distributed Testing**: Support for distributed training validation

#### 2. Enhanced Reporting
- **Interactive Dashboards**: Web-based test result visualization
- **Trend Analysis**: Long-term performance and reliability trends
- **Automated Insights**: AI-powered test failure analysis

#### 3. Extended Coverage
- **Real Data Testing**: Integration with actual microscopy datasets
- **Robustness Testing**: Extended PSF mismatch and noise robustness
- **Baseline Integration**: Comprehensive baseline method comparisons

#### 4. Developer Experience
- **IDE Integration**: Test runner plugins for popular IDEs
- **Quick Feedback**: Faster test execution for development workflows
- **Smart Test Selection**: Run only tests affected by code changes

### Extensibility

#### Adding New Test Categories
```python
# Template for new test category
class TestNewComponent(E2EPipelineTestBase):
    """Test new system component."""
    
    def test_component_functionality(self, temp_workspace, minimal_config):
        # Setup test environment
        # Execute component functionality
        # Validate results
        # Check error conditions
```

#### Custom Benchmarks
```python
# Template for new benchmark
def test_custom_benchmark(self):
    """Custom performance benchmark."""
    
    # Setup benchmark configuration
    # Execute benchmark operation
    # Measure performance metrics
    # Compare against baselines
```

## Conclusion

The PKL-Guided Diffusion E2E testing framework provides comprehensive validation of the entire system pipeline. With 20+ test methods across 6 major categories, automated CI/CD integration, and advanced performance profiling, it ensures system reliability and performance throughout development and deployment.

The framework is designed to support the critical ICLR 2025 deadline by providing:

- **Fast Feedback**: Quick validation during development
- **Comprehensive Coverage**: All major system components tested
- **Performance Monitoring**: Continuous performance tracking
- **CI/CD Integration**: Automated testing in GitHub Actions
- **Developer Productivity**: Easy-to-use commands and clear documentation

**Status: Complete and Ready for Production Use** ✅

All components have been implemented, tested, and validated. The framework is now operational and ready to ensure the reliability and performance of the PKL-Guided Diffusion system.
