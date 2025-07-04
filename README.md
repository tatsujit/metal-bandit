# MetalBandit: GPU-Accelerated Q-Learning Parameter Recovery

A comprehensive Julia implementation for recovering Q-learning model parameters (α and β) from behavioral data in multi-armed bandit environments, using Apple Silicon GPU acceleration via Metal.jl.

## Overview

This project implements **Q-learning parameter recovery** - the process of estimating the learning rate (α) and inverse temperature (β) parameters of Q-learning models from observed choice behavior. This is a fundamental problem in computational cognitive science and reinforcement learning research.

### What We Recover

- **α (Learning Rate)**: How quickly agents update Q-values from prediction errors (0 ≤ α ≤ 1)
- **β (Inverse Temperature)**: How deterministic vs exploratory the agent's action selection is (β ≥ 0)

### What We Don't Recover

- ❌ Reward probabilities of bandit arms (these are known environmental parameters)
- ❌ Q-values themselves (these are internal agent states)
- ❌ Action policies (these emerge from the Q-learning + softmax process)

## Key Features

- **GPU-Accelerated Parameter Recovery**: Fast Maximum Likelihood Estimation using Metal.jl
- **Multi-Scale Performance Analysis**: Comprehensive GPU vs CPU comparison across dataset sizes
- **Scientific Validation**: 100% estimation success with identical parameter recovery quality
- **Realistic Q-Learning Simulation**: Standard temporal difference learning with softmax action selection
- **Comprehensive Analysis**: Parameter recovery plots, correlation analysis, and performance benchmarking

## Architecture

### Core Components

1. **Q-Learning Simulation**: Generate behavioral data from known α and β parameters
2. **Parameter Recovery**: Use MLE to estimate α and β from choice and reward sequences
3. **GPU Acceleration**: Metal kernels for parallel computation across multiple subjects
4. **Performance Comparison**: Systematic GPU vs CPU analysis across scales

### Q-Learning Model

```
Q(a) ← Q(a) + α × [reward - Q(a)]           # Q-value update
P(a) = exp(β × Q(a)) / Σ exp(β × Q(a'))     # Softmax action selection
```

### Parameter Recovery Process

1. **Data Generation**: Simulate Q-learning behavior with known α, β
2. **MLE Optimization**: Find α̂, β̂ that maximize likelihood of observed choices
3. **Validation**: Compare recovered parameters to ground truth
4. **Analysis**: Assess recovery quality via correlation, bias, and error metrics

## Requirements

- Julia 1.9+
- Apple Silicon Mac (M1/M2/M3/M4 series)
- Required packages: Metal.jl, Optim.jl, CairoMakie.jl, CSV.jl, DataFrames.jl

## Quick Start

```julia
# Install dependencies
using Pkg; Pkg.instantiate()

# Run Q-learning parameter recovery experiment
include("q_parameter_recovery.jl")
experiment, fig, df = main_parameter_recovery_experiment()

# Results: 1000 subjects with random α ∈ [0,1], β ∈ [0,10]
# Output: Parameter recovery visualization and CSV results
```

## Performance Results

### Multi-Scale GPU vs CPU Analysis

| Scale | Dataset | GPU Time | CPU Time | CPU Advantage |
|-------|---------|----------|----------|---------------|
| Small | 200×4×200 | 6.5s | 3.1s | **2.08x faster** |
| Large | 2000×8×500 | 73.8s | 65.8s | **1.12x faster** |
| Ultra | 5000×8×1000 | 257.3s | 238.7s | **1.08x faster** |
| Extreme | 10000×8×500 | 264.7s | 253.1s | **1.05x faster** |

### Key Findings

- **CPU dominates** at small-medium scales (≤10K subjects)
- **GPU gap closing rapidly**: CPU advantage shrinks from 2.08x to 1.05x
- **Crossover point**: GPU likely faster at 15K+ subjects
- **Perfect scientific validity**: 100% estimation success, identical parameter recovery

## Parameter Recovery Quality

- **Learning Rate α**: Excellent recovery (r ≈ 0.93, R² ≈ 0.87)
- **Inverse Temperature β**: Moderate recovery (r ≈ 0.70, R² ≈ 0.49)
- **Success Rate**: 100% across all scales and methods
- **Bias**: Minimal systematic estimation bias

This recovery pattern is consistent with computational cognitive science literature - learning rates are easier to estimate than exploration parameters from behavioral data.

## Usage Examples

### Basic Parameter Recovery

```julia
# Generate Q-learning behavior with known parameters
true_alpha, true_beta = 0.3, 5.0
reward_probs = [0.8, 0.6, 0.4, 0.2]  # Environment rewards
actions, rewards = generate_q_learning_behavior(true_alpha, true_beta, reward_probs, 200)

# Recover parameters from behavioral data
estimated_params, log_likelihood, success = estimate_q_parameters(actions, rewards, 4)
estimated_alpha, estimated_beta = estimated_params

println("True: α=$true_alpha, β=$true_beta")
println("Estimated: α=$estimated_alpha, β=$estimated_beta")
```

### GPU vs CPU Comparison

```julia
# Small scale comparison
include("gpu_vs_cpu_comparison.jl")
comparison, timing_results = run_fair_gpu_vs_cpu_comparison(200, 4, 200)

# Large scale comparison  
include("large_scale_gpu_vs_cpu_comparison.jl")
comparison, timing_results = run_large_scale_comparison(n_subjects=5000, n_arms=8, n_trials=1000)
```

### Parameter Recovery Analysis

```julia
# Run 1000-subject parameter recovery experiment
experiment = run_parameter_recovery_experiment(1000, 4, 200)

# Analyze results
stats = analyze_recovery_results(experiment)
fig = create_recovery_visualization(experiment)

# Save results
df = save_results_to_csv(experiment, "parameter_recovery_results.csv")
```

## Technical Implementation

### Q-Learning Algorithm

1. **Initialization**: Q(a) = 0.5 for all actions
2. **Action Selection**: Choose action via softmax with temperature β
3. **Reward Observation**: Receive binary reward from Bernoulli distribution
4. **Q-Value Update**: Update chosen action's Q-value using learning rate α
5. **Repeat**: Continue for specified number of trials

### MLE Parameter Estimation

1. **Likelihood Function**: P(choices | α, β) based on Q-learning + softmax model
2. **Optimization**: Multi-start BFGS with 10 random initializations
3. **Parameter Bounds**: α ∈ (0, 1), β ∈ (0, 20)
4. **Convergence**: Multiple restarts ensure global optimum

### GPU Acceleration

- **Parallel Subjects**: Each GPU thread handles one subject's parameter estimation
- **Vectorized Operations**: Efficient softmax and Q-value computations
- **Memory Optimization**: Coalesced access patterns for Apple Silicon
- **Batch Processing**: Process multiple subjects simultaneously

## Practical Recommendations

### When to Use GPU vs CPU

- **Small datasets (≤1K subjects)**: Use CPU, GPU overhead not justified
- **Medium datasets (1K-10K subjects)**: Use CPU with 8+ threads
- **Large datasets (10K+ subjects)**: GPU becomes competitive
- **Very large datasets (15K+ subjects)**: GPU likely faster

### Threading Considerations

- **Always use multiple threads**: Single-threaded CPU is 3-4x slower than GPU
- **Optimal thread count**: 8 threads provide good performance on most systems
- **Memory constraints**: Large datasets may exceed GPU memory limits

## Scientific Applications

### Cognitive Modeling

- **Individual Differences**: Estimate learning rates across participants
- **Developmental Studies**: Track parameter changes over time
- **Clinical Applications**: Compare parameters between groups
- **Experimental Design**: Power analysis for detecting parameter differences

### Reinforcement Learning Research

- **Algorithm Comparison**: Compare Q-learning variants
- **Hyperparameter Optimization**: Find optimal α and β values
- **Model Validation**: Verify theoretical predictions with behavioral data
- **Simulation Studies**: Generate synthetic datasets for method development

## Testing

```bash
# Run comprehensive test suite (138 tests)
julia --project=. -e 'using Pkg; Pkg.test()'

# Run specific test categories
julia --project=. test/test_environment.jl    # Environment tests
julia --project=. test/test_agent.jl         # Agent tests  
julia --project=. test/test_kernels.jl       # GPU kernel tests
julia --project=. test/test_integration.jl   # End-to-end tests
julia --project=. test/test_performance.jl   # Performance tests
```

## Reproducibility

All experiments use fixed random seeds and identical algorithms across GPU/CPU implementations to ensure reproducible results. The repository includes:

- Complete source code with documentation
- Experimental data and analysis scripts
- Performance benchmarking results
- Comprehensive test coverage
- Visualization and analysis tools

## Contributing

This project focuses on defensive security research and educational applications in computational cognitive science. Contributions should maintain scientific rigor and avoid any potentially malicious applications.

## Citation

If you use this code in your research, please cite:

```
MetalBandit: GPU-Accelerated Q-Learning Parameter Recovery
A comprehensive implementation for multi-scale Q-learning parameter estimation
with Apple Silicon Metal.jl acceleration (2025)
```

## License

This project is open source and available for research and educational purposes.