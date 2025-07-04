# using Pkg
# Pkg.instantiate()

# Load the simulator
include("metal_bandit_simulator.jl")

# Run the demonstration
result = demonstrate_metal_bandit_simulator()

# Or run a custom simulation
env = MetalBernoulliEnvironment(8, 2000, 500)  # 8 arms, 2000 trials, 500 agents
agent = MetalQLearningAgent(8, 500, 2000; alpha=0.1f0, beta=3.0f0)


run_metal_bandit_simulation!(env, agent)

# Estimate parameters with GPU acceleration
estimated_params = gpu_mle_parameter_estimation(env, agent)

# Analyze and plot results
recovery_plot, metrics = plot_parameter_recovery(Array(env.true_params), estimated_params)
