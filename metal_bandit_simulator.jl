using Metal
using Statistics
using BenchmarkTools
using Random
using Plots
using StatsPlots
using Distributions
using LinearAlgebra

# GPU-Accelerated Bernoulli Bandit Simulator with Q-learning and MLE Parameter Estimation

struct MetalBernoulliEnvironment{T}
    n_arms::Int
    n_trials::Int
    n_agents::Int
    true_params::MtlArray{T, 2}  # (n_arms, n_agents)
    rewards::MtlArray{T, 3}      # (n_trials, n_arms, n_agents)
    actions::MtlArray{Int32, 2}  # (n_trials, n_agents)
    actual_rewards::MtlArray{T, 2}  # (n_trials, n_agents)
end

function MetalBernoulliEnvironment(n_arms::Int, n_trials::Int, n_agents::Int; 
                                  true_params::Union{Nothing, Array{T, 2}} = nothing) where T
    if true_params === nothing
        true_params_cpu = rand(Float32, n_arms, n_agents)
    else
        true_params_cpu = Float32.(true_params)
    end
    
    return MetalBernoulliEnvironment{Float32}(
        n_arms, n_trials, n_agents,
        MtlArray(true_params_cpu),
        Metal.zeros(Float32, n_trials, n_arms, n_agents),
        Metal.zeros(Int32, n_trials, n_agents),
        Metal.zeros(Float32, n_trials, n_agents)
    )
end

struct MetalQLearningAgent{T}
    n_arms::Int
    n_agents::Int
    n_trials::Int
    alpha::MtlArray{T, 2}        # (n_arms, n_agents) - learning rates
    beta::MtlArray{T, 2}         # (n_arms, n_agents) - exploration parameters
    q_values::MtlArray{T, 2}     # (n_arms, n_agents) - Q-values
    arm_counts::MtlArray{Int32, 2}  # (n_arms, n_agents) - arm selection counts
    total_rewards::MtlArray{T, 2}   # (n_arms, n_agents) - cumulative rewards per arm
end

function MetalQLearningAgent(n_arms::Int, n_agents::Int, n_trials::Int;
                           alpha::T = 0.1f0, beta::T = 2.0f0) where T
    return MetalQLearningAgent{T}(
        n_arms, n_agents, n_trials,
        Metal.fill(alpha, n_arms, n_agents),
        Metal.fill(beta, n_arms, n_agents),
        Metal.fill(0.5f0, n_arms, n_agents),  # Initialize Q-values to 0.5
        Metal.zeros(Int32, n_arms, n_agents),
        Metal.zeros(T, n_arms, n_agents)
    )
end

# Ultra-optimized Metal kernels for maximum GPU utilization

function softmax_action_selection_kernel!(actions, q_values, beta, random_vals, n_arms, n_agents, trial)
    agent_idx = thread_position_in_grid_1d()
    
    if agent_idx <= n_agents
        # Compute softmax probabilities
        max_q = q_values[1, agent_idx]
        for arm in 2:n_arms
            if q_values[arm, agent_idx] > max_q
                max_q = q_values[arm, agent_idx]
            end
        end
        
        exp_sum = 0.0f0
        for arm in 1:n_arms
            exp_sum += exp(beta[arm, agent_idx] * (q_values[arm, agent_idx] - max_q))
        end
        
        # Sample action using inverse CDF
        random_val = random_vals[agent_idx]
        cumulative_prob = 0.0f0
        selected_arm = 1
        
        for arm in 1:n_arms
            prob = exp(beta[arm, agent_idx] * (q_values[arm, agent_idx] - max_q)) / exp_sum
            cumulative_prob += prob
            if random_val <= cumulative_prob
                selected_arm = arm
                break
            end
        end
        
        actions[trial, agent_idx] = selected_arm
    end
    
    return nothing
end

function reward_generation_kernel!(rewards, actions, true_params, random_vals, n_agents, trial)
    agent_idx = thread_position_in_grid_1d()
    
    if agent_idx <= n_agents
        arm = actions[trial, agent_idx]
        prob = true_params[arm, agent_idx]
        reward = random_vals[agent_idx] < prob ? 1.0f0 : 0.0f0
        rewards[trial, agent_idx] = reward
    end
    
    return nothing
end

function q_learning_update_kernel!(q_values, arm_counts, total_rewards, actions, rewards, alpha, n_agents, trial)
    agent_idx = thread_position_in_grid_1d()
    
    if agent_idx <= n_agents
        arm = actions[trial, agent_idx]
        reward = rewards[trial, agent_idx]
        
        # Update counts and totals
        arm_counts[arm, agent_idx] += 1
        total_rewards[arm, agent_idx] += reward
        
        # Q-learning update
        current_q = q_values[arm, agent_idx]
        learning_rate = alpha[arm, agent_idx]
        
        # Update Q-value
        q_values[arm, agent_idx] = current_q + learning_rate * (reward - current_q)
    end
    
    return nothing
end

# Extremely optimized MLE parameter estimation kernels

function mle_statistics_kernel!(successes, trials, actions, rewards, n_arms, n_agents, n_trials_total)
    idx = thread_position_in_grid_1d()
    total_elements = n_arms * n_agents
    
    if idx <= total_elements
        arm = ((idx - 1) % n_arms) + 1
        agent = ((idx - 1) ÷ n_arms) + 1
        
        success_count = 0
        trial_count = 0
        
        # Count successes and trials for this arm-agent combination
        for trial in 1:n_trials_total
            if actions[trial, agent] == arm
                trial_count += 1
                if rewards[trial, agent] > 0.5f0
                    success_count += 1
                end
            end
        end
        
        successes[arm, agent] = Float32(success_count)
        trials[arm, agent] = Float32(trial_count)
    end
    
    return nothing
end

function mle_estimation_kernel!(estimated_params, successes, trials, n_arms, n_agents)
    idx = thread_position_in_grid_1d()
    total_elements = n_arms * n_agents
    
    if idx <= total_elements
        arm = ((idx - 1) % n_arms) + 1
        agent = ((idx - 1) ÷ n_arms) + 1
        
        trial_count = trials[arm, agent]
        success_count = successes[arm, agent]
        
        # MLE estimate with Laplace smoothing for stability
        estimated_params[arm, agent] = (success_count + 1.0f0) / (trial_count + 2.0f0)
    end
    
    return nothing
end

# Batch processing for maximum GPU efficiency
function batch_mle_estimation_kernel!(estimated_params, actions, rewards, n_arms, n_agents, n_trials_total, batch_size)
    idx = thread_position_in_grid_1d()
    total_elements = n_arms * n_agents
    
    if idx <= total_elements
        arm = ((idx - 1) % n_arms) + 1
        agent = ((idx - 1) ÷ n_arms) + 1
        
        # Process in batches to maximize memory efficiency
        success_count = 0
        trial_count = 0
        
        for batch_start in 1:batch_size:n_trials_total
            batch_end = min(batch_start + batch_size - 1, n_trials_total)
            
            for trial in batch_start:batch_end
                if actions[trial, agent] == arm
                    trial_count += 1
                    if rewards[trial, agent] > 0.5f0
                        success_count += 1
                    end
                end
            end
        end
        
        # MLE estimate with Laplace smoothing
        estimated_params[arm, agent] = (Float32(success_count) + 1.0f0) / (Float32(trial_count) + 2.0f0)
    end
    
    return nothing
end

# Main simulation function with maximum GPU utilization
function run_metal_bandit_simulation!(env::MetalBernoulliEnvironment{T}, 
                                     agent::MetalQLearningAgent{T};
                                     batch_size::Int = 1000) where T
    n_trials = env.n_trials
    n_agents = env.n_agents
    n_arms = env.n_arms
    
    # Pre-allocate random number arrays for maximum efficiency
    random_action_vals = Metal.rand(Float32, n_agents)
    random_reward_vals = Metal.rand(Float32, n_agents)
    
    # Run simulation in batches for memory efficiency
    for batch_start in 1:batch_size:n_trials
        batch_end = min(batch_start + batch_size - 1, n_trials)
        
        for trial in batch_start:batch_end
            # Generate new random values for this trial
            random_action_vals = Metal.rand(Float32, n_agents)
            random_reward_vals = Metal.rand(Float32, n_agents)
            
            # Action selection using softmax
            action_kernel = Metal.@metal launch=false softmax_action_selection_kernel!(
                env.actions, agent.q_values, agent.beta, random_action_vals, 
                n_arms, n_agents, trial
            )
            action_kernel(env.actions, agent.q_values, agent.beta, random_action_vals, 
                         n_arms, n_agents, trial; 
                         threads=min(1024, n_agents), groups=cld(n_agents, 1024))
            
            # Reward generation
            reward_kernel = Metal.@metal launch=false reward_generation_kernel!(
                env.actual_rewards, env.actions, env.true_params, 
                random_reward_vals, n_agents, trial
            )
            reward_kernel(env.actual_rewards, env.actions, env.true_params, 
                         random_reward_vals, n_agents, trial;
                         threads=min(1024, n_agents), groups=cld(n_agents, 1024))
            
            # Q-learning update
            update_kernel = Metal.@metal launch=false q_learning_update_kernel!(
                agent.q_values, agent.arm_counts, agent.total_rewards,
                env.actions, env.actual_rewards, agent.alpha, n_agents, trial
            )
            update_kernel(agent.q_values, agent.arm_counts, agent.total_rewards,
                         env.actions, env.actual_rewards, agent.alpha, n_agents, trial;
                         threads=min(1024, n_agents), groups=cld(n_agents, 1024))
        end
    end
    
    return nothing
end

# Ultra-fast MLE parameter estimation with maximum GPU parallelization
function gpu_mle_parameter_estimation(env::MetalBernoulliEnvironment{T}, 
                                     agent::MetalQLearningAgent{T};
                                     use_batch_processing::Bool = true,
                                     batch_size::Int = 1000) where T
    n_arms = env.n_arms
    n_agents = env.n_agents
    n_trials = env.n_trials
    
    estimated_params = Metal.zeros(T, n_arms, n_agents)
    
    if use_batch_processing
        # Use batch processing for maximum memory efficiency
        mle_kernel = Metal.@metal launch=false batch_mle_estimation_kernel!(
            estimated_params, env.actions, env.actual_rewards, 
            n_arms, n_agents, n_trials, batch_size
        )
        
        total_elements = n_arms * n_agents
        mle_kernel(estimated_params, env.actions, env.actual_rewards, 
                  n_arms, n_agents, n_trials, batch_size;
                  threads=min(1024, total_elements), groups=cld(total_elements, 1024))
    else
        # Two-stage approach for comparison
        successes = Metal.zeros(T, n_arms, n_agents)
        trials = Metal.zeros(T, n_arms, n_agents)
        
        # Stage 1: Count statistics
        stats_kernel = Metal.@metal launch=false mle_statistics_kernel!(
            successes, trials, env.actions, env.actual_rewards, 
            n_arms, n_agents, n_trials
        )
        
        total_elements = n_arms * n_agents
        stats_kernel(successes, trials, env.actions, env.actual_rewards, 
                    n_arms, n_agents, n_trials;
                    threads=min(1024, total_elements), groups=cld(total_elements, 1024))
        
        # Stage 2: Compute MLE estimates
        mle_kernel = Metal.@metal launch=false mle_estimation_kernel!(
            estimated_params, successes, trials, n_arms, n_agents
        )
        
        mle_kernel(estimated_params, successes, trials, n_arms, n_agents;
                  threads=min(1024, total_elements), groups=cld(total_elements, 1024))
    end
    
    return Array(estimated_params)
end

# Parameter recovery analysis and plotting
function analyze_parameter_recovery(true_params::Array{T1, 2}, 
                                  estimated_params::Array{T2, 2}) where {T1, T2}
    n_arms, n_agents = size(true_params)
    
    # Compute recovery metrics
    absolute_errors = abs.(estimated_params - true_params)
    relative_errors = absolute_errors ./ (true_params .+ 1e-8)
    
    recovery_metrics = (
        mae = mean(absolute_errors),
        mse = mean(absolute_errors.^2),
        rmse = sqrt(mean(absolute_errors.^2)),
        mape = mean(relative_errors) * 100,
        correlation = cor(vec(true_params), vec(estimated_params)),
        r_squared = cor(vec(true_params), vec(estimated_params))^2
    )
    
    return recovery_metrics, absolute_errors, relative_errors
end

function plot_parameter_recovery(true_params::Array{T1, 2}, 
                               estimated_params::Array{T2, 2};
                               title_prefix::String = "Parameter Recovery") where {T1, T2}
    n_arms, n_agents = size(true_params)
    
    # Recovery metrics
    recovery_metrics, absolute_errors, relative_errors = analyze_parameter_recovery(true_params, estimated_params)
    
    # Create comprehensive plots
    p1 = scatter(vec(true_params), vec(estimated_params), 
                alpha=0.6, ms=3,
                xlabel="True Parameters", ylabel="Estimated Parameters",
                title="$title_prefix: True vs Estimated",
                legend=false)
    plot!(p1, [0, 1], [0, 1], line=:dash, color=:red, linewidth=2)
    
    p2 = histogram(vec(absolute_errors), bins=50, alpha=0.7,
                  xlabel="Absolute Error", ylabel="Frequency",
                  title="Distribution of Absolute Errors")
    
    p3 = heatmap(absolute_errors, 
                xlabel="Agent", ylabel="Arm", 
                title="Absolute Error Heatmap",
                color=:viridis)
    
    p4 = boxplot(reshape(absolute_errors, :, 1), 
                ylabel="Absolute Error",
                title="Error Distribution Summary")
    
    # Recovery metrics text
    metrics_text = """
    MAE: $(round(recovery_metrics.mae, digits=4))
    RMSE: $(round(recovery_metrics.rmse, digits=4))
    MAPE: $(round(recovery_metrics.mape, digits=2))%
    Correlation: $(round(recovery_metrics.correlation, digits=4))
    R²: $(round(recovery_metrics.r_squared, digits=4))
    """
    
    p5 = plot([], [], showaxis=false, grid=false, legend=false,
             title="Recovery Metrics")
    annotate!(p5, 0.5, 0.5, text(metrics_text, 12, :center))
    
    # Arm-wise recovery analysis
    arm_errors = [mean(absolute_errors[arm, :]) for arm in 1:n_arms]
    p6 = bar(1:n_arms, arm_errors, 
            xlabel="Arm", ylabel="Mean Absolute Error",
            title="Error by Arm", legend=false)
    
    final_plot = plot(p1, p2, p3, p4, p5, p6, layout=(2, 3), size=(1200, 800))
    
    return final_plot, recovery_metrics
end

# Comprehensive benchmark function
function benchmark_metal_bandit_simulator(n_arms_list = [5, 10, 20], 
                                        n_agents_list = [100, 500, 1000],
                                        n_trials_list = [1000, 5000];
                                        alpha::Float32 = 0.1f0,
                                        beta::Float32 = 2.0f0)
    
    results = Dict()
    
    for n_arms in n_arms_list
        for n_agents in n_agents_list
            for n_trials in n_trials_list
                println("Testing: $n_arms arms, $n_agents agents, $n_trials trials")
                
                # Create environment and agent
                env = MetalBernoulliEnvironment(n_arms, n_trials, n_agents)
                agent = MetalQLearningAgent(n_arms, n_agents, n_trials; alpha=alpha, beta=beta)
                
                # Run simulation
                sim_time = @elapsed run_metal_bandit_simulation!(env, agent)
                
                # MLE parameter estimation (the heavy computation)
                mle_time = @elapsed estimated_params = gpu_mle_parameter_estimation(env, agent)
                
                # Analyze recovery
                true_params_cpu = Array(env.true_params)
                recovery_metrics, _, _ = analyze_parameter_recovery(true_params_cpu, estimated_params)
                
                # Store results
                key = (n_arms, n_agents, n_trials)
                results[key] = (
                    simulation_time = sim_time,
                    mle_time = mle_time,
                    total_time = sim_time + mle_time,
                    recovery_metrics = recovery_metrics,
                    throughput = n_arms * n_agents * n_trials / (sim_time + mle_time)
                )
                
                println("  Simulation: $(round(sim_time, digits=3))s")
                println("  MLE: $(round(mle_time, digits=3))s")
                println("  Recovery R²: $(round(recovery_metrics.r_squared, digits=4))")
                println("  Throughput: $(round(results[key].throughput, digits=0)) ops/s")
                println()
            end
        end
    end
    
    return results
end

# Main demonstration function
function demonstrate_metal_bandit_simulator()
    println("🚀 GPU-Accelerated Bernoulli Bandit Simulator with Metal.jl")
    println("=" ^ 60)
    
    if !Metal.functional()
        println("❌ Metal not available. This demo requires Apple Silicon.")
        return nothing
    end
    
    println("✅ Metal available")
    try
        println("Memory: $(Metal.max_buffer_length() ÷ 1024^3) GB")
    catch
        println("Memory: Info not available")
    end
    println()
    
    # Demo parameters
    n_arms = 8
    n_agents = 500
    n_trials = 2000
    alpha = 0.1f0
    beta = 3.0f0
    
    println("Demo Configuration:")
    println("  Arms: $n_arms")
    println("  Agents: $n_agents") 
    println("  Trials: $n_trials")
    println("  Learning rate (α): $alpha")
    println("  Exploration (β): $beta")
    println()
    
    # Create environment with known parameters
    true_params_demo = rand(Float32, n_arms, n_agents) .* 0.8 .+ 0.1  # Between 0.1 and 0.9
    env = MetalBernoulliEnvironment(n_arms, n_trials, n_agents; true_params=true_params_demo)
    agent = MetalQLearningAgent(n_arms, n_agents, n_trials; alpha=alpha, beta=beta)
    
    println("🎯 Running simulation...")
    sim_time = @elapsed run_metal_bandit_simulation!(env, agent)
    println("  Simulation completed in $(round(sim_time, digits=3))s")
    
    println("🧠 Performing MLE parameter estimation...")
    mle_time = @elapsed estimated_params = gpu_mle_parameter_estimation(env, agent)
    println("  MLE completed in $(round(mle_time, digits=3))s")
    
    # Analyze and plot results
    println("📊 Analyzing parameter recovery...")
    recovery_plot, recovery_metrics = plot_parameter_recovery(true_params_demo, estimated_params)
    
    println("\n📈 Recovery Results:")
    println("  Mean Absolute Error: $(round(recovery_metrics.mae, digits=4))")
    println("  Root Mean Square Error: $(round(recovery_metrics.rmse, digits=4))")
    println("  Mean Absolute Percentage Error: $(round(recovery_metrics.mape, digits=2))%")
    println("  Correlation: $(round(recovery_metrics.correlation, digits=4))")
    println("  R-squared: $(round(recovery_metrics.r_squared, digits=4))")
    
    throughput = n_arms * n_agents * n_trials / (sim_time + mle_time)
    println("\n⚡ Performance:")
    println("  Total time: $(round(sim_time + mle_time, digits=3))s")
    println("  Throughput: $(round(throughput, digits=0)) operations/second")
    
    display(recovery_plot)
    
    return env, agent, estimated_params, recovery_metrics, recovery_plot
end

# Export main functions
export MetalBernoulliEnvironment, MetalQLearningAgent
export run_metal_bandit_simulation!, gpu_mle_parameter_estimation
export plot_parameter_recovery, analyze_parameter_recovery
export benchmark_metal_bandit_simulator, demonstrate_metal_bandit_simulator