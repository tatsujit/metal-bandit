using Metal
using Statistics
using Random
using LinearAlgebra
using StatsBase

"""
Focused GPU Optimizations for Q-Learning Parameter Recovery

Key optimizations:
1. GPU-accelerated data generation
2. Vectorized operations
3. Reduced memory transfers
4. Simplified parameter estimation
"""

# GPU kernel for efficient Q-learning simulation
function gpu_simulate_qlearning_kernel!(actions, rewards, q_values, alpha_vec, beta_vec, 
                                       reward_probs, random_vals, n_arms, n_trials, n_subjects)
    subject_idx = thread_position_in_grid_1d()
    
    if subject_idx <= n_subjects
        α = alpha_vec[subject_idx]
        β = beta_vec[subject_idx]
        
        # Initialize Q-values for this subject
        for arm in 1:n_arms
            q_values[arm, subject_idx] = 0.5f0
        end
        
        # Run Q-learning simulation
        for trial in 1:n_trials
            # Compute softmax probabilities
            max_q = q_values[1, subject_idx]
            for arm in 2:n_arms
                if q_values[arm, subject_idx] > max_q
                    max_q = q_values[arm, subject_idx]
                end
            end
            
            # Calculate softmax denominator
            exp_sum = 0.0f0
            for arm in 1:n_arms
                exp_sum += exp(β * (q_values[arm, subject_idx] - max_q))
            end
            
            # Sample action using inverse CDF
            rand_val = random_vals[trial, subject_idx]
            cumulative_prob = 0.0f0
            action = 1
            
            for arm in 1:n_arms
                prob = exp(β * (q_values[arm, subject_idx] - max_q)) / exp_sum
                cumulative_prob += prob
                if rand_val <= cumulative_prob
                    action = arm
                    break
                end
            end
            
            actions[trial, subject_idx] = action
            
            # Generate reward
            reward_rand = random_vals[trial + n_trials, subject_idx]
            reward = reward_rand < reward_probs[action, subject_idx] ? 1.0f0 : 0.0f0
            rewards[trial, subject_idx] = reward
            
            # Update Q-value
            prediction_error = reward - q_values[action, subject_idx]
            q_values[action, subject_idx] += α * prediction_error
        end
    end
    
    return nothing
end

# Optimized GPU-accelerated parameter recovery
function gpu_accelerated_recovery(n_subjects::Int = 1000, n_arms::Int = 4, 
                                n_trials::Int = 200; seed::Int = 42)
    Random.seed!(seed)
    
    println("🚀 GPU加速 Q学習パラメータ回復実験")
    println("被験者数: $n_subjects, アーム数: $n_arms, 試行数: $n_trials")
    
    # Generate true parameters
    true_alpha = rand(Float32, n_subjects)
    true_beta = rand(Float32, n_subjects) * 10.0f0
    true_reward_probs = rand(Float32, n_arms, n_subjects) * 0.6f0 .+ 0.2f0
    
    # Allocate GPU memory
    d_actions = MtlArray(zeros(Int32, n_trials, n_subjects))
    d_rewards = MtlArray(zeros(Float32, n_trials, n_subjects))
    d_q_values = MtlArray(zeros(Float32, n_arms, n_subjects))
    d_alpha_vec = MtlArray(true_alpha)
    d_beta_vec = MtlArray(true_beta)
    d_reward_probs = MtlArray(true_reward_probs)
    d_random_vals = MtlArray(rand(Float32, n_trials * 2, n_subjects))
    
    # GPU simulation
    println("GPU データ生成実行中...")
    gpu_time = @elapsed begin
        @metal threads=min(1024, n_subjects) gpu_simulate_qlearning_kernel!(
            d_actions, d_rewards, d_q_values, d_alpha_vec, d_beta_vec,
            d_reward_probs, d_random_vals, n_arms, n_trials, n_subjects
        )
        Metal.synchronize()
    end
    
    # Transfer data back to CPU for parameter estimation
    println("パラメータ推定実行中...")
    actions_cpu = Array(d_actions)
    rewards_cpu = Array(d_rewards)
    
    # Fast parameter estimation using grid search
    estimated_alpha = zeros(Float32, n_subjects)
    estimated_beta = zeros(Float32, n_subjects)
    
    estimation_time = @elapsed begin
        Threads.@threads for subject in 1:n_subjects
            best_nll = Inf
            best_alpha = 0.5f0
            best_beta = 5.0f0
            
            # Coarse grid search for speed
            alpha_grid = [0.1f0, 0.3f0, 0.5f0, 0.7f0, 0.9f0]
            beta_grid = [1.0f0, 3.0f0, 5.0f0, 7.0f0, 10.0f0]
            
            for α in alpha_grid
                for β in beta_grid
                    nll = compute_negative_log_likelihood(
                        actions_cpu[:, subject], rewards_cpu[:, subject], α, β, n_arms
                    )
                    if nll < best_nll
                        best_nll = nll
                        best_alpha = α
                        best_beta = β
                    end
                end
            end
            
            estimated_alpha[subject] = best_alpha
            estimated_beta[subject] = best_beta
        end
    end
    
    total_time = gpu_time + estimation_time
    
    # Calculate recovery statistics
    α_correlation = cor(true_alpha, estimated_alpha)
    β_correlation = cor(true_beta, estimated_beta)
    
    println("✅ GPU加速実験完了！")
    println("  GPU生成時間: $(round(gpu_time, digits=3))s")
    println("  推定時間: $(round(estimation_time, digits=3))s")
    println("  総時間: $(round(total_time, digits=3))s")
    println("  α回復相関: $(round(α_correlation, digits=3))")
    println("  β回復相関: $(round(β_correlation, digits=3))")
    
    return (
        true_alpha = true_alpha,
        true_beta = true_beta,
        estimated_alpha = estimated_alpha,
        estimated_beta = estimated_beta,
        gpu_time = gpu_time,
        estimation_time = estimation_time,
        total_time = total_time,
        alpha_correlation = α_correlation,
        beta_correlation = β_correlation
    )
end

# Fast negative log-likelihood computation
function compute_negative_log_likelihood(actions::Vector{Int32}, rewards::Vector{Float32}, 
                                       α::Float32, β::Float32, n_arms::Int)
    n_trials = length(actions)
    q_values = fill(0.5f0, n_arms)
    nll = 0.0
    
    for trial in 1:n_trials
        action = actions[trial]
        reward = rewards[trial]
        
        # Compute action probability using softmax
        max_q = maximum(q_values)
        exp_sum = sum(exp(β * (q - max_q)) for q in q_values)
        action_prob = exp(β * (q_values[action] - max_q)) / exp_sum
        
        # Add to negative log-likelihood
        nll -= log(max(action_prob, 1e-10))
        
        # Q-learning update
        prediction_error = reward - q_values[action]
        q_values[action] += α * prediction_error
    end
    
    return nll
end

# Performance comparison function
function test_gpu_acceleration()
    println("🔥 GPU加速性能テスト")
    println("=" ^ 50)
    
    # Test parameters
    test_subjects = 1000
    test_arms = 4
    test_trials = 200
    
    # Run GPU-accelerated version
    println("GPU加速版実行中...")
    gpu_result = gpu_accelerated_recovery(test_subjects, test_arms, test_trials)
    
    # Compare with original if available
    try
        include("q_parameter_recovery.jl")
        println("\\nオリジナル版比較実行中...")
        original_time = @elapsed begin
            original_result = run_parameter_recovery_experiment(test_subjects, test_arms, test_trials)
        end
        
        speedup = original_time / gpu_result.total_time
        
        println("\\n🏆 性能比較結果")
        println("=" ^ 40)
        println("オリジナル: $(round(original_time, digits=2))s")
        println("GPU加速:    $(round(gpu_result.total_time, digits=2))s")
        println("高速化:     $(round(speedup, digits=2))x")
        
        if speedup > 1.5
            println("🚀 GPU加速成功！")
        else
            println("⚡ さらなる最適化が必要")
        end
        
        return (gpu_result = gpu_result, original_time = original_time, speedup = speedup)
        
    catch e
        println("オリジナル版との比較をスキップ: $e")
        return (gpu_result = gpu_result, original_time = nothing, speedup = nothing)
    end
end

export gpu_accelerated_recovery, test_gpu_acceleration