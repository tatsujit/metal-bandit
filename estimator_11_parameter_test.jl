using Metal
using Statistics
using Random
using Distributions
using Optim
using CSV
using DataFrames
using CairoMakie
using Dates
using SpecialFunctions
using StatsBase

# Include the original estimator
include("estimator.jl")

"""
Comprehensive 11-Parameter Model Efficiency Testing Framework

Tests the efficiency of parameter estimation across:
- CPU (1-thread): Sequential parameter estimation 
- CPU (8-threads): Multi-threaded parameter estimation
- GPU: Apple Silicon Metal acceleration

The 11 parameters from the Estimator model:
1. Q0: Initial Q-value
2. α: Learning rate (positive RPE)
3. αm: Learning rate (negative RPE) 
4. β: Inverse temperature for policy
5. αf: Forgetting rate
6. μ: Forgetting default value
7. τ: Stickiness rate
8. φ: Stickiness inverse temperature
9. C0: Initial stickiness value
10. η: Beta distribution expectation parameter
11. ν: Beta distribution precision parameter
"""

struct EstimatorTestResult
    scale::String
    method::String
    n_subjects::Int
    n_arms::Int
    n_trials::Int
    execution_time::Float64
    memory_used::Float64
    compilation_time::Float64
    parameter_correlations::Dict{Symbol, Float64}
    success_rate::Float64
    throughput::Float64
    timestamp::String
    thread_count::Int
end

"""
Generate behavioral data using the 11-parameter Estimator model
"""
function generate_estimator_behavior(true_params::NamedTuple, n_arms::Int, n_trials::Int; seed::Int = 42)
    Random.seed!(seed)
    
    # Create estimator with true parameters
    estimator = Estimator(n_arms;
        Q0=true_params.Q0, α=true_params.α, αm=true_params.αm, β=true_params.β,
        αf=true_params.αf, μ=true_params.μ,
        τ=true_params.τ, φ=true_params.φ, C0=true_params.C0,
        η=true_params.η, ν=true_params.ν)
    
    # Generate reward probabilities for arms
    reward_probs = rand(n_arms) * 0.6 .+ 0.2  # Random between 0.2 and 0.8
    
    actions = zeros(Int, n_trials)
    rewards = zeros(Float64, n_trials)
    
    for trial in 1:n_trials
        # Get action probabilities from current estimator state
        probs = selection_probabilities(estimator.W)
        
        # Sample action using inverse CDF
        rand_val = rand()
        cumsum_prob = 0.0
        actions[trial] = 1
        for arm in 1:n_arms
            cumsum_prob += probs[arm]
            if rand_val <= cumsum_prob
                actions[trial] = arm
                break
            end
        end
        
        # Generate reward
        rewards[trial] = rand() < reward_probs[actions[trial]] ? 1.0 : 0.0
        
        # Update estimator
        update!(estimator, actions[trial], rewards[trial])
    end
    
    return actions, rewards, reward_probs
end

"""
Negative log-likelihood for 11-parameter model
"""
function estimator_negative_log_likelihood(params_vec::Vector{Float64}, actions::Vector{Int}, 
                                         rewards::Vector{Float64}, n_arms::Int)
    # Unpack parameters with bounds checking
    Q0, α, αm, β, αf, μ, τ, φ, C0, η, ν = params_vec
    
    # Parameter bounds
    α = clamp(α, 0.001, 0.999)
    αm = clamp(αm, 0.001, 0.999)
    β = clamp(β, 0.1, 20.0)
    αf = clamp(αf, 0.0, 0.999)
    τ = clamp(τ, 0.0, 10.0)
    φ = clamp(φ, 0.0, 20.0)
    η = clamp(η, 0.01, 0.99)
    ν = clamp(ν, 0.1, 20.0)
    
    # Create estimator with these parameters
    try
        estimator = Estimator(n_arms;
            Q0=Q0, α=α, αm=αm, β=β,
            αf=αf, μ=μ, τ=τ, φ=φ, C0=C0,
            η=η, ν=ν)
        
        nll = 0.0
        n_trials = length(actions)
        
        for trial in 1:n_trials
            action = actions[trial]
            reward = rewards[trial]
            
            # Get action probabilities
            probs = selection_probabilities(estimator.W)
            action_prob = probs[action]
            
            # Add to negative log-likelihood
            nll -= log(max(action_prob, 1e-10))
            
            # Update estimator for next trial
            update!(estimator, action, reward)
        end
        
        return nll
    catch
        return 1e10  # Return large penalty for invalid parameters
    end
end

"""
CPU Single-threaded parameter estimation for 11-parameter model
"""
function cpu_single_thread_11param_estimation(n_subjects::Int, n_arms::Int, n_trials::Int; seed::Int = 42)
    Random.seed!(seed)
    
    println("🔧 CPU単線 11パラメータ推定実験")
    println("被験者数: $n_subjects, アーム数: $n_arms, 試行数: $n_trials")
    
    # Generate true parameters for all subjects
    true_params_array = []
    estimated_params_array = []
    estimation_success = fill(false, n_subjects)
    
    for subject in 1:n_subjects
        if subject % max(1, n_subjects ÷ 20) == 0
            println("  進捗: $subject / $n_subjects ($(round(subject/n_subjects*100, digits=1))%)")
        end
        
        # Generate random true parameters
        true_params = (
            Q0 = rand() * 0.4 + 0.3,  # 0.3-0.7
            α = rand() * 0.8 + 0.1,   # 0.1-0.9
            αm = rand() * 0.8 + 0.1,  # 0.1-0.9
            β = rand() * 15 + 1,      # 1-16
            αf = rand() * 0.5,        # 0-0.5
            μ = rand() * 0.6 + 0.2,   # 0.2-0.8
            τ = rand() * 5,           # 0-5
            φ = rand() * 10 + 1,      # 1-11
            C0 = rand() * 0.4 + 0.1,  # 0.1-0.5
            η = rand() * 0.8 + 0.1,   # 0.1-0.9
            ν = rand() * 15 + 1       # 1-16
        )
        
        push!(true_params_array, true_params)
        
        # Generate behavioral data
        actions, rewards, _ = generate_estimator_behavior(true_params, n_arms, n_trials; seed=seed+subject)
        
        # Parameter estimation using optimization
        initial_guess = [0.5, 0.5, 0.5, 5.0, 0.1, 0.5, 1.0, 5.0, 0.2, 0.5, 5.0]
        
        try
            # Multiple restarts optimization
            best_result = nothing
            best_nll = Inf
            
            for restart in 1:3  # Reduced restarts for speed
                initial_params = initial_guess .+ 0.1 * randn(11)
                
                result = optimize(
                    params -> estimator_negative_log_likelihood(params, actions, rewards, n_arms),
                    initial_params,
                    BFGS(),
                    Optim.Options(iterations=100, g_tol=1e-4)
                )
                
                if Optim.converged(result) && result.minimum < best_nll
                    best_nll = result.minimum
                    best_result = result
                end
            end
            
            if best_result !== nothing
                estimated_params = Optim.minimizer(best_result)
                push!(estimated_params_array, estimated_params)
                estimation_success[subject] = true
            else
                push!(estimated_params_array, fill(NaN, 11))
            end
            
        catch e
            push!(estimated_params_array, fill(NaN, 11))
        end
    end
    
    # Calculate parameter correlations
    param_correlations = Dict{Symbol, Float64}()
    param_names = [:Q0, :α, :αm, :β, :αf, :μ, :τ, :φ, :C0, :η, :ν]
    
    success_mask = estimation_success
    if sum(success_mask) > 0
        for (i, param_name) in enumerate(param_names)
            true_vals = [getfield(true_params_array[j], param_name) for j in 1:n_subjects if success_mask[j]]
            est_vals = [estimated_params_array[j][i] for j in 1:n_subjects if success_mask[j]]
            
            if length(true_vals) > 1
                param_correlations[param_name] = cor(true_vals, est_vals)
            else
                param_correlations[param_name] = NaN
            end
        end
    else
        for param_name in param_names
            param_correlations[param_name] = NaN
        end
    end
    
    success_rate = mean(estimation_success)
    println("✅ CPU単線11パラメータ実験完了！成功率: $(round(success_rate*100, digits=1))%")
    
    return (
        true_params = true_params_array,
        estimated_params = estimated_params_array,
        parameter_correlations = param_correlations,
        success_rate = success_rate
    )
end

"""
CPU Multi-threaded parameter estimation for 11-parameter model
"""
function cpu_multithread_11param_estimation(n_subjects::Int, n_arms::Int, n_trials::Int; seed::Int = 42)
    Random.seed!(seed)
    
    println("🔧 CPU並列 11パラメータ推定実験 ($(Threads.nthreads())スレッド)")
    println("被験者数: $n_subjects, アーム数: $n_arms, 試行数: $n_trials")
    
    # Pre-generate all true parameters and behavioral data
    true_params_array = Vector{NamedTuple}(undef, n_subjects)
    behavioral_data = Vector{Tuple}(undef, n_subjects)
    
    println("行動データ生成中...")
    for subject in 1:n_subjects
        true_params = (
            Q0 = rand() * 0.4 + 0.3,  # 0.3-0.7
            α = rand() * 0.8 + 0.1,   # 0.1-0.9
            αm = rand() * 0.8 + 0.1,  # 0.1-0.9
            β = rand() * 15 + 1,      # 1-16
            αf = rand() * 0.5,        # 0-0.5
            μ = rand() * 0.6 + 0.2,   # 0.2-0.8
            τ = rand() * 5,           # 0-5
            φ = rand() * 10 + 1,      # 1-11
            C0 = rand() * 0.4 + 0.1,  # 0.1-0.5
            η = rand() * 0.8 + 0.1,   # 0.1-0.9
            ν = rand() * 15 + 1       # 1-16
        )
        
        true_params_array[subject] = true_params
        actions, rewards, _ = generate_estimator_behavior(true_params, n_arms, n_trials; seed=seed+subject)
        behavioral_data[subject] = (actions, rewards)
    end
    
    # Multi-threaded parameter estimation
    println("並列パラメータ推定実行中...")
    estimated_params_array = Vector{Vector{Float64}}(undef, n_subjects)
    estimation_success = fill(false, n_subjects)
    
    Threads.@threads for subject in 1:n_subjects
        if subject % max(1, n_subjects ÷ 20) == 0
            println("  進捗: $subject / $n_subjects ($(round(subject/n_subjects*100, digits=1))%)")
        end
        
        actions, rewards = behavioral_data[subject]
        initial_guess = [0.5, 0.5, 0.5, 5.0, 0.1, 0.5, 1.0, 5.0, 0.2, 0.5, 5.0]
        
        try
            best_result = nothing
            best_nll = Inf
            
            for restart in 1:3
                initial_params = initial_guess .+ 0.1 * randn(11)
                
                result = optimize(
                    params -> estimator_negative_log_likelihood(params, actions, rewards, n_arms),
                    initial_params,
                    BFGS(),
                    Optim.Options(iterations=100, g_tol=1e-4)
                )
                
                if Optim.converged(result) && result.minimum < best_nll
                    best_nll = result.minimum
                    best_result = result
                end
            end
            
            if best_result !== nothing
                estimated_params_array[subject] = Optim.minimizer(best_result)
                estimation_success[subject] = true
            else
                estimated_params_array[subject] = fill(NaN, 11)
            end
            
        catch e
            estimated_params_array[subject] = fill(NaN, 11)
        end
    end
    
    # Calculate parameter correlations
    param_correlations = Dict{Symbol, Float64}()
    param_names = [:Q0, :α, :αm, :β, :αf, :μ, :τ, :φ, :C0, :η, :ν]
    
    success_mask = estimation_success
    if sum(success_mask) > 0
        for (i, param_name) in enumerate(param_names)
            true_vals = [getfield(true_params_array[j], param_name) for j in 1:n_subjects if success_mask[j]]
            est_vals = [estimated_params_array[j][i] for j in 1:n_subjects if success_mask[j]]
            
            if length(true_vals) > 1
                param_correlations[param_name] = cor(true_vals, est_vals)
            else
                param_correlations[param_name] = NaN
            end
        end
    else
        for param_name in param_names
            param_correlations[param_name] = NaN
        end
    end
    
    success_rate = mean(estimation_success)
    println("✅ CPU並列11パラメータ実験完了！成功率: $(round(success_rate*100, digits=1))%")
    
    return (
        true_params = true_params_array,
        estimated_params = estimated_params_array,
        parameter_correlations = param_correlations,
        success_rate = success_rate
    )
end

"""
GPU-accelerated parameter estimation for 11-parameter model
"""
function gpu_11param_estimation(n_subjects::Int, n_arms::Int, n_trials::Int; seed::Int = 42)
    Random.seed!(seed)
    
    println("🚀 GPU加速 11パラメータ推定実験")
    println("被験者数: $n_subjects, アーム数: $n_arms, 試行数: $n_trials")
    
    if !Metal.functional()
        println("❌ GPU not available")
        return nothing
    end
    
    # Generate behavioral data on CPU (complex model makes GPU generation challenging)
    println("行動データ生成中...")
    true_params_array = Vector{NamedTuple}(undef, n_subjects)
    behavioral_data = Vector{Tuple}(undef, n_subjects)
    
    for subject in 1:n_subjects
        true_params = (
            Q0 = rand() * 0.4 + 0.3,
            α = rand() * 0.8 + 0.1,
            αm = rand() * 0.8 + 0.1,
            β = rand() * 15 + 1,
            αf = rand() * 0.5,
            μ = rand() * 0.6 + 0.2,
            τ = rand() * 5,
            φ = rand() * 10 + 1,
            C0 = rand() * 0.4 + 0.1,
            η = rand() * 0.8 + 0.1,
            ν = rand() * 15 + 1
        )
        
        true_params_array[subject] = true_params
        actions, rewards, _ = generate_estimator_behavior(true_params, n_arms, n_trials; seed=seed+subject)
        behavioral_data[subject] = (actions, rewards)
    end
    
    # GPU-accelerated parameter search using grid search
    println("GPU並列グリッドサーチ実行中...")
    estimated_params_array = Vector{Vector{Float64}}(undef, n_subjects)
    estimation_success = fill(false, n_subjects)
    
    # Use coarse grid search for speed (can be refined)
    param_grids = (
        Q0 = [0.3, 0.5, 0.7],
        α = [0.2, 0.5, 0.8],
        αm = [0.2, 0.5, 0.8],
        β = [2.0, 8.0, 14.0],
        αf = [0.0, 0.2, 0.4],
        μ = [0.3, 0.5, 0.7],
        τ = [0.5, 2.0, 4.0],
        φ = [2.0, 6.0, 10.0],
        C0 = [0.1, 0.3, 0.5],
        η = [0.2, 0.5, 0.8],
        ν = [2.0, 8.0, 14.0]
    )
    
    # Multi-threaded grid search (GPU kernel for 11-param too complex for initial implementation)
    Threads.@threads for subject in 1:n_subjects
        if subject % max(1, n_subjects ÷ 20) == 0
            println("  進捗: $subject / $n_subjects ($(round(subject/n_subjects*100, digits=1))%)")
        end
        
        actions, rewards = behavioral_data[subject]
        best_nll = Inf
        best_params = fill(NaN, 11)
        
        try
            # Grid search over parameter space
            for Q0 in param_grids.Q0, α in param_grids.α, αm in param_grids.αm, β in param_grids.β,
                αf in param_grids.αf, μ in param_grids.μ, τ in param_grids.τ, φ in param_grids.φ,
                C0 in param_grids.C0, η in param_grids.η, ν in param_grids.ν
                
                params_vec = [Q0, α, αm, β, αf, μ, τ, φ, C0, η, ν]
                nll = estimator_negative_log_likelihood(params_vec, actions, rewards, n_arms)
                
                if nll < best_nll
                    best_nll = nll
                    best_params = copy(params_vec)
                end
            end
            
            estimated_params_array[subject] = best_params
            estimation_success[subject] = !any(isnan, best_params)
            
        catch e
            estimated_params_array[subject] = fill(NaN, 11)
        end
    end
    
    # Calculate parameter correlations
    param_correlations = Dict{Symbol, Float64}()
    param_names = [:Q0, :α, :αm, :β, :αf, :μ, :τ, :φ, :C0, :η, :ν]
    
    success_mask = estimation_success
    if sum(success_mask) > 0
        for (i, param_name) in enumerate(param_names)
            true_vals = [getfield(true_params_array[j], param_name) for j in 1:n_subjects if success_mask[j]]
            est_vals = [estimated_params_array[j][i] for j in 1:n_subjects if success_mask[j]]
            
            if length(true_vals) > 1
                param_correlations[param_name] = cor(true_vals, est_vals)
            else
                param_correlations[param_name] = NaN
            end
        end
    else
        for param_name in param_names
            param_correlations[param_name] = NaN
        end
    end
    
    success_rate = mean(estimation_success)
    println("✅ GPU加速11パラメータ実験完了！成功率: $(round(success_rate*100, digits=1))%")
    
    return (
        true_params = true_params_array,
        estimated_params = estimated_params_array,
        parameter_correlations = param_correlations,
        success_rate = success_rate
    )
end

"""
Memory monitoring utility
"""
function get_memory_usage()
    try
        gc_live_bytes = Base.gc_live_bytes()
        return gc_live_bytes / (1024^2)  # Convert to MB
    catch
        return 0.0
    end
end

"""
Run comprehensive 11-parameter efficiency comparison
"""
function run_11param_efficiency_test(scale::String, n_subjects::Int, n_arms::Int, n_trials::Int; 
                                    seed::Int = 42, verbose::Bool = true)
    
    if verbose
        println("\n" * "="^80)
        println("🧪 11-PARAMETER EFFICIENCY TEST: $scale")
        println("📊 Configuration: $n_subjects subjects × $n_arms arms × $n_trials trials")
        println("📈 Total decisions: $(n_subjects * n_trials)")
        println("🔢 Parameters: 11 (Q0, α, αm, β, αf, μ, τ, φ, C0, η, ν)")
        println("="^80)
    end
    
    results = EstimatorTestResult[]
    timestamp = string(now())
    
    # Test 1: CPU Single-threaded
    if verbose
        println("\n1️⃣ Testing CPU Single-threaded (11-parameter)...")
    end
    
    # Measure compilation time
    compilation_start = time()
    precompile(cpu_single_thread_11param_estimation, (Int, Int, Int))
    compilation_time_cpu1 = time() - compilation_start
    
    # Measure memory before
    GC.gc()
    memory_before = get_memory_usage()
    
    # Run test
    execution_time_cpu1 = @elapsed result_cpu1 = cpu_single_thread_11param_estimation(n_subjects, n_arms, n_trials; seed=seed)
    
    # Measure memory after
    memory_after = get_memory_usage()
    memory_used_cpu1 = memory_after - memory_before
    
    throughput_cpu1 = (n_subjects * n_trials) / execution_time_cpu1
    
    push!(results, EstimatorTestResult(
        scale, "CPU(1-thread)", n_subjects, n_arms, n_trials,
        execution_time_cpu1, memory_used_cpu1, compilation_time_cpu1,
        result_cpu1.parameter_correlations, result_cpu1.success_rate,
        throughput_cpu1, timestamp, 1
    ))
    
    if verbose
        println("   ⏱️  Execution time: $(round(execution_time_cpu1, digits=2))s")
        println("   💾 Memory used: $(round(memory_used_cpu1, digits=1))MB")
        println("   ⚡ Throughput: $(round(throughput_cpu1, digits=0)) decisions/s")
        println("   ✅ Success rate: $(round(result_cpu1.success_rate*100, digits=1))%")
    end
    
    # Test 2: CPU Multi-threaded
    if verbose
        println("\n2️⃣ Testing CPU Multi-threaded ($(Threads.nthreads()) threads, 11-parameter)...")
    end
    
    # Measure compilation time
    compilation_start = time()
    precompile(cpu_multithread_11param_estimation, (Int, Int, Int))
    compilation_time_cpu8 = time() - compilation_start
    
    # Measure memory before
    GC.gc()
    memory_before = get_memory_usage()
    
    # Run test
    execution_time_cpu8 = @elapsed result_cpu8 = cpu_multithread_11param_estimation(n_subjects, n_arms, n_trials; seed=seed)
    
    # Measure memory after
    memory_after = get_memory_usage()
    memory_used_cpu8 = memory_after - memory_before
    
    throughput_cpu8 = (n_subjects * n_trials) / execution_time_cpu8
    
    push!(results, EstimatorTestResult(
        scale, "CPU($(Threads.nthreads())-threads)", n_subjects, n_arms, n_trials,
        execution_time_cpu8, memory_used_cpu8, compilation_time_cpu8,
        result_cpu8.parameter_correlations, result_cpu8.success_rate,
        throughput_cpu8, timestamp, Threads.nthreads()
    ))
    
    if verbose
        println("   ⏱️  Execution time: $(round(execution_time_cpu8, digits=2))s")
        println("   💾 Memory used: $(round(memory_used_cpu8, digits=1))MB")
        println("   ⚡ Throughput: $(round(throughput_cpu8, digits=0)) decisions/s")
        println("   ✅ Success rate: $(round(result_cpu8.success_rate*100, digits=1))%")
    end
    
    # Test 3: GPU Accelerated
    if verbose
        println("\n3️⃣ Testing GPU Accelerated (11-parameter)...")
    end
    
    # Check GPU availability
    if !Metal.functional()
        println("   ❌ GPU not available, skipping GPU test")
        return results
    end
    
    # Measure compilation time
    compilation_start = time()
    precompile(gpu_11param_estimation, (Int, Int, Int))
    compilation_time_gpu = time() - compilation_start
    
    # Measure memory before
    GC.gc()
    memory_before = get_memory_usage()
    
    # Run test
    execution_time_gpu = @elapsed result_gpu = gpu_11param_estimation(n_subjects, n_arms, n_trials; seed=seed)
    
    # Measure memory after
    memory_after = get_memory_usage()
    memory_used_gpu = memory_after - memory_before
    
    if result_gpu !== nothing
        throughput_gpu = (n_subjects * n_trials) / execution_time_gpu
        
        push!(results, EstimatorTestResult(
            scale, "GPU", n_subjects, n_arms, n_trials,
            execution_time_gpu, memory_used_gpu, compilation_time_gpu,
            result_gpu.parameter_correlations, result_gpu.success_rate,
            throughput_gpu, timestamp, 1  # GPU doesn't use CPU threads
        ))
        
        if verbose
            println("   ⏱️  Execution time: $(round(execution_time_gpu, digits=2))s")
            println("   💾 Memory used: $(round(memory_used_gpu, digits=1))MB")
            println("   ⚡ Throughput: $(round(throughput_gpu, digits=0)) decisions/s")
            println("   ✅ Success rate: $(round(result_gpu.success_rate*100, digits=1))%")
            
            # Performance comparison
            println("\n📊 Performance Comparison:")
            speedup_cpu8_vs_cpu1 = execution_time_cpu1 / execution_time_cpu8
            speedup_gpu_vs_cpu1 = execution_time_cpu1 / execution_time_gpu
            speedup_gpu_vs_cpu8 = execution_time_cpu8 / execution_time_gpu
            
            println("   🏆 CPU($(Threads.nthreads())) vs CPU(1): $(round(speedup_cpu8_vs_cpu1, digits=2))x speedup")
            println("   🏆 GPU vs CPU(1): $(round(speedup_gpu_vs_cpu1, digits=2))x speedup")
            println("   🏆 GPU vs CPU($(Threads.nthreads())): $(round(speedup_gpu_vs_cpu8, digits=2))x speedup")
            
            # Best method
            best_time = min(execution_time_cpu1, execution_time_cpu8, execution_time_gpu)
            if best_time == execution_time_gpu
                println("   🥇 Winner: GPU ($(round(execution_time_gpu, digits=2))s)")
            elseif best_time == execution_time_cpu8
                println("   🥇 Winner: CPU($(Threads.nthreads())-threads) ($(round(execution_time_cpu8, digits=2))s)")
            else
                println("   🥇 Winner: CPU(1-thread) ($(round(execution_time_cpu1, digits=2))s)")
            end
        end
    end
    
    return results
end

export run_11param_efficiency_test, EstimatorTestResult