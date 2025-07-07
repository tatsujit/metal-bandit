using Random
using Statistics
using Distributions
using CSV
using DataFrames
using CairoMakie
using Dates
using LinearAlgebra
using Optim

# Include dependencies
include("estimator.jl")
include("comprehensive_parameter_recovery_study.jl")

"""
9-Parameter Recovery Study using 11-parameter model with C0=0.0, Q0=0.0
"""

function run_9parameter_recovery_study()
    println("🔬 9-PARAMETER RECOVERY STUDY (C0=0.0, Q0=0.0 fixed)")
    println("🧵 Using 14 threads for optimal performance")
    println("="^80)
    
    # Test conditions (subset for faster results)
    test_conditions = [
        (2, 2, 150, true, "2arms_150steps_stationary"),
        (2, 2, 150, false, "2arms_150steps_nonstationary"),
        (9, 5, 150, true, "9arms_5avail_150steps_stationary"),
        (9, 9, 150, true, "9arms_150steps_stationary"),
        (9, 9, 150, false, "9arms_150steps_nonstationary"),
    ]
    
    n_subjects_per_condition = 20  # Smaller sample size for faster results
    
    println("\n📊 Test Conditions:")
    for (i, (n_arms, n_avail, n_steps, is_stat, desc)) in enumerate(test_conditions)
        println("   $i. $desc: $n_subjects_per_condition subjects")
    end
    
    all_results = []
    start_time = time()
    
    # Run each condition
    for (i, (n_arms, n_avail_arms, n_steps, is_stationary, description)) in enumerate(test_conditions)
        condition_start = time()
        
        println("\n" * "="^60)
        println("🔄 Condition $i/$(length(test_conditions)): $description")
        
        # Create environment
        change_points = is_stationary ? Int[] : [51, 101]
        env = LAEnvironment(n_arms, n_avail_arms, n_steps; 
                           is_stationary=is_stationary, change_points=change_points)
        
        # Run parameter recovery
        condition_results = run_9param_recovery_condition(env, n_subjects_per_condition, description)
        
        push!(all_results, condition_results)
        
        condition_time = time() - condition_start
        println("⏰ Condition completed in $(round(condition_time, digits=1))s")
    end
    
    total_time = time() - start_time
    println("\n🎉 STUDY COMPLETED!")
    println("Total time: $(round(total_time/60, digits=1)) minutes")
    
    # Create comparison table
    create_9param_comparison_table(all_results)
    
    return all_results
end

function run_9param_recovery_condition(env::LAEnvironment, n_subjects::Int, description::String)
    println("   Subjects: $n_subjects")
    println("   Environment: $(env.n_arms) arms, $(env.n_avail_arms) available")
    println("   Stationary: $(env.is_stationary)")
    
    # Generate behavioral data with C0=0.0, Q0=0.0
    println("   🔄 Generating behavioral data...")
    behavioral_data = generate_9param_behavioral_data(env, n_subjects)
    
    # Run parameter estimation
    println("   🧠 Running parameter estimation (14 threads)...")
    start_time = time()
    
    estimation_results = estimate_9parameters(behavioral_data, env.n_arms)
    
    execution_time = time() - start_time
    
    # Calculate recovery metrics
    recovery_metrics = calculate_9param_metrics(
        behavioral_data.true_params, 
        estimation_results.estimated_params,
        estimation_results.success_mask
    )
    
    println("   ✅ Completed in $(round(execution_time, digits=1))s")
    println("   Success rate: $(round(recovery_metrics.success_rate*100, digits=1))%")
    println("   Average correlation: $(round(recovery_metrics.avg_correlation, digits=3))")
    
    return (
        condition = description,
        n_arms = env.n_arms,
        n_subjects = n_subjects,
        execution_time = execution_time,
        recovery_metrics = recovery_metrics,
        estimation_results = estimation_results
    )
end

function generate_9param_behavioral_data(env::LAEnvironment, n_subjects::Int)
    Random.seed!(42)  # Reproducible results
    
    true_params_array = []
    actions_array = Matrix{Int}(undef, n_subjects, length(env.available_arms))
    rewards_array = Matrix{Float64}(undef, n_subjects, length(env.available_arms))
    
    for subject in 1:n_subjects
        # Sample true parameters (9 parameters only, C0=0.0, Q0=0.0)
        true_params = (
            Q0 = 0.0,  # Fixed
            α = 0.1 + 0.8 * rand(),       # 0.1-0.9
            αm = 0.1 + 0.8 * rand(),      # 0.1-0.9  
            β = 1.0 + 15.0 * rand(),      # 1.0-16.0
            αf = 0.5 * rand(),            # 0.0-0.5
            μ = 0.2 + 0.6 * rand(),       # 0.2-0.8
            τ = 5.0 * rand(),             # 0.0-5.0
            φ = 1.0 + 10.0 * rand(),      # 1.0-11.0
            C0 = 0.0,  # Fixed
            η = 0.1 + 0.8 * rand(),       # 0.1-0.9
            ν = 1.0 + 15.0 * rand()       # 1.0-16.0
        )
        push!(true_params_array, true_params)
        
        # Generate behavior with Estimator using fixed C0=0.0, Q0=0.0
        estimator = Estimator(env.n_arms;
            Q0=0.0, α=true_params.α, αm=true_params.αm, β=true_params.β,
            αf=true_params.αf, μ=true_params.μ, τ=true_params.τ, 
            φ=true_params.φ, C0=0.0, η=true_params.η, ν=true_params.ν
        )
        
        actions = Vector{Int}(undef, length(env.available_arms))
        rewards = Vector{Float64}(undef, length(env.available_arms))
        
        for step in 1:length(env.available_arms)
            available_arms = env.available_arms[step]
            probs = selection_probabilities(estimator.W, available_arms)
            
            # Sample action
            rand_val = rand()
            cumsum_prob = 0.0
            chosen_arm = available_arms[1]
            
            for arm in available_arms
                cumsum_prob += probs[arm]
                if rand_val <= cumsum_prob
                    chosen_arm = arm
                    break
                end
            end
            
            actions[step] = chosen_arm
            
            # Generate reward
            current_probs = get_reward_probs(env, step)
            rewards[step] = rand() < current_probs[chosen_arm] ? 1.0 : 0.0
            
            # Update estimator
            update!(estimator, chosen_arm, rewards[step])
        end
        
        actions_array[subject, :] = actions
        rewards_array[subject, :] = rewards
    end
    
    return (
        true_params = true_params_array,
        actions = actions_array,
        rewards = rewards_array,
        n_subjects = n_subjects,
        n_steps = length(env.available_arms)
    )
end

function estimate_9parameters(behavioral_data, n_arms)
    n_subjects = behavioral_data.n_subjects
    # Still 11 columns but C0 and Q0 will be fixed at 0.0
    estimated_params = Matrix{Float64}(undef, n_subjects, 11)
    success_mask = fill(false, n_subjects)
    nlls = fill(Inf, n_subjects)
    
    # Parallel estimation
    Threads.@threads for subject in 1:n_subjects
        actions = behavioral_data.actions[subject, :]
        rewards = behavioral_data.rewards[subject, :]
        
        # Estimate parameters with C0=0.0, Q0=0.0 fixed
        result = estimate_9param_single_subject(actions, rewards, n_arms)
        
        if result !== nothing && result.success
            estimated_params[subject, :] = result.params
            success_mask[subject] = true
            nlls[subject] = result.nll
        else
            estimated_params[subject, :] = fill(NaN, 11)
        end
    end
    
    return (
        estimated_params = estimated_params,
        success_mask = success_mask,
        nlls = nlls,
        success_rate = mean(success_mask)
    )
end

function estimate_9param_single_subject(actions::Vector{Int}, rewards::Vector{Float64}, n_arms::Int)
    try
        # Define custom NLL function that fixes C0=0.0, Q0=0.0
        function nll_9param(params_vec::Vector{Float64})
            # params_vec has 9 elements
            # Reconstruct 11-element vector with C0=0.0, Q0=0.0
            full_params = [0.0;  # Q0 fixed
                          params_vec[1:7];  # α, αm, β, αf, μ, τ, φ
                          0.0;  # C0 fixed
                          params_vec[8:9]]  # η, ν
            
            return estimator_negative_log_likelihood(full_params, actions, rewards, n_arms)
        end
        
        # Initial guess for 9 parameters
        initial_guess = [0.5, 0.5, 5.0, 0.1, 0.5, 1.0, 5.0, 0.5, 5.0]
        
        # Bounds for 9 parameters
        lower_bounds = [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.1]
        upper_bounds = [1.0, 1.0, 20.0, 1.0, 1.0, 10.0, 20.0, 1.0, 20.0]
        
        best_result = nothing
        best_nll = Inf
        
        # Multiple restarts
        for restart in 1:3
            perturbed_guess = initial_guess .+ 0.1 * randn(9)
            perturbed_guess = max.(lower_bounds, min.(upper_bounds, perturbed_guess))
            
            result = optimize(
                nll_9param,
                lower_bounds,
                upper_bounds,
                perturbed_guess,
                Fminbox(BFGS()),
                Optim.Options(iterations=100, g_tol=1e-4)
            )
            
            if Optim.converged(result) && result.minimum < best_nll
                best_nll = result.minimum
                best_result = result
            end
        end
        
        if best_result !== nothing
            # Convert back to 11-parameter format
            params_9 = Optim.minimizer(best_result)
            full_params = [0.0;  # Q0
                          params_9[1:7];  # α, αm, β, αf, μ, τ, φ
                          0.0;  # C0
                          params_9[8:9]]  # η, ν
            
            return (
                params = full_params,
                nll = best_result.minimum,
                success = true
            )
        end
        
        return nothing
    catch
        return nothing
    end
end

function calculate_9param_metrics(true_params_array, estimated_params, success_mask)
    # Focus on the 9 free parameters (excluding Q0 and C0)
    param_names = [:α, :αm, :β, :αf, :μ, :τ, :φ, :η, :ν]
    param_indices = [2, 3, 4, 5, 6, 7, 8, 10, 11]  # Indices in 11-param vector
    
    correlations = Dict{Symbol, Float64}()
    rmse_values = Dict{Symbol, Float64}()
    
    successful_subjects = findall(success_mask)
    
    if length(successful_subjects) > 1
        for (i, (param_name, idx)) in enumerate(zip(param_names, param_indices))
            true_vals = [getfield(true_params_array[j], param_name) for j in successful_subjects]
            est_vals = [estimated_params[j, idx] for j in successful_subjects]
            
            if length(true_vals) > 1 && !all(isnan.(est_vals))
                correlations[param_name] = cor(true_vals, est_vals)
                rmse_values[param_name] = sqrt(mean((true_vals .- est_vals).^2))
            else
                correlations[param_name] = NaN
                rmse_values[param_name] = NaN
            end
        end
    else
        for param_name in param_names
            correlations[param_name] = NaN
            rmse_values[param_name] = NaN
        end
    end
    
    # Overall metrics
    valid_correlations = [v for v in values(correlations) if !isnan(v)]
    avg_correlation = length(valid_correlations) > 0 ? mean(valid_correlations) : NaN
    
    return (
        correlations = correlations,
        rmse_values = rmse_values,
        avg_correlation = avg_correlation,
        success_rate = mean(success_mask),
        n_successful = length(successful_subjects)
    )
end

function create_9param_comparison_table(all_results)
    println("\n📊 9-PARAMETER RECOVERY RESULTS (C0=0.0, Q0=0.0)")
    println("="^80)
    
    # Calculate average correlations across conditions
    param_names = [:α, :αm, :β, :αf, :μ, :τ, :φ, :η, :ν]
    avg_correlations = Dict{Symbol, Vector{Float64}}()
    
    for param in param_names
        avg_correlations[param] = []
        for result in all_results
            corr = get(result.recovery_metrics.correlations, param, NaN)
            if !isnan(corr)
                push!(avg_correlations[param], corr)
            end
        end
    end
    
    # Create summary table
    println("\n| Parameter | Average Correlation | Std Dev | Min | Max | N Conditions |")
    println("|-----------|-------------------|---------|-----|-----|--------------|")
    
    param_descriptions = Dict(
        :α => "Learning rate",
        :αm => "Memory decay",
        :β => "Inverse temperature",
        :αf => "Forgetting rate",
        :μ => "Utility weight",
        :τ => "Exploration parameter",
        :φ => "Value scale",
        :η => "Perseveration learning",
        :ν => "Perseveration inverse temp"
    )
    
    # Sort by average correlation
    sorted_params = []
    for param in param_names
        if !isempty(avg_correlations[param])
            avg_corr = mean(avg_correlations[param])
            push!(sorted_params, (param, avg_corr, avg_correlations[param]))
        end
    end
    sort!(sorted_params, by=x->x[2], rev=true)
    
    for (param, avg_corr, corr_values) in sorted_params
        desc = get(param_descriptions, param, "Unknown")
        std_corr = std(corr_values)
        min_corr = minimum(corr_values)
        max_corr = maximum(corr_values)
        n_cond = length(corr_values)
        
        status = if avg_corr > 0.3
            "🟢"
        elseif avg_corr > 0.1
            "🟡"
        elseif avg_corr > 0.0
            "🟠"
        else
            "🔴"
        end
        
        println("| $param ($desc) | $(round(avg_corr, digits=3)) $status | $(round(std_corr, digits=3)) | $(round(min_corr, digits=3)) | $(round(max_corr, digits=3)) | $n_cond |")
    end
    
    # Overall statistics
    all_success_rates = [r.recovery_metrics.success_rate for r in all_results]
    all_avg_correlations = [r.recovery_metrics.avg_correlation for r in all_results]
    
    println("\n📊 OVERALL STATISTICS")
    println("Average Success Rate: $(round(mean(all_success_rates)*100, digits=1))%")
    println("Average Correlation (across all parameters): $(round(mean(all_avg_correlations), digits=3))")
    
    # Condition-specific results
    println("\n📈 CONDITION-SPECIFIC RESULTS")
    println("| Condition | Success Rate | Avg Correlation | Time (s) |")
    println("|-----------|--------------|-----------------|----------|")
    
    for result in all_results
        println("| $(result.condition) | $(round(result.recovery_metrics.success_rate*100, digits=1))% | $(round(result.recovery_metrics.avg_correlation, digits=3)) | $(round(result.execution_time, digits=1)) |")
    end
    
    # Save detailed results
    save_9param_results(all_results, sorted_params)
    
    return sorted_params
end

function save_9param_results(all_results, sorted_params)
    # Create detailed CSV
    df_data = []
    
    for (param, avg_corr, corr_values) in sorted_params
        push!(df_data, (
            Parameter = string(param),
            Average_Correlation = round(avg_corr, digits=3),
            Std_Dev = round(std(corr_values), digits=3),
            Min = round(minimum(corr_values), digits=3),
            Max = round(maximum(corr_values), digits=3),
            N_Conditions = length(corr_values)
        ))
    end
    
    df = DataFrame(df_data)
    CSV.write("9parameter_recovery_results.csv", df)
    println("\n📁 Results saved to: 9parameter_recovery_results.csv")
end

# Run the study
if abspath(PROGRAM_FILE) == @__FILE__
    println("Starting 9-parameter recovery study...")
    results = run_9parameter_recovery_study()
end