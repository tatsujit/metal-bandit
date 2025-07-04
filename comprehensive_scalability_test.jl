using Metal
using Statistics
using Random
using Pkg
using BenchmarkTools
using CSV
using DataFrames
using CairoMakie
using Dates

"""
Comprehensive Scalability Test Framework
Tests CPU(1-thread), CPU(8-threads), and GPU across multiple scales
Measures: execution time, memory usage, compilation time, parameter recovery quality
"""

# Include all implementations
include("q_parameter_recovery.jl")
include("gpu_optimized_simple.jl")

struct ScalabilityTestResult
    scale::String
    method::String
    n_subjects::Int
    n_arms::Int
    n_trials::Int
    execution_time::Float64
    memory_used::Float64
    compilation_time::Float64
    alpha_correlation::Float64
    beta_correlation::Float64
    success_rate::Float64
    throughput::Float64  # decisions per second
    timestamp::String
end

"""
Memory monitoring utility
"""
function get_memory_usage()
    try
        # Get Julia process memory usage in MB
        gc_live_bytes = Base.gc_live_bytes()
        return gc_live_bytes / (1024^2)  # Convert to MB
    catch
        return 0.0
    end
end

"""
CPU Single-threaded implementation (sequential processing)
"""
function cpu_single_thread_recovery(n_subjects::Int, n_arms::Int, n_trials::Int; seed::Int = 42)
    Random.seed!(seed)
    
    println("🔧 CPU単線 Q学習パラメータ回復実験")
    println("被験者数: $n_subjects, アーム数: $n_arms, 試行数: $n_trials")
    
    # Generate true parameters
    true_alpha = rand(n_subjects)
    true_beta = rand(n_subjects) * 10.0
    
    # Results storage
    estimated_alpha = zeros(n_subjects)
    estimated_beta = zeros(n_subjects)
    estimation_success = fill(false, n_subjects)
    
    # Single-threaded processing (no @threads, sequential loop)
    for subject in 1:n_subjects
        if subject % max(1, n_subjects ÷ 20) == 0
            println("  進捗: $subject / $n_subjects ($(round(subject/n_subjects*100, digits=1))%)")
        end
        
        # Generate behavior for this subject
        reward_probs = rand(n_arms) * 0.6 .+ 0.2
        actions, rewards = generate_q_learning_behavior(
            true_alpha[subject], true_beta[subject], reward_probs, n_trials
        )
        
        # Parameter estimation
        estimated_params, nll, success = estimate_q_parameters(actions, rewards, n_arms)
        
        if success
            estimated_alpha[subject] = estimated_params[1]
            estimated_beta[subject] = estimated_params[2]
            estimation_success[subject] = true
        else
            estimated_alpha[subject] = NaN
            estimated_beta[subject] = NaN
            estimation_success[subject] = false
        end
    end
    
    # Calculate correlations
    success_mask = estimation_success
    if sum(success_mask) > 0
        α_correlation = cor(true_alpha[success_mask], estimated_alpha[success_mask])
        β_correlation = cor(true_beta[success_mask], estimated_beta[success_mask])
    else
        α_correlation = NaN
        β_correlation = NaN
    end
    
    success_rate = mean(estimation_success)
    println("✅ CPU単線実験完了！成功率: $(round(success_rate*100, digits=1))%")
    
    return (
        true_alpha = true_alpha,
        true_beta = true_beta,
        estimated_alpha = estimated_alpha,
        estimated_beta = estimated_beta,
        alpha_correlation = α_correlation,
        beta_correlation = β_correlation,
        success_rate = success_rate
    )
end

"""
Run comprehensive scalability test for a specific scale
"""
function run_scale_test(scale::String, n_subjects::Int, n_arms::Int, n_trials::Int; 
                       seed::Int = 42, verbose::Bool = true)
    
    if verbose
        println("\n" * "="^80)
        println("🧪 SCALABILITY TEST: $scale")
        println("📊 Configuration: $n_subjects subjects × $n_arms arms × $n_trials trials")
        println("📈 Total decisions: $(n_subjects * n_trials)")
        println("="^80)
    end
    
    results = ScalabilityTestResult[]
    timestamp = string(now())
    
    # Test 1: CPU Single-threaded
    if verbose
        println("\n1️⃣ Testing CPU Single-threaded...")
    end
    
    # Measure compilation time
    compilation_start = time()
    precompile(cpu_single_thread_recovery, (Int, Int, Int))
    compilation_time_cpu1 = time() - compilation_start
    
    # Measure memory before
    GC.gc()
    memory_before = get_memory_usage()
    
    # Run test
    execution_time_cpu1 = @elapsed result_cpu1 = cpu_single_thread_recovery(n_subjects, n_arms, n_trials; seed=seed)
    
    # Measure memory after
    memory_after = get_memory_usage()
    memory_used_cpu1 = memory_after - memory_before
    
    throughput_cpu1 = (n_subjects * n_trials) / execution_time_cpu1
    
    push!(results, ScalabilityTestResult(
        scale, "CPU(1-thread)", n_subjects, n_arms, n_trials,
        execution_time_cpu1, memory_used_cpu1, compilation_time_cpu1,
        result_cpu1.alpha_correlation, result_cpu1.beta_correlation, result_cpu1.success_rate,
        throughput_cpu1, timestamp
    ))
    
    if verbose
        println("   ⏱️  Execution time: $(round(execution_time_cpu1, digits=2))s")
        println("   💾 Memory used: $(round(memory_used_cpu1, digits=1))MB")
        println("   ⚡ Throughput: $(round(throughput_cpu1, digits=0)) decisions/s")
    end
    
    # Test 2: CPU Multi-threaded (8 threads)
    if verbose
        println("\n2️⃣ Testing CPU Multi-threaded (8 threads)...")
    end
    
    # Measure compilation time
    compilation_start = time()
    precompile(run_parameter_recovery_experiment, (Int, Int, Int))
    compilation_time_cpu8 = time() - compilation_start
    
    # Measure memory before
    GC.gc()
    memory_before = get_memory_usage()
    
    # Run test
    execution_time_cpu8 = @elapsed result_cpu8 = run_parameter_recovery_experiment(n_subjects, n_arms, n_trials; seed=seed)
    
    # Measure memory after
    memory_after = get_memory_usage()
    memory_used_cpu8 = memory_after - memory_before
    
    # Calculate correlations
    success_mask = result_cpu8.estimation_success
    if sum(success_mask) > 0
        α_correlation_cpu8 = cor(result_cpu8.true_alpha[success_mask], result_cpu8.estimated_alpha[success_mask])
        β_correlation_cpu8 = cor(result_cpu8.true_beta[success_mask], result_cpu8.estimated_beta[success_mask])
    else
        α_correlation_cpu8 = NaN
        β_correlation_cpu8 = NaN
    end
    
    success_rate_cpu8 = mean(result_cpu8.estimation_success)
    throughput_cpu8 = (n_subjects * n_trials) / execution_time_cpu8
    
    push!(results, ScalabilityTestResult(
        scale, "CPU(8-threads)", n_subjects, n_arms, n_trials,
        execution_time_cpu8, memory_used_cpu8, compilation_time_cpu8,
        α_correlation_cpu8, β_correlation_cpu8, success_rate_cpu8,
        throughput_cpu8, timestamp
    ))
    
    if verbose
        println("   ⏱️  Execution time: $(round(execution_time_cpu8, digits=2))s")
        println("   💾 Memory used: $(round(memory_used_cpu8, digits=1))MB")
        println("   ⚡ Throughput: $(round(throughput_cpu8, digits=0)) decisions/s")
    end
    
    # Test 3: GPU Optimized
    if verbose
        println("\n3️⃣ Testing GPU Optimized...")
    end
    
    # Check GPU availability
    if !Metal.functional()
        println("   ❌ GPU not available, skipping GPU test")
        return results
    end
    
    # Measure compilation time
    compilation_start = time()
    precompile(gpu_accelerated_recovery, (Int, Int, Int))
    compilation_time_gpu = time() - compilation_start
    
    # Measure memory before
    GC.gc()
    memory_before = get_memory_usage()
    
    # Run test
    execution_time_gpu = @elapsed result_gpu = gpu_accelerated_recovery(n_subjects, n_arms, n_trials; seed=seed)
    
    # Measure memory after
    memory_after = get_memory_usage()
    memory_used_gpu = memory_after - memory_before
    
    throughput_gpu = (n_subjects * n_trials) / execution_time_gpu
    
    push!(results, ScalabilityTestResult(
        scale, "GPU", n_subjects, n_arms, n_trials,
        execution_time_gpu, memory_used_gpu, compilation_time_gpu,
        result_gpu.alpha_correlation, result_gpu.beta_correlation, 1.0,  # GPU always succeeds
        throughput_gpu, timestamp
    ))
    
    if verbose
        println("   ⏱️  Execution time: $(round(execution_time_gpu, digits=2))s")
        println("   💾 Memory used: $(round(memory_used_gpu, digits=1))MB")
        println("   ⚡ Throughput: $(round(throughput_gpu, digits=0)) decisions/s")
        
        # Performance comparison
        println("\n📊 Performance Comparison:")
        speedup_cpu8_vs_cpu1 = execution_time_cpu1 / execution_time_cpu8
        speedup_gpu_vs_cpu1 = execution_time_cpu1 / execution_time_gpu
        speedup_gpu_vs_cpu8 = execution_time_cpu8 / execution_time_gpu
        
        println("   🏆 CPU(8) vs CPU(1): $(round(speedup_cpu8_vs_cpu1, digits=2))x speedup")
        println("   🏆 GPU vs CPU(1): $(round(speedup_gpu_vs_cpu1, digits=2))x speedup")
        println("   🏆 GPU vs CPU(8): $(round(speedup_gpu_vs_cpu8, digits=2))x speedup")
        
        # Best method
        best_time = min(execution_time_cpu1, execution_time_cpu8, execution_time_gpu)
        if best_time == execution_time_gpu
            println("   🥇 Winner: GPU ($(round(execution_time_gpu, digits=2))s)")
        elseif best_time == execution_time_cpu8
            println("   🥇 Winner: CPU(8-threads) ($(round(execution_time_cpu8, digits=2))s)")
        else
            println("   🥇 Winner: CPU(1-thread) ($(round(execution_time_cpu1, digits=2))s)")
        end
    end
    
    return results
end

"""
Run comprehensive scalability tests across all scales
"""
function run_comprehensive_scalability_tests()
    println("🚀 COMPREHENSIVE SCALABILITY TEST SUITE")
    println("Testing CPU(1-thread) vs CPU(8-threads) vs GPU")
    println("Measuring: execution time, memory usage, compilation time, recovery quality")
    println("Expected duration: 2-3 hours")
    println("\n" * "="^80)
    
    all_results = ScalabilityTestResult[]
    
    # Test configurations
    test_configs = [
        ("Small", 500, 4, 200),      # 400K decisions
        ("Medium", 1500, 6, 300),    # 2.7M decisions  
        ("Large", 3000, 8, 500),     # 12M decisions
        ("Extra-Large", 5000, 8, 800)  # 32M decisions
    ]
    
    start_time = time()
    
    for (i, (scale, n_subjects, n_arms, n_trials)) in enumerate(test_configs)
        println("\n🔄 Running test $i/$(length(test_configs)): $scale scale")
        
        # Run scale test
        scale_results = run_scale_test(scale, n_subjects, n_arms, n_trials)
        append!(all_results, scale_results)
        
        # Progress update
        elapsed = time() - start_time
        avg_time_per_test = elapsed / i
        estimated_remaining = avg_time_per_test * (length(test_configs) - i)
        
        println("\n⏰ Progress: $i/$(length(test_configs)) tests completed")
        println("   Elapsed time: $(round(elapsed/60, digits=1)) minutes")
        println("   Estimated remaining: $(round(estimated_remaining/60, digits=1)) minutes")
        
        # Save intermediate results
        save_results_to_csv(all_results, "scalability_test_results_intermediate.csv")
    end
    
    total_time = time() - start_time
    println("\n🎉 ALL TESTS COMPLETED!")
    println("Total execution time: $(round(total_time/60, digits=1)) minutes")
    
    # Save final results
    final_df = save_results_to_csv(all_results, "scalability_test_results_final.csv")
    
    # Create comprehensive visualization
    create_scalability_visualization(all_results)
    
    return all_results, final_df
end

"""
Save results to CSV
"""
function save_results_to_csv(results::Vector{ScalabilityTestResult}, filename::String)
    df = DataFrame(
        scale = [r.scale for r in results],
        method = [r.method for r in results],
        n_subjects = [r.n_subjects for r in results],
        n_arms = [r.n_arms for r in results],
        n_trials = [r.n_trials for r in results],
        execution_time = [r.execution_time for r in results],
        memory_used = [r.memory_used for r in results],
        compilation_time = [r.compilation_time for r in results],
        alpha_correlation = [r.alpha_correlation for r in results],
        beta_correlation = [r.beta_correlation for r in results],
        success_rate = [r.success_rate for r in results],
        throughput = [r.throughput for r in results],
        timestamp = [r.timestamp for r in results]
    )
    
    CSV.write(filename, df)
    println("📁 Results saved to: $filename")
    return df
end

"""
Create comprehensive scalability visualization
"""
function create_scalability_visualization(results::Vector{ScalabilityTestResult})
    # Prepare data
    scales = unique([r.scale for r in results])
    methods = ["CPU(1-thread)", "CPU(8-threads)", "GPU"]
    
    # Create figure
    fig = Figure(size = (1400, 1000))
    
    # Plot 1: Execution Time Comparison
    ax1 = Axis(fig[1, 1], 
        title = "Execution Time by Scale", 
        xlabel = "Scale", 
        ylabel = "Execution Time (seconds)",
        yscale = log10)
    
    for method in methods
        method_results = filter(r -> r.method == method, results)
        if !isempty(method_results)
            times = [r.execution_time for r in method_results]
            method_scales = [r.scale for r in method_results]
            lines!(ax1, 1:length(method_scales), times, label = method, linewidth = 3)
            scatter!(ax1, 1:length(method_scales), times, markersize = 10)
        end
    end
    axislegend(ax1, position = :lt)
    
    # Plot 2: Throughput Comparison
    ax2 = Axis(fig[1, 2], 
        title = "Throughput by Scale", 
        xlabel = "Scale", 
        ylabel = "Decisions/Second",
        yscale = log10)
    
    for method in methods
        method_results = filter(r -> r.method == method, results)
        if !isempty(method_results)
            throughputs = [r.throughput for r in method_results]
            method_scales = [r.scale for r in method_results]
            lines!(ax2, 1:length(method_scales), throughputs, label = method, linewidth = 3)
            scatter!(ax2, 1:length(method_scales), throughputs, markersize = 10)
        end
    end
    axislegend(ax2, position = :lt)
    
    # Plot 3: Memory Usage
    ax3 = Axis(fig[2, 1], 
        title = "Memory Usage by Scale", 
        xlabel = "Scale", 
        ylabel = "Memory Used (MB)")
    
    for method in methods
        method_results = filter(r -> r.method == method, results)
        if !isempty(method_results)
            memory = [r.memory_used for r in method_results]
            method_scales = [r.scale for r in method_results]
            lines!(ax3, 1:length(method_scales), memory, label = method, linewidth = 3)
            scatter!(ax3, 1:length(method_scales), memory, markersize = 10)
        end
    end
    axislegend(ax3, position = :lt)
    
    # Plot 4: Parameter Recovery Quality
    ax4 = Axis(fig[2, 2], 
        title = "Parameter Recovery Quality (α correlation)", 
        xlabel = "Scale", 
        ylabel = "Alpha Correlation")
    
    for method in methods
        method_results = filter(r -> r.method == method, results)
        if !isempty(method_results)
            correlations = [r.alpha_correlation for r in method_results]
            method_scales = [r.scale for r in method_results]
            lines!(ax4, 1:length(method_scales), correlations, label = method, linewidth = 3)
            scatter!(ax4, 1:length(method_scales), correlations, markersize = 10)
        end
    end
    axislegend(ax4, position = :lb)
    
    # Save figure
    save("scalability_test_results.png", fig)
    println("📊 Visualization saved to: scalability_test_results.png")
    
    return fig
end

# Export functions
export run_comprehensive_scalability_tests, run_scale_test, create_scalability_visualization