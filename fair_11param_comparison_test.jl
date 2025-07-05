using Metal
using Statistics
using Random
using CSV
using DataFrames
using CairoMakie
using Dates

# Include the updated estimator with fair GPU comparison
include("estimator_11_parameter_test.jl")

"""
Fair 11-Parameter Model Efficiency Testing

This script runs a fair comparison between CPU and GPU implementations
where ALL methods use identical BFGS optimization algorithms.

Previous unfair comparison:
- CPU: Used BFGS optimization 
- GPU: Used grid search

Fair comparison:
- CPU(1-thread): BFGS optimization
- CPU(8-threads): BFGS optimization  
- GPU: BFGS optimization (multi-threaded)
"""

function run_fair_11param_comparison(scale::String, n_subjects::Int, n_arms::Int, n_trials::Int; seed::Int = 42)
    println("\n" * "="^100)
    println("🔬 FAIR 11-PARAMETER EFFICIENCY COMPARISON")
    println("🎯 Scale: $scale")
    println("📊 Configuration: $n_subjects subjects × $n_arms arms × $n_trials trials")
    println("🔧 ALL METHODS USE IDENTICAL BFGS OPTIMIZATION")
    println("="^100)
    
    results = EstimatorTestResult[]
    timestamp = string(now())
    
    # Test 1: CPU Single-threaded (BFGS)
    println("\n1️⃣ Testing CPU Single-threaded with BFGS optimization...")
    
    GC.gc()
    memory_before = get_memory_usage()
    execution_time_cpu1 = @elapsed result_cpu1 = cpu_single_thread_11param_estimation(n_subjects, n_arms, n_trials; seed=seed)
    memory_after = get_memory_usage()
    memory_used_cpu1 = memory_after - memory_before
    throughput_cpu1 = (n_subjects * n_trials) / execution_time_cpu1
    
    push!(results, EstimatorTestResult(
        scale, "CPU(1-thread) BFGS", n_subjects, n_arms, n_trials,
        execution_time_cpu1, memory_used_cpu1, 0.0,
        result_cpu1.parameter_correlations, result_cpu1.success_rate,
        throughput_cpu1, timestamp, 1
    ))
    
    println("   ⏱️  Execution time: $(round(execution_time_cpu1, digits=2))s")
    println("   💾 Memory used: $(round(memory_used_cpu1, digits=1))MB")
    println("   ⚡ Throughput: $(round(throughput_cpu1, digits=0)) decisions/s")
    println("   ✅ Success rate: $(round(result_cpu1.success_rate*100, digits=1))%")
    
    # Test 2: CPU Multi-threaded (BFGS)
    println("\n2️⃣ Testing CPU Multi-threaded ($(Threads.nthreads()) threads) with BFGS optimization...")
    
    GC.gc()
    memory_before = get_memory_usage()
    execution_time_cpu8 = @elapsed result_cpu8 = cpu_multithread_11param_estimation(n_subjects, n_arms, n_trials; seed=seed)
    memory_after = get_memory_usage()
    memory_used_cpu8 = memory_after - memory_before
    throughput_cpu8 = (n_subjects * n_trials) / execution_time_cpu8
    
    push!(results, EstimatorTestResult(
        scale, "CPU($(Threads.nthreads())-threads) BFGS", n_subjects, n_arms, n_trials,
        execution_time_cpu8, memory_used_cpu8, 0.0,
        result_cpu8.parameter_correlations, result_cpu8.success_rate,
        throughput_cpu8, timestamp, Threads.nthreads()
    ))
    
    println("   ⏱️  Execution time: $(round(execution_time_cpu8, digits=2))s")
    println("   💾 Memory used: $(round(memory_used_cpu8, digits=1))MB")
    println("   ⚡ Throughput: $(round(throughput_cpu8, digits=0)) decisions/s")
    println("   ✅ Success rate: $(round(result_cpu8.success_rate*100, digits=1))%")
    
    # Test 3: GPU with BFGS (Fair Comparison)
    println("\n3️⃣ Testing GPU with BFGS optimization (FAIR COMPARISON)...")
    
    if !Metal.functional()
        println("   ❌ GPU not available")
        return results
    end
    
    GC.gc()
    memory_before = get_memory_usage()
    execution_time_gpu = @elapsed result_gpu = gpu_11param_estimation(n_subjects, n_arms, n_trials; seed=seed)
    memory_after = get_memory_usage()
    memory_used_gpu = memory_after - memory_before
    
    if result_gpu !== nothing
        throughput_gpu = (n_subjects * n_trials) / execution_time_gpu
        
        push!(results, EstimatorTestResult(
            scale, "GPU BFGS", n_subjects, n_arms, n_trials,
            execution_time_gpu, memory_used_gpu, 0.0,
            result_gpu.parameter_correlations, result_gpu.success_rate,
            throughput_gpu, timestamp, Threads.nthreads()  # GPU uses CPU threads for optimization
        ))
        
        println("   ⏱️  Execution time: $(round(execution_time_gpu, digits=2))s")
        println("   💾 Memory used: $(round(memory_used_gpu, digits=1))MB")
        println("   ⚡ Throughput: $(round(throughput_gpu, digits=0)) decisions/s")
        println("   ✅ Success rate: $(round(result_gpu.success_rate*100, digits=1))%")
        
        # Fair Performance Comparison Analysis
        println("\n📊 FAIR PERFORMANCE COMPARISON ANALYSIS:")
        println("🔧 All methods use identical BFGS optimization algorithm")
        
        speedup_cpu8_vs_cpu1 = execution_time_cpu1 / execution_time_cpu8
        speedup_gpu_vs_cpu1 = execution_time_cpu1 / execution_time_gpu
        speedup_cpu8_vs_gpu = execution_time_gpu / execution_time_cpu8
        
        println("\n🏆 Performance Ratios:")
        println("   CPU($(Threads.nthreads())) vs CPU(1): $(round(speedup_cpu8_vs_cpu1, digits=2))x speedup")
        println("   GPU vs CPU(1): $(round(speedup_gpu_vs_cpu1, digits=2))x speedup")
        
        if speedup_cpu8_vs_gpu > 1.0
            println("   CPU($(Threads.nthreads())) vs GPU: $(round(speedup_cpu8_vs_gpu, digits=2))x FASTER")
        else
            println("   GPU vs CPU($(Threads.nthreads())): $(round(1/speedup_cpu8_vs_gpu, digits=2))x FASTER")
        end
        
        # Best method determination
        best_time = min(execution_time_cpu1, execution_time_cpu8, execution_time_gpu)
        if best_time == execution_time_gpu
            println("\n🥇 WINNER: GPU ($(round(execution_time_gpu, digits=2))s)")
            advantage_gpu_vs_cpu8 = execution_time_cpu8 / execution_time_gpu
            advantage_gpu_vs_cpu1 = execution_time_cpu1 / execution_time_gpu
            println("   GPU advantage over CPU(8): $(round(advantage_gpu_vs_cpu8, digits=2))x")
            println("   GPU advantage over CPU(1): $(round(advantage_gpu_vs_cpu1, digits=2))x")
        elseif best_time == execution_time_cpu8
            println("\n🥇 WINNER: CPU($(Threads.nthreads())-threads) ($(round(execution_time_cpu8, digits=2))s)")
            advantage_cpu8_vs_gpu = execution_time_gpu / execution_time_cpu8
            advantage_cpu8_vs_cpu1 = execution_time_cpu1 / execution_time_cpu8
            println("   CPU(8) advantage over GPU: $(round(advantage_cpu8_vs_gpu, digits=2))x")
            println("   CPU(8) advantage over CPU(1): $(round(advantage_cpu8_vs_cpu1, digits=2))x")
        else
            println("\n🥇 WINNER: CPU(1-thread) ($(round(execution_time_cpu1, digits=2))s)")
        end
        
        # Threading efficiency with fair comparison
        threading_efficiency = speedup_cpu8_vs_cpu1 / Threads.nthreads() * 100
        println("\n🧵 Threading Efficiency Analysis:")
        println("   CPU(8) vs CPU(1) speedup: $(round(speedup_cpu8_vs_cpu1, digits=2))x")
        println("   Threading efficiency: $(round(threading_efficiency, digits=1))% ($(round(speedup_cpu8_vs_cpu1, digits=2))/$(Threads.nthreads()))")
        
        # Success rate comparison
        println("\n✅ Success Rate Comparison:")
        println("   CPU(1): $(round(result_cpu1.success_rate*100, digits=1))%")
        println("   CPU(8): $(round(result_cpu8.success_rate*100, digits=1))%")
        println("   GPU: $(round(result_gpu.success_rate*100, digits=1))%")
        
    end
    
    return results
end

function run_comprehensive_fair_comparison()
    println("🚀 COMPREHENSIVE FAIR 11-PARAMETER COMPARISON")
    println("🔧 All methods use identical BFGS optimization")
    println("📊 Testing multiple scales for thorough analysis")
    
    all_results = EstimatorTestResult[]
    
    # Test configurations
    test_configs = [
        ("Small", 25, 4, 50),      # Reduced for faster testing
        ("Medium", 50, 4, 100),    # Moderate scale
        ("Large", 100, 4, 150)     # Larger scale
    ]
    
    start_time = time()
    
    for (i, (scale, n_subjects, n_arms, n_trials)) in enumerate(test_configs)
        println("\n" * "🔄"^50)
        println("Running fair comparison test $i/$(length(test_configs)): $scale scale")
        
        scale_results = run_fair_11param_comparison(scale, n_subjects, n_arms, n_trials; seed=42+i*100)
        append!(all_results, scale_results)
        
        elapsed = time() - start_time
        avg_time_per_test = elapsed / i
        estimated_remaining = avg_time_per_test * (length(test_configs) - i)
        
        println("\n⏰ Progress: $i/$(length(test_configs)) tests completed")
        println("   Elapsed time: $(round(elapsed/60, digits=1)) minutes")
        println("   Estimated remaining: $(round(estimated_remaining/60, digits=1)) minutes")
    end
    
    total_time = time() - start_time
    println("\n🎉 ALL FAIR COMPARISON TESTS COMPLETED!")
    println("Total execution time: $(round(total_time/60, digits=1)) minutes")
    
    # Save results
    df = save_fair_comparison_results(all_results, "fair_11param_comparison_results.csv")
    
    # Create visualization
    create_fair_comparison_visualization(all_results)
    
    # Analysis
    analyze_fair_comparison_results(all_results)
    
    return all_results, df
end

function save_fair_comparison_results(results::Vector{EstimatorTestResult}, filename::String)
    df = DataFrame(
        scale = [r.scale for r in results],
        method = [r.method for r in results],
        n_subjects = [r.n_subjects for r in results],
        n_arms = [r.n_arms for r in results],
        n_trials = [r.n_trials for r in results],
        execution_time = [r.execution_time for r in results],
        memory_used = [r.memory_used for r in results],
        success_rate = [r.success_rate for r in results],
        throughput = [r.throughput for r in results],
        thread_count = [r.thread_count for r in results],
        timestamp = [r.timestamp for r in results]
    )
    
    # Add parameter correlations
    param_names = [:Q0, :α, :αm, :β, :αf, :μ, :τ, :φ, :C0, :η, :ν]
    for param_name in param_names
        col_name = Symbol("correlation_" * string(param_name))
        df[!, col_name] = [get(r.parameter_correlations, param_name, NaN) for r in results]
    end
    
    CSV.write(filename, df)
    println("📁 Fair comparison results saved to: $filename")
    return df
end

function create_fair_comparison_visualization(results::Vector{EstimatorTestResult})
    scales = unique([r.scale for r in results])
    methods = unique([r.method for r in results])
    
    fig = Figure(size = (1400, 1000))
    
    colors = Dict(
        "CPU(1-thread) BFGS" => :red,
        "CPU(8-threads) BFGS" => :blue,
        "GPU BFGS" => :green
    )
    
    # Execution Time Comparison
    ax1 = Axis(fig[1, 1], 
        title = "Fair Comparison: Execution Time (All use BFGS)", 
        xlabel = "Scale", 
        ylabel = "Execution Time (seconds)",
        yscale = log10)
    
    for method in methods
        method_results = filter(r -> r.method == method, results)
        if !isempty(method_results)
            times = [r.execution_time for r in method_results]
            color = get(colors, method, :black)
            lines!(ax1, 1:length(method_results), times, label = method, linewidth = 3, color = color)
            scatter!(ax1, 1:length(method_results), times, markersize = 12, color = color)
        end
    end
    axislegend(ax1, position = :lt)
    
    # Throughput Comparison
    ax2 = Axis(fig[1, 2], 
        title = "Fair Comparison: Throughput (All use BFGS)", 
        xlabel = "Scale", 
        ylabel = "Decisions/Second",
        yscale = log10)
    
    for method in methods
        method_results = filter(r -> r.method == method, results)
        if !isempty(method_results)
            throughputs = [r.throughput for r in method_results]
            color = get(colors, method, :black)
            lines!(ax2, 1:length(method_results), throughputs, label = method, linewidth = 3, color = color)
            scatter!(ax2, 1:length(method_results), throughputs, markersize = 12, color = color)
        end
    end
    axislegend(ax2, position = :lt)
    
    # Success Rate Comparison
    ax3 = Axis(fig[2, 1], 
        title = "Fair Comparison: Success Rate (All use BFGS)", 
        xlabel = "Scale", 
        ylabel = "Success Rate (%)")
    
    for method in methods
        method_results = filter(r -> r.method == method, results)
        if !isempty(method_results)
            success_rates = [r.success_rate * 100 for r in method_results]
            color = get(colors, method, :black)
            lines!(ax3, 1:length(method_results), success_rates, label = method, linewidth = 3, color = color)
            scatter!(ax3, 1:length(method_results), success_rates, markersize = 12, color = color)
        end
    end
    axislegend(ax3, position = :lb)
    
    # Memory Usage Comparison
    ax4 = Axis(fig[2, 2], 
        title = "Fair Comparison: Memory Usage (All use BFGS)", 
        xlabel = "Scale", 
        ylabel = "Memory Used (MB)")
    
    for method in methods
        method_results = filter(r -> r.method == method, results)
        if !isempty(method_results)
            memory = [r.memory_used for r in method_results]
            color = get(colors, method, :black)
            lines!(ax4, 1:length(method_results), memory, label = method, linewidth = 3, color = color)
            scatter!(ax4, 1:length(method_results), memory, markersize = 12, color = color)
        end
    end
    axislegend(ax4, position = :lt)
    
    save("fair_11param_comparison_results.png", fig)
    println("📊 Fair comparison visualization saved to: fair_11param_comparison_results.png")
    
    return fig
end

function analyze_fair_comparison_results(results::Vector{EstimatorTestResult})
    println("\n📊 FAIR COMPARISON ANALYSIS SUMMARY")
    println("🔧 All methods use identical BFGS optimization algorithm")
    println("="^80)
    
    scales = unique([r.scale for r in results])
    
    for scale in scales
        scale_results = filter(r -> r.scale == scale, results)
        
        println("\n📈 $scale Scale Analysis:")
        
        times = [r.execution_time for r in scale_results]
        methods = [r.method for r in scale_results]
        
        # Find best method
        fastest_idx = argmin(times)
        fastest_method = methods[fastest_idx]
        fastest_time = times[fastest_idx]
        
        println("   🥇 Winner: $fastest_method ($(round(fastest_time, digits=2))s)")
        
        # Calculate advantages
        for (i, (method, time)) in enumerate(zip(methods, times))
            if i != fastest_idx
                advantage = time / fastest_time
                println("   📈 $fastest_method vs $method: $(round(advantage, digits=2))x faster")
            end
        end
        
        # Threading efficiency
        cpu1_result = findfirst(r -> contains(r.method, "CPU(1-thread)"), scale_results)
        cpu8_result = findfirst(r -> contains(r.method, "CPU(8-threads)"), scale_results)
        
        if cpu1_result !== nothing && cpu8_result !== nothing
            cpu1_time = scale_results[cpu1_result].execution_time
            cpu8_time = scale_results[cpu8_result].execution_time
            speedup = cpu1_time / cpu8_time
            efficiency = speedup / 8 * 100
            println("   🧵 Threading efficiency: $(round(efficiency, digits=1))% ($(round(speedup, digits=2))x speedup)")
        end
    end
    
    println("\n🔍 KEY INSIGHTS FROM FAIR COMPARISON:")
    println("   • All methods now use identical BFGS optimization")
    println("   • Performance differences reflect true architectural advantages")
    println("   • No algorithmic bias favoring any particular method")
    println("   • Results show pure hardware/threading efficiency differences")
end

# Run the comprehensive fair comparison
if abspath(PROGRAM_FILE) == @__FILE__
    println("Starting fair 11-parameter efficiency comparison...")
    results, df = run_comprehensive_fair_comparison()
end

export run_fair_11param_comparison, run_comprehensive_fair_comparison