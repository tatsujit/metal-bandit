using Metal
using Statistics
using Random
using CSV
using DataFrames
using CairoMakie
using Dates

# Include the 11-parameter framework
include("estimator_11_parameter_test.jl")

"""
Comprehensive 11-Parameter Model Efficiency Testing

Tests efficiency across multiple scales to understand how parameter complexity 
affects the comparison between CPU(1), CPU(8), and GPU methods.
"""

function save_11param_results_to_csv(results::Vector{EstimatorTestResult}, filename::String)
    df = DataFrame(
        scale = [r.scale for r in results],
        method = [r.method for r in results],
        n_subjects = [r.n_subjects for r in results],
        n_arms = [r.n_arms for r in results],
        n_trials = [r.n_trials for r in results],
        execution_time = [r.execution_time for r in results],
        memory_used = [r.memory_used for r in results],
        compilation_time = [r.compilation_time for r in results],
        success_rate = [r.success_rate for r in results],
        throughput = [r.throughput for r in results],
        thread_count = [r.thread_count for r in results],
        timestamp = [r.timestamp for r in results]
    )
    
    # Add individual parameter correlations as separate columns
    param_names = [:Q0, :α, :αm, :β, :αf, :μ, :τ, :φ, :C0, :η, :ν]
    for param_name in param_names
        col_name = Symbol("correlation_" * string(param_name))
        df[!, col_name] = [get(r.parameter_correlations, param_name, NaN) for r in results]
    end
    
    CSV.write(filename, df)
    println("📁 11-parameter results saved to: $filename")
    return df
end

function create_11param_efficiency_visualization(results::Vector{EstimatorTestResult})
    # Prepare data
    scales = unique([r.scale for r in results])
    methods = unique([r.method for r in results])
    
    # Create figure
    fig = Figure(size = (1600, 1200))
    
    # Colors for consistency
    colors = Dict(
        "CPU(1-thread)" => :red,
        "CPU(8-threads)" => :blue,
        "GPU" => :green
    )
    
    # Plot 1: Execution Time Comparison
    ax1 = Axis(fig[1, 1], 
        title = "11-Parameter Model: Execution Time by Scale", 
        xlabel = "Scale", 
        ylabel = "Execution Time (seconds)",
        yscale = log10)
    
    for method in methods
        method_results = filter(r -> r.method == method, results)
        if !isempty(method_results)
            times = [r.execution_time for r in method_results]
            method_scales = [r.scale for r in method_results]
            color = get(colors, method, :black)
            lines!(ax1, 1:length(method_scales), times, label = method, linewidth = 3, color = color)
            scatter!(ax1, 1:length(method_scales), times, markersize = 12, color = color)
        end
    end
    axislegend(ax1, position = :lt)
    
    # Plot 2: Throughput Comparison
    ax2 = Axis(fig[1, 2], 
        title = "11-Parameter Model: Throughput by Scale", 
        xlabel = "Scale", 
        ylabel = "Decisions/Second",
        yscale = log10)
    
    for method in methods
        method_results = filter(r -> r.method == method, results)
        if !isempty(method_results)
            throughputs = [r.throughput for r in method_results]
            method_scales = [r.scale for r in method_results]
            color = get(colors, method, :black)
            lines!(ax2, 1:length(method_scales), throughputs, label = method, linewidth = 3, color = color)
            scatter!(ax2, 1:length(method_scales), throughputs, markersize = 12, color = color)
        end
    end
    axislegend(ax2, position = :lt)
    
    # Plot 3: Memory Usage
    ax3 = Axis(fig[2, 1], 
        title = "11-Parameter Model: Memory Usage by Scale", 
        xlabel = "Scale", 
        ylabel = "Memory Used (MB)")
    
    for method in methods
        method_results = filter(r -> r.method == method, results)
        if !isempty(method_results)
            memory = [r.memory_used for r in method_results]
            method_scales = [r.scale for r in method_results]
            color = get(colors, method, :black)
            lines!(ax3, 1:length(method_scales), memory, label = method, linewidth = 3, color = color)
            scatter!(ax3, 1:length(method_scales), memory, markersize = 12, color = color)
        end
    end
    axislegend(ax3, position = :lt)
    
    # Plot 4: Success Rate Comparison
    ax4 = Axis(fig[2, 2], 
        title = "11-Parameter Model: Success Rate by Scale", 
        xlabel = "Scale", 
        ylabel = "Success Rate (%)")
    
    for method in methods
        method_results = filter(r -> r.method == method, results)
        if !isempty(method_results)
            success_rates = [r.success_rate * 100 for r in method_results]
            method_scales = [r.scale for r in method_results]
            color = get(colors, method, :black)
            lines!(ax4, 1:length(method_scales), success_rates, label = method, linewidth = 3, color = color)
            scatter!(ax4, 1:length(method_scales), success_rates, markersize = 12, color = color)
        end
    end
    axislegend(ax4, position = :lb)
    
    # Save figure
    save("11_parameter_efficiency_results.png", fig)
    println("📊 11-parameter visualization saved to: 11_parameter_efficiency_results.png")
    
    return fig
end

function run_comprehensive_11param_efficiency_tests()
    println("🚀 COMPREHENSIVE 11-PARAMETER EFFICIENCY TEST SUITE")
    println("Testing CPU(1-thread) vs CPU(8-threads) vs GPU")
    println("Model: 11-parameter cognitive bandit model")
    println("Expected duration: 6-8 hours")
    println("\n" * "="^80)
    
    all_results = EstimatorTestResult[]
    
    # Test configurations (scaled for 11-parameter complexity)
    test_configs = [
        ("Small", 50, 4, 100),        # 20K decisions
        ("Medium", 150, 4, 150),      # 90K decisions  
        ("Large", 300, 4, 200),       # 240K decisions
        ("Extra-Large", 500, 4, 250)  # 500K decisions
    ]
    
    start_time = time()
    
    for (i, (scale, n_subjects, n_arms, n_trials)) in enumerate(test_configs)
        println("\n🔄 Running 11-parameter test $i/$(length(test_configs)): $scale scale")
        
        # Run scale test
        scale_results = run_11param_efficiency_test(scale, n_subjects, n_arms, n_trials)
        append!(all_results, scale_results)
        
        # Progress update
        elapsed = time() - start_time
        avg_time_per_test = elapsed / i
        estimated_remaining = avg_time_per_test * (length(test_configs) - i)
        
        println("\n⏰ Progress: $i/$(length(test_configs)) tests completed")
        println("   Elapsed time: $(round(elapsed/60, digits=1)) minutes")
        println("   Estimated remaining: $(round(estimated_remaining/60, digits=1)) minutes")
        
        # Save intermediate results
        save_11param_results_to_csv(all_results, "11param_efficiency_results_intermediate.csv")
    end
    
    total_time = time() - start_time
    println("\n🎉 ALL 11-PARAMETER TESTS COMPLETED!")
    println("Total execution time: $(round(total_time/60, digits=1)) minutes")
    
    # Save final results
    final_df = save_11param_results_to_csv(all_results, "11param_efficiency_results_final.csv")
    
    # Create comprehensive visualization
    create_11param_efficiency_visualization(all_results)
    
    # Analysis summary
    analyze_11param_results(all_results)
    
    return all_results, final_df
end

function analyze_11param_results(results::Vector{EstimatorTestResult})
    println("\n📊 11-PARAMETER EFFICIENCY ANALYSIS SUMMARY")
    println("="^80)
    
    scales = unique([r.scale for r in results])
    
    for scale in scales
        scale_results = filter(r -> r.scale == scale, results)
        
        println("\n📈 $scale Scale Analysis:")
        
        for result in scale_results
            println("   $(result.method):")
            println("     ⏱️  Time: $(round(result.execution_time, digits=2))s")
            println("     💾 Memory: $(round(result.memory_used, digits=1))MB")
            println("     ⚡ Throughput: $(round(result.throughput, digits=0)) decisions/s")
            println("     ✅ Success Rate: $(round(result.success_rate*100, digits=1))%")
            println("     🧵 Threads: $(result.thread_count)")
            
            # Show top parameter correlations
            if !isempty(result.parameter_correlations)
                correlations = filter(p -> !isnan(p.second), result.parameter_correlations)
                if !isempty(correlations)
                    sorted_corrs = sort(collect(correlations), by=x->abs(x.second), rev=true)
                    top_3 = take(sorted_corrs, min(3, length(sorted_corrs)))
                    println("     📊 Top parameter correlations: ", 
                           join(["$(k)=$(round(v, digits=3))" for (k,v) in top_3], ", "))
                end
            end
        end
        
        # Performance comparison for this scale
        if length(scale_results) >= 2
            times = [r.execution_time for r in scale_results]
            methods = [r.method for r in scale_results]
            
            fastest_idx = argmin(times)
            fastest_method = methods[fastest_idx]
            fastest_time = times[fastest_idx]
            
            println("\n   🏆 $scale Scale Winner: $fastest_method ($(round(fastest_time, digits=2))s)")
            
            for (i, (method, time)) in enumerate(zip(methods, times))
                if i != fastest_idx
                    speedup = time / fastest_time
                    println("     📈 $fastest_method vs $method: $(round(speedup, digits=2))x faster")
                end
            end
        end
    end
    
    println("\n🔍 CROSS-SCALE ANALYSIS:")
    
    # Threading efficiency analysis
    cpu1_results = filter(r -> r.method == "CPU(1-thread)", results)
    cpu8_results = filter(r -> r.method == "CPU(8-threads)", results)
    
    if !isempty(cpu1_results) && !isempty(cpu8_results)
        println("\n🧵 Threading Efficiency (CPU 8-thread vs 1-thread):")
        for (cpu1, cpu8) in zip(cpu1_results, cpu8_results)
            speedup = cpu1.execution_time / cpu8.execution_time
            efficiency = speedup / 8 * 100  # Theoretical max is 8x
            println("   $(cpu1.scale): $(round(speedup, digits=2))x speedup, $(round(efficiency, digits=1))% efficiency")
        end
    end
    
    # Parameter complexity impact
    println("\n🔢 Parameter Complexity Impact:")
    println("   • 11 parameters significantly increase computational complexity")
    println("   • Grid search becomes computationally intensive")
    println("   • Multi-threading benefits increase with problem complexity")
    
    # GPU performance analysis
    gpu_results = filter(r -> r.method == "GPU", results)
    if !isempty(gpu_results)
        println("\n🚀 GPU Performance Analysis:")
        for gpu_result in gpu_results
            println("   $(gpu_result.scale): Success rate $(round(gpu_result.success_rate*100, digits=1))%")
        end
        println("   • GPU uses simplified grid search for 11-parameter model")
        println("   • Complex parameter interactions challenge GPU implementation")
    end
end

# Run the comprehensive test suite
if abspath(PROGRAM_FILE) == @__FILE__
    println("Starting comprehensive 11-parameter efficiency testing...")
    results, df = run_comprehensive_11param_efficiency_tests()
end

export run_comprehensive_11param_efficiency_tests, save_11param_results_to_csv, analyze_11param_results