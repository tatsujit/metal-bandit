using Metal
using Statistics
using Random
using CSV
using DataFrames
using CairoMakie
using Dates

# Include the updated estimator
include("estimator_11_parameter_test.jl")

"""
Large-Scale Definitive Fair Comparison Test

This test runs much larger scales to eliminate measurement noise and provide
definitive conclusions about GPU vs CPU performance with identical BFGS optimization.

Time budget: 8 hours
Goal: Definitive performance comparison with statistical significance
"""

function design_large_scale_test_plan()
    println("🎯 LARGE-SCALE DEFINITIVE FAIR COMPARISON DESIGN")
    println("⏰ Time budget: 8 hours (28,800 seconds)")
    println("🔧 All methods use identical BFGS optimization")
    
    # Estimate execution times based on previous results
    # Small scale (50 subjects): ~10s per method
    # Medium scale (100 subjects): ~20s per method  
    # Large scale (200 subjects): ~40s per method
    # Very large scale (500 subjects): ~100s per method
    # Massive scale (1000 subjects): ~200s per method
    
    test_configs = [
        ("Large", 200, 4, 150),      # ~40s per method, 120s total
        ("Very-Large", 500, 4, 200), # ~100s per method, 300s total
        ("Massive", 1000, 4, 250),   # ~200s per method, 600s total
        ("Ultra", 1500, 4, 300),     # ~300s per method, 900s total
        ("Mega", 2000, 4, 350),      # ~400s per method, 1200s total
    ]
    
    total_estimated_time = 0
    for (scale, n_subjects, n_arms, n_trials) in test_configs
        decisions = n_subjects * n_trials
        estimated_time_per_method = n_subjects * 0.2  # Conservative estimate
        total_scale_time = estimated_time_per_method * 3  # 3 methods
        total_estimated_time += total_scale_time
        
        println("📊 $scale: $n_subjects subjects × $n_trials trials = $(decisions) decisions")
        println("   Estimated time: $(round(total_scale_time/60, digits=1)) minutes")
    end
    
    println("\n⏰ Total estimated time: $(round(total_estimated_time/3600, digits=1)) hours")
    
    if total_estimated_time > 28800
        println("⚠️  Estimated time exceeds 8-hour budget, adjusting...")
        # Reduce to fit budget
        test_configs = [
            ("Large", 150, 4, 120),      
            ("Very-Large", 300, 4, 150), 
            ("Massive", 600, 4, 200),   
            ("Ultra", 1000, 4, 250),     
        ]
        println("🔧 Adjusted test plan to fit 8-hour budget")
    end
    
    return test_configs
end

function run_single_large_scale_test(scale::String, n_subjects::Int, n_arms::Int, n_trials::Int; 
                                   seed::Int = 42, repetitions::Int = 3)
    println("\n" * "="^100)
    println("🔬 LARGE-SCALE DEFINITIVE TEST: $scale")
    println("📊 Configuration: $n_subjects subjects × $n_arms arms × $n_trials trials")
    println("🔢 Total decisions: $(n_subjects * n_trials)")
    println("🔄 Repetitions: $repetitions (for statistical significance)")
    println("🔧 ALL METHODS USE IDENTICAL BFGS OPTIMIZATION")
    println("="^100)
    
    all_results = []
    
    for rep in 1:repetitions
        println("\n🔄 Repetition $rep/$repetitions")
        rep_seed = seed + rep * 1000
        
        # CPU Single-threaded
        println("\n1️⃣ CPU(1-thread) with BFGS - Rep $rep...")
        GC.gc()
        memory_before = get_memory_usage()
        time_cpu1 = @elapsed result_cpu1 = cpu_single_thread_11param_estimation(n_subjects, n_arms, n_trials; seed=rep_seed)
        memory_after = get_memory_usage()
        memory_cpu1 = memory_after - memory_before
        throughput_cpu1 = (n_subjects * n_trials) / time_cpu1
        
        push!(all_results, (scale, rep, "CPU(1)", time_cpu1, result_cpu1.success_rate, throughput_cpu1, memory_cpu1))
        println("   Rep $rep: $(round(time_cpu1, digits=2))s, Success: $(round(result_cpu1.success_rate*100, digits=1))%")
        
        # CPU Multi-threaded  
        println("\n2️⃣ CPU(8-threads) with BFGS - Rep $rep...")
        GC.gc()
        memory_before = get_memory_usage()
        time_cpu8 = @elapsed result_cpu8 = cpu_multithread_11param_estimation(n_subjects, n_arms, n_trials; seed=rep_seed)
        memory_after = get_memory_usage()
        memory_cpu8 = memory_after - memory_before
        throughput_cpu8 = (n_subjects * n_trials) / time_cpu8
        
        push!(all_results, (scale, rep, "CPU(8)", time_cpu8, result_cpu8.success_rate, throughput_cpu8, memory_cpu8))
        println("   Rep $rep: $(round(time_cpu8, digits=2))s, Success: $(round(result_cpu8.success_rate*100, digits=1))%")
        
        # GPU with BFGS
        println("\n3️⃣ GPU with BFGS - Rep $rep...")
        GC.gc()
        memory_before = get_memory_usage()
        time_gpu = @elapsed result_gpu = gpu_11param_estimation(n_subjects, n_arms, n_trials; seed=rep_seed)
        memory_after = get_memory_usage()
        memory_gpu = memory_after - memory_before
        throughput_gpu = (n_subjects * n_trials) / time_gpu
        
        push!(all_results, (scale, rep, "GPU", time_gpu, result_gpu.success_rate, throughput_gpu, memory_gpu))
        println("   Rep $rep: $(round(time_gpu, digits=2))s, Success: $(round(result_gpu.success_rate*100, digits=1))%")
        
        # Quick analysis for this repetition
        println("\n📊 Repetition $rep Summary:")
        cpu8_vs_gpu = time_gpu / time_cpu8
        if cpu8_vs_gpu < 1.0
            println("   🚀 GPU faster: $(round(1/cpu8_vs_gpu, digits=3))x advantage")
        else
            println("   🖥️  CPU(8) faster: $(round(cpu8_vs_gpu, digits=3))x advantage")
        end
    end
    
    return all_results
end

function analyze_large_scale_statistical_significance(results)
    println("\n📊 STATISTICAL SIGNIFICANCE ANALYSIS")
    println("="^80)
    
    scales = unique([r[1] for r in results])
    
    for scale in scales
        scale_results = filter(r -> r[1] == scale, results)
        
        # Group by method
        cpu1_times = [r[4] for r in scale_results if r[3] == "CPU(1)"]
        cpu8_times = [r[4] for r in scale_results if r[3] == "CPU(8)"]
        gpu_times = [r[4] for r in scale_results if r[3] == "GPU"]
        
        println("\n📈 $scale Scale Statistical Analysis:")
        
        # Calculate means and standard deviations
        cpu1_mean = mean(cpu1_times)
        cpu1_std = std(cpu1_times)
        cpu8_mean = mean(cpu8_times)
        cpu8_std = std(cpu8_times)
        gpu_mean = mean(gpu_times)
        gpu_std = std(gpu_times)
        
        println("   CPU(1): $(round(cpu1_mean, digits=2))s ± $(round(cpu1_std, digits=2))s")
        println("   CPU(8): $(round(cpu8_mean, digits=2))s ± $(round(cpu8_std, digits=2))s")
        println("   GPU:    $(round(gpu_mean, digits=2))s ± $(round(gpu_std, digits=2))s")
        
        # Performance ratios with confidence
        cpu8_vs_gpu_ratio = gpu_mean / cpu8_mean
        gpu_vs_cpu8_ratio = cpu8_mean / gpu_mean
        
        # Calculate coefficient of variation for reliability
        cpu8_cv = cpu8_std / cpu8_mean * 100
        gpu_cv = gpu_std / gpu_mean * 100
        
        println("\n   📊 Performance Analysis:")
        if gpu_mean < cpu8_mean
            advantage = gpu_vs_cpu8_ratio
            println("   🚀 GPU WINNER: $(round(advantage, digits=3))x faster than CPU(8)")
            println("   🎯 GPU advantage: $(round((advantage-1)*100, digits=1))%")
        else
            advantage = cpu8_vs_gpu_ratio  
            println("   🖥️  CPU(8) WINNER: $(round(advantage, digits=3))x faster than GPU")
            println("   🎯 CPU(8) advantage: $(round((advantage-1)*100, digits=1))%")
        end
        
        println("   📈 Measurement reliability:")
        println("     CPU(8) CV: $(round(cpu8_cv, digits=1))% ($(cpu8_cv < 5 ? "excellent" : cpu8_cv < 10 ? "good" : "fair"))")
        println("     GPU CV: $(round(gpu_cv, digits=1))% ($(gpu_cv < 5 ? "excellent" : gpu_cv < 10 ? "good" : "fair"))")
        
        # Statistical significance test (simplified)
        difference_mean = abs(gpu_mean - cpu8_mean)
        pooled_std = sqrt((cpu8_std^2 + gpu_std^2) / 2)
        effect_size = difference_mean / pooled_std
        
        if effect_size > 0.8
            significance = "Large effect"
        elseif effect_size > 0.5
            significance = "Medium effect"
        elseif effect_size > 0.2
            significance = "Small effect"
        else
            significance = "Negligible effect"
        end
        
        println("   🔬 Effect size: $(round(effect_size, digits=2)) ($significance)")
        
        # Memory analysis
        cpu8_memory = [r[7] for r in scale_results if r[3] == "CPU(8)"]
        gpu_memory = [r[7] for r in scale_results if r[3] == "GPU"]
        
        if !isempty(cpu8_memory) && !isempty(gpu_memory)
            cpu8_mem_mean = mean(cpu8_memory)
            gpu_mem_mean = mean(gpu_memory)
            memory_advantage = cpu8_mem_mean / gpu_mem_mean
            
            println("   💾 Memory usage:")
            println("     CPU(8): $(round(cpu8_mem_mean, digits=1))MB")
            println("     GPU: $(round(gpu_mem_mean, digits=1))MB")
            println("     GPU memory advantage: $(round(memory_advantage, digits=2))x less usage")
        end
    end
end

function run_comprehensive_large_scale_comparison()
    println("🚀 COMPREHENSIVE LARGE-SCALE DEFINITIVE COMPARISON")
    println("⏰ Time budget: 8 hours")
    println("🎯 Goal: Eliminate measurement noise, achieve statistical significance")
    
    # Design test plan
    test_configs = design_large_scale_test_plan()
    
    all_results = []
    start_time = time()
    
    for (i, (scale, n_subjects, n_arms, n_trials)) in enumerate(test_configs)
        scale_start = time()
        
        println("\n" * "🔄"^60)
        println("Running large-scale test $i/$(length(test_configs)): $scale")
        
        # Run with multiple repetitions for statistical significance
        scale_results = run_single_large_scale_test(scale, n_subjects, n_arms, n_trials; 
                                                  seed=42+i*1000, repetitions=3)
        append!(all_results, scale_results)
        
        scale_elapsed = time() - scale_start
        total_elapsed = time() - start_time
        remaining_tests = length(test_configs) - i
        avg_time_per_test = total_elapsed / i
        estimated_remaining = avg_time_per_test * remaining_tests
        
        println("\n⏰ Progress Update:")
        println("   Scale completed in: $(round(scale_elapsed/60, digits=1)) minutes")
        println("   Total elapsed: $(round(total_elapsed/3600, digits=2)) hours")
        println("   Estimated remaining: $(round(estimated_remaining/3600, digits=2)) hours")
        println("   Tests completed: $i/$(length(test_configs))")
        
        # Save intermediate results
        save_large_scale_results(all_results, "large_scale_results_intermediate.csv")
        
        # Check if we're exceeding time budget
        if total_elapsed + estimated_remaining > 28800  # 8 hours
            println("⚠️  Approaching time budget limit, may stop early")
        end
    end
    
    total_time = time() - start_time
    println("\n🎉 LARGE-SCALE TESTING COMPLETED!")
    println("Total execution time: $(round(total_time/3600, digits=2)) hours")
    
    # Save final results
    final_df = save_large_scale_results(all_results, "large_scale_definitive_results.csv")
    
    # Statistical analysis
    analyze_large_scale_statistical_significance(all_results)
    
    # Create visualization
    create_large_scale_visualization(all_results)
    
    return all_results, final_df
end

function save_large_scale_results(results, filename::String)
    df = DataFrame(
        scale = [r[1] for r in results],
        repetition = [r[2] for r in results],
        method = [r[3] for r in results],
        execution_time = [r[4] for r in results],
        success_rate = [r[5] for r in results],
        throughput = [r[6] for r in results],
        memory_used = [r[7] for r in results],
        timestamp = [string(now()) for r in results]
    )
    
    CSV.write(filename, df)
    println("📁 Large-scale results saved to: $filename")
    return df
end

function create_large_scale_visualization(results)
    scales = unique([r[1] for r in results])
    methods = unique([r[3] for r in results])
    
    fig = Figure(size = (1600, 1200))
    
    colors = Dict(
        "CPU(1)" => :red,
        "CPU(8)" => :blue,
        "GPU" => :green
    )
    
    # Mean execution times with error bars
    ax1 = Axis(fig[1, 1], 
        title = "Large-Scale Fair Comparison: Execution Time (±std)", 
        xlabel = "Scale", 
        ylabel = "Execution Time (seconds)",
        yscale = log10)
    
    x_pos = 1
    for scale in scales
        scale_results = filter(r -> r[1] == scale, results)
        
        for method in methods
            method_times = [r[4] for r in scale_results if r[3] == method]
            if !isempty(method_times)
                mean_time = mean(method_times)
                std_time = std(method_times)
                color = get(colors, method, :black)
                
                scatter!(ax1, [x_pos], [mean_time], markersize = 15, color = color, label = method)
                errorbars!(ax1, [x_pos], [mean_time], [std_time], color = color, linewidth = 2)
                x_pos += 0.3
            end
        end
        x_pos += 1
    end
    axislegend(ax1, position = :lt)
    
    save("large_scale_definitive_comparison.png", fig)
    println("📊 Large-scale visualization saved to: large_scale_definitive_comparison.png")
    
    return fig
end

# Run the comprehensive large-scale test
if abspath(PROGRAM_FILE) == @__FILE__
    println("Starting large-scale definitive fair comparison...")
    results, df = run_comprehensive_large_scale_comparison()
end

export run_comprehensive_large_scale_comparison, analyze_large_scale_statistical_significance