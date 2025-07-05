using Metal
using Statistics
using Random
using CSV
using DataFrames

# Include the updated estimator
include("estimator_11_parameter_test.jl")

"""
Medium Fair Comparison Test
Tests multiple scales to demonstrate the fair comparison results
"""

function medium_fair_test()
    println("🚀 MEDIUM FAIR 11-PARAMETER COMPARISON")
    println("🔧 All methods use identical BFGS optimization")
    
    # Test configurations
    test_configs = [
        ("Small", 20, 4, 50),
        ("Medium", 35, 4, 75),
        ("Large", 50, 4, 100)
    ]
    
    all_results = []
    
    for (scale, n_subjects, n_arms, n_trials) in test_configs
        println("\n" * "="^70)
        println("🔄 Testing $scale scale: $n_subjects subjects × $n_arms arms × $n_trials trials")
        println("📊 Total decisions: $(n_subjects * n_trials)")
        
        scale_results = []
        seed = 42
        
        # CPU Single-threaded
        println("\n1️⃣ CPU(1-thread) with BFGS...")
        time_cpu1 = @elapsed result_cpu1 = cpu_single_thread_11param_estimation(n_subjects, n_arms, n_trials; seed=seed)
        throughput_cpu1 = (n_subjects * n_trials) / time_cpu1
        println("   Time: $(round(time_cpu1, digits=2))s, Success: $(round(result_cpu1.success_rate*100, digits=1))%")
        push!(scale_results, ("CPU(1)", time_cpu1, result_cpu1.success_rate, throughput_cpu1))
        
        # CPU Multi-threaded  
        println("\n2️⃣ CPU(8-threads) with BFGS...")
        time_cpu8 = @elapsed result_cpu8 = cpu_multithread_11param_estimation(n_subjects, n_arms, n_trials; seed=seed)
        throughput_cpu8 = (n_subjects * n_trials) / time_cpu8
        println("   Time: $(round(time_cpu8, digits=2))s, Success: $(round(result_cpu8.success_rate*100, digits=1))%")
        push!(scale_results, ("CPU(8)", time_cpu8, result_cpu8.success_rate, throughput_cpu8))
        
        # GPU with BFGS
        println("\n3️⃣ GPU with BFGS (FAIR)...")
        time_gpu = @elapsed result_gpu = gpu_11param_estimation(n_subjects, n_arms, n_trials; seed=seed)
        throughput_gpu = (n_subjects * n_trials) / time_gpu
        println("   Time: $(round(time_gpu, digits=2))s, Success: $(round(result_gpu.success_rate*100, digits=1))%")
        push!(scale_results, ("GPU", time_gpu, result_gpu.success_rate, throughput_gpu))
        
        # Analysis for this scale
        println("\n📊 $scale Scale Results:")
        times = [r[2] for r in scale_results]
        methods = [r[1] for r in scale_results]
        
        fastest_idx = argmin(times)
        winner = methods[fastest_idx]
        winner_time = times[fastest_idx]
        
        println("🥇 Winner: $winner ($(round(winner_time, digits=2))s)")
        
        cpu1_time, cpu8_time, gpu_time = times[1], times[2], times[3]
        
        # Performance ratios
        cpu8_vs_cpu1 = cpu1_time / cpu8_time
        gpu_vs_cpu1 = cpu1_time / gpu_time
        gpu_vs_cpu8 = cpu8_time / gpu_time
        
        println("   CPU(8) vs CPU(1): $(round(cpu8_vs_cpu1, digits=2))x")
        println("   GPU vs CPU(1): $(round(gpu_vs_cpu1, digits=2))x")
        if gpu_vs_cpu8 > 1.0
            println("   GPU vs CPU(8): $(round(gpu_vs_cpu8, digits=2))x FASTER")
        else
            println("   CPU(8) vs GPU: $(round(1/gpu_vs_cpu8, digits=2))x FASTER")
        end
        
        # Threading efficiency
        threading_efficiency = cpu8_vs_cpu1 / 8 * 100
        println("   Threading efficiency: $(round(threading_efficiency, digits=1))%")
        
        # Store results with scale info
        for (method, time, success, throughput) in scale_results
            push!(all_results, (scale, method, time, success, throughput, n_subjects * n_trials))
        end
    end
    
    # Overall analysis
    println("\n" * "="^70)
    println("📊 COMPREHENSIVE FAIR COMPARISON ANALYSIS")
    println("="^70)
    
    # Group by scale and analyze trends
    for scale in ["Small", "Medium", "Large"]
        scale_data = filter(r -> r[1] == scale, all_results)
        if !isempty(scale_data)
            println("\n📈 $scale Scale Summary:")
            
            times = [r[3] for r in scale_data]
            methods = [r[2] for r in scale_data]
            
            fastest_idx = argmin(times)
            winner = methods[fastest_idx]
            
            println("   🥇 Winner: $winner")
            
            # Show all times
            for (_, method, time, success, throughput, total_decisions) in scale_data
                println("   $method: $(round(time, digits=2))s ($(round(success*100, digits=1))% success)")
            end
        end
    end
    
    # Key insights
    println("\n🔍 KEY INSIGHTS FROM FAIR COMPARISON:")
    
    # Check if GPU wins at any scale
    gpu_wins = 0
    cpu8_wins = 0
    cpu1_wins = 0
    
    for scale in ["Small", "Medium", "Large"]
        scale_data = filter(r -> r[1] == scale, all_results)
        if !isempty(scale_data)
            times = [r[3] for r in scale_data]
            methods = [r[2] for r in scale_data]
            fastest_idx = argmin(times)
            winner = methods[fastest_idx]
            
            if winner == "GPU"
                gpu_wins += 1
            elseif winner == "CPU(8)"
                cpu8_wins += 1
            else
                cpu1_wins += 1
            end
        end
    end
    
    println("   🏆 Wins across scales:")
    println("     GPU: $gpu_wins/3 scales")
    println("     CPU(8): $cpu8_wins/3 scales") 
    println("     CPU(1): $cpu1_wins/3 scales")
    
    if gpu_wins > 0
        println("\n   🚀 CRITICAL FINDING: GPU WINS when using fair BFGS optimization!")
        println("   🔧 Previous unfair comparison used grid search for GPU vs BFGS for CPU")
        println("   📊 Fair comparison reveals GPU competitive performance")
    else
        println("\n   🖥️  CPU methods dominate even with fair BFGS optimization")
        println("   🔧 GPU disadvantage not solely due to algorithm choice")
    end
    
    # Save results
    df = DataFrame(
        scale = [r[1] for r in all_results],
        method = [r[2] for r in all_results],
        execution_time = [r[3] for r in all_results],
        success_rate = [r[4] for r in all_results],
        throughput = [r[5] for r in all_results],
        total_decisions = [r[6] for r in all_results]
    )
    
    CSV.write("fair_comparison_results.csv", df)
    println("\n📁 Results saved to: fair_comparison_results.csv")
    
    return all_results, df
end

# Run medium test
results, df = medium_fair_test()