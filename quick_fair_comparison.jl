using Metal
using Statistics
using Random

# Include the updated estimator
include("estimator_11_parameter_test.jl")

"""
Quick Fair Comparison Test
Tests small scale to quickly demonstrate the fair comparison results
"""

function quick_fair_test()
    println("🚀 QUICK FAIR 11-PARAMETER COMPARISON")
    println("🔧 All methods use identical BFGS optimization")
    
    # Small test for quick results
    n_subjects = 15
    n_arms = 4  
    n_trials = 40
    seed = 42
    
    println("\n📊 Configuration: $n_subjects subjects × $n_arms arms × $n_trials trials")
    println("🔢 Total decisions: $(n_subjects * n_trials)")
    
    results = []
    
    # CPU Single-threaded
    println("\n1️⃣ CPU(1-thread) with BFGS...")
    time_cpu1 = @elapsed result_cpu1 = cpu_single_thread_11param_estimation(n_subjects, n_arms, n_trials; seed=seed)
    throughput_cpu1 = (n_subjects * n_trials) / time_cpu1
    println("   Time: $(round(time_cpu1, digits=2))s, Success: $(round(result_cpu1.success_rate*100, digits=1))%")
    push!(results, ("CPU(1)", time_cpu1, result_cpu1.success_rate, throughput_cpu1))
    
    # CPU Multi-threaded  
    println("\n2️⃣ CPU(8-threads) with BFGS...")
    time_cpu8 = @elapsed result_cpu8 = cpu_multithread_11param_estimation(n_subjects, n_arms, n_trials; seed=seed)
    throughput_cpu8 = (n_subjects * n_trials) / time_cpu8
    println("   Time: $(round(time_cpu8, digits=2))s, Success: $(round(result_cpu8.success_rate*100, digits=1))%")
    push!(results, ("CPU(8)", time_cpu8, result_cpu8.success_rate, throughput_cpu8))
    
    # GPU with BFGS
    println("\n3️⃣ GPU with BFGS (FAIR)...")
    time_gpu = @elapsed result_gpu = gpu_11param_estimation(n_subjects, n_arms, n_trials; seed=seed)
    throughput_gpu = (n_subjects * n_trials) / time_gpu
    println("   Time: $(round(time_gpu, digits=2))s, Success: $(round(result_gpu.success_rate*100, digits=1))%")
    push!(results, ("GPU", time_gpu, result_gpu.success_rate, throughput_gpu))
    
    # Analysis
    println("\n📊 FAIR COMPARISON RESULTS:")
    println("="^60)
    
    times = [r[2] for r in results]
    methods = [r[1] for r in results]
    success_rates = [r[3] for r in results]
    throughputs = [r[4] for r in results]
    
    # Find winner
    fastest_idx = argmin(times)
    winner = methods[fastest_idx]
    winner_time = times[fastest_idx]
    
    println("🥇 WINNER: $winner ($(round(winner_time, digits=2))s)")
    
    # Performance ratios
    cpu1_time, cpu8_time, gpu_time = times[1], times[2], times[3]
    
    println("\n🏆 Performance Ratios:")
    println("   CPU(8) vs CPU(1): $(round(cpu1_time/cpu8_time, digits=2))x speedup")
    println("   GPU vs CPU(1): $(round(cpu1_time/gpu_time, digits=2))x speedup") 
    println("   GPU vs CPU(8): $(round(cpu8_time/gpu_time, digits=2))x speedup")
    
    # Threading efficiency
    threading_efficiency = (cpu1_time/cpu8_time) / 8 * 100
    println("   Threading efficiency: $(round(threading_efficiency, digits=1))%")
    
    println("\n✅ Success Rates:")
    for (method, _, success, _) in results
        println("   $method: $(round(success*100, digits=1))%")
    end
    
    println("\n⚡ Throughput (decisions/sec):")
    for (method, _, _, throughput) in results
        println("   $method: $(round(throughput, digits=0))")
    end
    
    # Key insights
    println("\n🔍 KEY INSIGHTS:")
    if gpu_time < cpu8_time
        advantage = cpu8_time / gpu_time
        println("   🚀 GPU WINS when using fair BFGS optimization!")
        println("   🚀 GPU is $(round(advantage, digits=2))x faster than CPU(8)")
        println("   🔧 Previous unfair comparison used grid search for GPU")
    else
        advantage = gpu_time / cpu8_time  
        println("   🖥️  CPU(8) wins with $(round(advantage, digits=2))x advantage")
    end
    
    return results
end

# Run quick test
results = quick_fair_test()