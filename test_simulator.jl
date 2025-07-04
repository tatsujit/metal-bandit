#!/usr/bin/env julia

# Test script for the Metal Bandit Simulator
include("metal_bandit_simulator.jl")

function main()
    println("Testing Metal Bandit Simulator...")
    
    # Run the main demonstration
    result = demonstrate_metal_bandit_simulator()
    
    if result !== nothing
        println("\n✅ Demo completed successfully!")
        
        # Run a quick benchmark
        println("\n🏎️  Running performance benchmark...")
        benchmark_results = benchmark_metal_bandit_simulator([5], [100, 500], [1000])
        
        println("\n📊 Benchmark Results:")
        for (config, metrics) in benchmark_results
            n_arms, n_agents, n_trials = config
            println("  $n_arms arms, $n_agents agents, $n_trials trials:")
            println("    Total time: $(round(metrics.total_time, digits=3))s")
            println("    Throughput: $(round(metrics.throughput, digits=0)) ops/s")
            println("    Recovery R²: $(round(metrics.recovery_metrics.r_squared, digits=4))")
        end
    else
        println("❌ Demo failed - Metal not available")
    end
end

# Run if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end