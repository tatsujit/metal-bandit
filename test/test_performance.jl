using Test
using Metal
using Statistics
using BenchmarkTools
using Random

# Add the parent directory to the path
push!(LOAD_PATH, "..")
include("../metal_bandit_simulator.jl")
include("../metal-bandit.jl")  # Include original file for generate_test_data function

@testset "Performance Tests" begin
    
    @testset "Metal Availability" begin
        @test Metal.functional() == true
        if Metal.functional()
            println("Metal Device: Available")
            # Note: Some Metal functions may not be accessible in all environments
        end
    end
    
    @testset "GPU Memory Allocation Performance" begin
        # Test GPU memory allocation performance
        sizes = [(100, 100), (500, 500), (1000, 1000)]
        
        for (rows, cols) in sizes
            # Test Float32 allocation
            time_float32 = @elapsed begin
                arr = Metal.zeros(Float32, rows, cols)
                Metal.synchronize()
            end
            
            # Test Int32 allocation
            time_int32 = @elapsed begin
                arr = Metal.zeros(Int32, rows, cols)
                Metal.synchronize()
            end
            
            @test time_float32 < 1.0  # Should be fast
            @test time_int32 < 1.0    # Should be fast
            
            println("Allocation ($rows×$cols): Float32=$(round(time_float32*1000, digits=2))ms, Int32=$(round(time_int32*1000, digits=2))ms")
        end
    end
    
    @testset "Simulation Performance Scaling" begin
        # Test how performance scales with problem size
        base_config = (n_arms=5, n_trials=100, n_agents=50)
        
        # Scale by agents
        agent_scales = [1, 2, 4, 8]
        agent_times = Float64[]
        
        for scale in agent_scales
            n_agents = base_config.n_agents * scale
            env = MetalBernoulliEnvironment(base_config.n_arms, base_config.n_trials, n_agents)
            agent = MetalQLearningAgent(base_config.n_arms, n_agents, base_config.n_trials)
            
            time = @elapsed begin
                run_metal_bandit_simulation!(env, agent)
                Metal.synchronize()
            end
            
            push!(agent_times, time)
            @test time < 10.0  # Should complete within 10 seconds
        end
        
        # Check that scaling is reasonable (not exponential)
        @test agent_times[end] < agent_times[1] * 20  # Should not be more than 20x slower for 8x agents
        
        println("Agent scaling times: $(round.(agent_times .* 1000, digits=2)) ms")
    end
    
    @testset "MLE Performance Comparison" begin
        # Compare batch vs non-batch MLE performance
        n_arms, n_trials, n_agents = 8, 500, 200
        
        env = MetalBernoulliEnvironment(n_arms, n_trials, n_agents)
        agent = MetalQLearningAgent(n_arms, n_agents, n_trials)
        run_metal_bandit_simulation!(env, agent)
        
        # Time batch processing
        time_batch = @elapsed begin
            est_batch = gpu_mle_parameter_estimation(env, agent; use_batch_processing=true)
            Metal.synchronize()
        end
        
        # Time non-batch processing
        time_nonbatch = @elapsed begin
            est_nonbatch = gpu_mle_parameter_estimation(env, agent; use_batch_processing=false)
            Metal.synchronize()
        end
        
        @test time_batch < 5.0
        @test time_nonbatch < 5.0
        
        # Results should be similar
        @test mean(abs.(est_batch - est_nonbatch)) < 0.1
        
        println("MLE Performance - Batch: $(round(time_batch*1000, digits=2))ms, Non-batch: $(round(time_nonbatch*1000, digits=2))ms")
    end
    
    @testset "Kernel Launch Overhead" begin
        # Test kernel launch overhead
        n_arms, n_agents = 5, 100
        q_values = Metal.rand(Float32, n_arms, n_agents)
        beta = Metal.fill(2.0f0, n_arms, n_agents)
        actions = Metal.zeros(Int32, 1, n_agents)
        random_vals = Metal.rand(Float32, n_agents)
        
        # Compile kernel first
        kernel = Metal.@metal launch=false softmax_action_selection_kernel!(
            actions, q_values, beta, random_vals, n_arms, n_agents, 1
        )
        kernel(actions, q_values, beta, random_vals, n_arms, n_agents, 1;
               threads=n_agents, groups=1)
        Metal.synchronize()
        
        # Time multiple kernel launches
        n_launches = 100
        time_launches = @elapsed begin
            for i in 1:n_launches
                kernel(actions, q_values, beta, random_vals, n_arms, n_agents, 1;
                       threads=n_agents, groups=1)
            end
            Metal.synchronize()
        end
        
        avg_launch_time = time_launches / n_launches
        @test avg_launch_time < 0.001  # Should be less than 1ms per launch
        
        println("Average kernel launch time: $(round(avg_launch_time*1000000, digits=2))μs")
    end
    
    @testset "Memory Bandwidth Test" begin
        # Test memory bandwidth utilization
        sizes = [1024, 2048, 4096, 8192]
        
        for size in sizes
            # Create large arrays
            src = Metal.rand(Float32, size, size)
            dst = Metal.zeros(Float32, size, size)
            
            # Simple copy operation
            time_copy = @elapsed begin
                dst .= src
                Metal.synchronize()
            end
            
            bytes_transferred = size * size * sizeof(Float32) * 2  # Read + Write
            bandwidth_gb_s = bytes_transferred / time_copy / 1e9
            
            @test bandwidth_gb_s > 0.01  # Should achieve at least 0.01 GB/s (very conservative)
            
            println("Memory bandwidth ($size×$size): $(round(bandwidth_gb_s, digits=2)) GB/s")
        end
    end
    
    @testset "Throughput Measurement" begin
        # Measure overall throughput
        configurations = [
            (n_arms=5, n_trials=200, n_agents=100),
            (n_arms=10, n_trials=500, n_agents=200),
            (n_arms=8, n_trials=1000, n_agents=500)
        ]
        
        for config in configurations
            env = MetalBernoulliEnvironment(config.n_arms, config.n_trials, config.n_agents)
            agent = MetalQLearningAgent(config.n_arms, config.n_agents, config.n_trials)
            
            total_time = @elapsed begin
                run_metal_bandit_simulation!(env, agent)
                estimated_params = gpu_mle_parameter_estimation(env, agent)
                Metal.synchronize()
            end
            
            total_operations = config.n_arms * config.n_agents * config.n_trials
            throughput = total_operations / total_time
            
            @test throughput > 1000  # Should process at least 1000 operations per second
            
            println("Throughput $(config): $(round(throughput, digits=0)) ops/sec")
        end
    end
    
    @testset "GPU vs CPU Comparison" begin
        # Compare GPU vs CPU performance for MLE (using original CPU implementation)
        n_arms, n_trials, n_agents = 5, 200, 100
        
        # Generate test data
        true_params, observations, rewards = generate_test_data(n_agents, n_arms, n_trials)
        
        # CPU MLE time
        cpu_time = @elapsed cpu_result = mle_estimate_cpu(observations, rewards)
        
        # GPU MLE time
        env = MetalBernoulliEnvironment(n_arms, n_trials, n_agents)
        agent = MetalQLearningAgent(n_arms, n_agents, n_trials)
        run_metal_bandit_simulation!(env, agent)
        
        gpu_time = @elapsed gpu_result = gpu_mle_parameter_estimation(env, agent)
        
        speedup = cpu_time / gpu_time
        
        # GPU should be faster (though on small problems CPU might be competitive)
        @test speedup > 0.1  # At least not much slower
        
        println("CPU vs GPU MLE - CPU: $(round(cpu_time*1000, digits=2))ms, GPU: $(round(gpu_time*1000, digits=2))ms, Speedup: $(round(speedup, digits=2))x")
    end
    
    @testset "Stress Test" begin
        # Stress test with larger problem size
        if Metal.functional()
            n_arms, n_trials, n_agents = 15, 2000, 500
            
            env = MetalBernoulliEnvironment(n_arms, n_trials, n_agents)
            agent = MetalQLearningAgent(n_arms, n_agents, n_trials)
            
            # This should complete without errors
            @test_nowarn begin
                simulation_time = @elapsed run_metal_bandit_simulation!(env, agent)
                mle_time = @elapsed estimated_params = gpu_mle_parameter_estimation(env, agent)
                total_time = simulation_time + mle_time
                
                @test total_time < 30.0  # Should complete within 30 seconds
                @test size(estimated_params) == (n_arms, n_agents)
                @test all(0.0 .<= estimated_params .<= 1.0)
                
                println("Stress test completed - Simulation: $(round(simulation_time, digits=2))s, MLE: $(round(mle_time, digits=2))s")
            end
        end
    end
end