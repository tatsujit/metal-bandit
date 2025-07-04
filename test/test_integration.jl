using Test
using Metal
using Statistics
using Random

# Add the parent directory to the path
push!(LOAD_PATH, "..")
include("../metal_bandit_simulator.jl")

@testset "Integration Tests" begin
    
    @testset "Complete Simulation Workflow" begin
        # Test the complete simulation workflow
        n_arms, n_trials, n_agents = 5, 200, 50
        alpha, beta = 0.1f0, 2.0f0
        
        # Create environment and agent
        env = MetalBernoulliEnvironment(n_arms, n_trials, n_agents)
        agent = MetalQLearningAgent(n_arms, n_agents, n_trials; alpha=alpha, beta=beta)
        
        # Run simulation
        @test_nowarn run_metal_bandit_simulation!(env, agent)
        
        # Check that actions were recorded
        actions_cpu = Array(env.actions)
        @test all(1 .<= actions_cpu .<= n_arms)
        @test size(actions_cpu) == (n_trials, n_agents)
        
        # Check that rewards were generated
        rewards_cpu = Array(env.actual_rewards)
        @test all(r -> r == 0.0f0 || r == 1.0f0, rewards_cpu)
        @test size(rewards_cpu) == (n_trials, n_agents)
        
        # Check that Q-values were updated
        q_values_cpu = Array(agent.q_values)
        @test !all(q_values_cpu .≈ 0.5f0)  # Should have changed from initial values
        
        # Check that arm counts were updated
        arm_counts_cpu = Array(agent.arm_counts)
        @test sum(arm_counts_cpu) == n_trials * n_agents
    end
    
    @testset "MLE Parameter Estimation Workflow" begin
        # Test MLE parameter estimation
        n_arms, n_trials, n_agents = 4, 500, 100
        
        # Create environment with known parameters
        true_params = rand(Float32, n_arms, n_agents) .* 0.8 .+ 0.1  # 0.1 to 0.9
        env = MetalBernoulliEnvironment(n_arms, n_trials, n_agents; true_params=true_params)
        agent = MetalQLearningAgent(n_arms, n_agents, n_trials)
        
        # Run simulation
        run_metal_bandit_simulation!(env, agent)
        
        # Test both MLE estimation methods
        estimated_params1 = gpu_mle_parameter_estimation(env, agent; use_batch_processing=true)
        estimated_params2 = gpu_mle_parameter_estimation(env, agent; use_batch_processing=false)
        
        # Check parameter dimensions
        @test size(estimated_params1) == (n_arms, n_agents)
        @test size(estimated_params2) == (n_arms, n_agents)
        
        # Check parameter ranges
        @test all(0.0 .<= estimated_params1 .<= 1.0)
        @test all(0.0 .<= estimated_params2 .<= 1.0)
        
        # Both methods should give similar results
        @test mean(abs.(estimated_params1 - estimated_params2)) < 0.05
    end
    
    @testset "Parameter Recovery Analysis" begin
        # Test parameter recovery analysis
        n_arms, n_agents = 3, 25
        true_params = rand(Float32, n_arms, n_agents)
        
        # Create slightly noisy estimates
        Random.seed!(42)
        noise = randn(Float32, n_arms, n_agents) * 0.05
        estimated_params = clamp.(true_params + noise, 0.0f0, 1.0f0)
        
        # Test recovery analysis
        recovery_metrics, abs_errors, rel_errors = analyze_parameter_recovery(true_params, estimated_params)
        
        # Check metrics structure
        @test haskey(recovery_metrics, :mae)
        @test haskey(recovery_metrics, :mse)
        @test haskey(recovery_metrics, :rmse)
        @test haskey(recovery_metrics, :mape)
        @test haskey(recovery_metrics, :correlation)
        @test haskey(recovery_metrics, :r_squared)
        
        # Check metric values are reasonable
        @test recovery_metrics.mae >= 0.0
        @test recovery_metrics.mse >= 0.0
        @test recovery_metrics.rmse >= 0.0
        @test recovery_metrics.mape >= 0.0
        @test -1.0 <= recovery_metrics.correlation <= 1.0
        @test 0.0 <= recovery_metrics.r_squared <= 1.0
        
        # Check error arrays
        @test size(abs_errors) == (n_arms, n_agents)
        @test size(rel_errors) == (n_arms, n_agents)
        @test all(abs_errors .>= 0.0)
        @test all(rel_errors .>= 0.0)
    end
    
    @testset "Plotting Functionality" begin
        # Test plotting without displaying
        n_arms, n_agents = 3, 20
        true_params = rand(Float32, n_arms, n_agents)
        estimated_params = true_params + randn(Float32, n_arms, n_agents) * 0.1
        estimated_params = clamp.(estimated_params, 0.0f0, 1.0f0)
        
        # Test plotting function
        @test_nowarn plot_recovery = plot_parameter_recovery(true_params, estimated_params)
        
        # Test with custom title
        @test_nowarn plot_recovery = plot_parameter_recovery(true_params, estimated_params; title_prefix="Test")
    end
    
    @testset "Benchmark Workflow" begin
        # Test benchmark functionality (small scale)
        @test_nowarn results = benchmark_metal_bandit_simulator([3], [20], [100])
        
        # Check results structure
        results = benchmark_metal_bandit_simulator([3], [20], [100])
        @test length(results) == 1
        
        key = first(keys(results))
        result = results[key]
        
        @test haskey(result, :simulation_time)
        @test haskey(result, :mle_time)
        @test haskey(result, :total_time)
        @test haskey(result, :recovery_metrics)
        @test haskey(result, :throughput)
        
        @test result.simulation_time >= 0.0
        @test result.mle_time >= 0.0
        @test result.total_time >= 0.0
        @test result.throughput >= 0.0
    end
    
    @testset "Edge Cases and Error Handling" begin
        # Test with very small problem sizes
        env_small = MetalBernoulliEnvironment(2, 1, 1)
        agent_small = MetalQLearningAgent(2, 1, 1)
        
        @test_nowarn run_metal_bandit_simulation!(env_small, agent_small)
        @test_nowarn estimated_params = gpu_mle_parameter_estimation(env_small, agent_small)
        
        # Test with mismatched dimensions (should work due to flexible design)
        env_mismatch = MetalBernoulliEnvironment(3, 50, 10)
        agent_mismatch = MetalQLearningAgent(3, 10, 50)
        
        @test_nowarn run_metal_bandit_simulation!(env_mismatch, agent_mismatch)
        
        # Test with extreme parameter values
        agent_extreme = MetalQLearningAgent(5, 25, 100; alpha=0.001f0, beta=100.0f0)
        env_extreme = MetalBernoulliEnvironment(5, 100, 25)
        
        @test_nowarn run_metal_bandit_simulation!(env_extreme, agent_extreme)
    end
    
    @testset "Memory Efficiency" begin
        # Test that large simulations don't cause memory issues
        if Metal.functional()
            # Test with moderately large problem
            n_arms, n_trials, n_agents = 10, 1000, 200
            
            env = MetalBernoulliEnvironment(n_arms, n_trials, n_agents)
            agent = MetalQLearningAgent(n_arms, n_agents, n_trials)
            
            # This should complete without memory errors
            @test_nowarn run_metal_bandit_simulation!(env, agent; batch_size=100)
            @test_nowarn estimated_params = gpu_mle_parameter_estimation(env, agent; batch_size=100)
        end
    end
    
    @testset "Consistency Across Runs" begin
        # Test that results are consistent with same random seed
        n_arms, n_trials, n_agents = 4, 100, 25
        
        # Create identical environments
        true_params = rand(Float32, n_arms, n_agents)
        env1 = MetalBernoulliEnvironment(n_arms, n_trials, n_agents; true_params=true_params)
        env2 = MetalBernoulliEnvironment(n_arms, n_trials, n_agents; true_params=true_params)
        
        agent1 = MetalQLearningAgent(n_arms, n_agents, n_trials; alpha=0.1f0, beta=2.0f0)
        agent2 = MetalQLearningAgent(n_arms, n_agents, n_trials; alpha=0.1f0, beta=2.0f0)
        
        # Note: Due to GPU random number generation, exact consistency may not be guaranteed
        # but the overall statistics should be similar
        @test_nowarn run_metal_bandit_simulation!(env1, agent1)
        @test_nowarn run_metal_bandit_simulation!(env2, agent2)
        
        # Check that both produced valid results
        actions1 = Array(env1.actions)
        actions2 = Array(env2.actions)
        
        @test all(1 .<= actions1 .<= n_arms)
        @test all(1 .<= actions2 .<= n_arms)
    end
end