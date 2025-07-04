using Test
using Metal
using Statistics
using Random

# Add the parent directory to the path
push!(LOAD_PATH, "..")
include("../metal_bandit_simulator.jl")

@testset "GPU Kernel Tests" begin
    
    @testset "Softmax Action Selection Kernel" begin
        # Test softmax action selection
        n_arms, n_agents = 3, 10
        q_values = Metal.rand(Float32, n_arms, n_agents)
        beta = Metal.fill(2.0f0, n_arms, n_agents)
        actions = Metal.zeros(Int32, 1, n_agents)
        random_vals = Metal.rand(Float32, n_agents)
        
        # Run kernel
        kernel = Metal.@metal launch=false softmax_action_selection_kernel!(
            actions, q_values, beta, random_vals, n_arms, n_agents, 1
        )
        kernel(actions, q_values, beta, random_vals, n_arms, n_agents, 1;
               threads=n_agents, groups=1)
        
        actions_cpu = Array(actions)
        
        # Check that actions are in valid range
        @test all(1 .<= actions_cpu .<= n_arms)
        @test size(actions_cpu) == (1, n_agents)
    end
    
    @testset "Reward Generation Kernel" begin
        # Test reward generation
        n_agents = 20
        n_arms = 4
        actions = Metal.ones(Int32, 1, n_agents)  # All select arm 1
        true_params = Metal.fill(0.7f0, n_arms, n_agents)  # 70% success rate
        rewards = Metal.zeros(Float32, 1, n_agents)
        random_vals = Metal.rand(Float32, n_agents)
        
        # Run kernel
        kernel = Metal.@metal launch=false reward_generation_kernel!(
            rewards, actions, true_params, random_vals, n_agents, 1
        )
        kernel(rewards, actions, true_params, random_vals, n_agents, 1;
               threads=n_agents, groups=1)
        
        rewards_cpu = Array(rewards)
        
        # Check that rewards are binary
        @test all(r -> r == 0.0f0 || r == 1.0f0, rewards_cpu)
        @test size(rewards_cpu) == (1, n_agents)
    end
    
    @testset "Q-Learning Update Kernel" begin
        # Test Q-learning update
        n_arms, n_agents = 3, 15
        q_values = Metal.fill(0.5f0, n_arms, n_agents)
        arm_counts = Metal.zeros(Int32, n_arms, n_agents)
        total_rewards = Metal.zeros(Float32, n_arms, n_agents)
        actions = Metal.ones(Int32, 1, n_agents)  # All select arm 1
        rewards = Metal.ones(Float32, 1, n_agents)  # All get reward 1
        alpha = Metal.fill(0.1f0, n_arms, n_agents)
        
        # Run kernel
        kernel = Metal.@metal launch=false q_learning_update_kernel!(
            q_values, arm_counts, total_rewards, actions, rewards, alpha, n_agents, 1
        )
        kernel(q_values, arm_counts, total_rewards, actions, rewards, alpha, n_agents, 1;
               threads=n_agents, groups=1)
        
        q_values_cpu = Array(q_values)
        arm_counts_cpu = Array(arm_counts)
        total_rewards_cpu = Array(total_rewards)
        
        # Check that Q-values were updated (should be 0.5 + 0.1 * (1 - 0.5) = 0.55)
        @test all(q_values_cpu[1, :] .≈ 0.55f0)
        @test all(arm_counts_cpu[1, :] .== 1)
        @test all(total_rewards_cpu[1, :] .≈ 1.0f0)
    end
    
    @testset "MLE Statistics Kernel" begin
        # Test MLE statistics computation
        n_arms, n_agents, n_trials = 3, 10, 50
        actions = Metal.ones(Int32, n_trials, n_agents)  # All select arm 1
        rewards = Metal.ones(Float32, n_trials, n_agents)  # All get reward 1
        successes = Metal.zeros(Float32, n_arms, n_agents)
        trials = Metal.zeros(Float32, n_arms, n_agents)
        
        # Run kernel
        kernel = Metal.@metal launch=false mle_statistics_kernel!(
            successes, trials, actions, rewards, n_arms, n_agents, n_trials
        )
        total_elements = n_arms * n_agents
        kernel(successes, trials, actions, rewards, n_arms, n_agents, n_trials;
               threads=min(1024, total_elements), groups=cld(total_elements, 1024))
        
        successes_cpu = Array(successes)
        trials_cpu = Array(trials)
        
        # Arm 1 should have n_trials successes and trials
        @test all(successes_cpu[1, :] .≈ Float32(n_trials))
        @test all(trials_cpu[1, :] .≈ Float32(n_trials))
        # Other arms should have 0
        @test all(successes_cpu[2:end, :] .≈ 0.0f0)
        @test all(trials_cpu[2:end, :] .≈ 0.0f0)
    end
    
    @testset "MLE Estimation Kernel" begin
        # Test MLE parameter estimation
        n_arms, n_agents = 4, 8
        successes = Metal.ones(Float32, n_arms, n_agents) .* 7.0f0  # 7 successes
        trials = Metal.ones(Float32, n_arms, n_agents) .* 10.0f0    # 10 trials
        estimated_params = Metal.zeros(Float32, n_arms, n_agents)
        
        # Run kernel
        kernel = Metal.@metal launch=false mle_estimation_kernel!(
            estimated_params, successes, trials, n_arms, n_agents
        )
        total_elements = n_arms * n_agents
        kernel(estimated_params, successes, trials, n_arms, n_agents;
               threads=min(1024, total_elements), groups=cld(total_elements, 1024))
        
        estimated_params_cpu = Array(estimated_params)
        
        # With Laplace smoothing: (7 + 1) / (10 + 2) = 8/12 = 2/3
        expected_value = 8.0f0 / 12.0f0
        @test all(estimated_params_cpu .≈ expected_value)
    end
    
    @testset "Batch MLE Estimation Kernel" begin
        # Test batch MLE estimation
        n_arms, n_agents, n_trials = 2, 5, 100
        batch_size = 20
        
        # Create test data: arm 1 selected with 80% success rate
        actions = Metal.ones(Int32, n_trials, n_agents)
        rewards = Metal.zeros(Float32, n_trials, n_agents)
        # Set first 80% of trials to success
        success_trials = Int(n_trials * 0.8)
        rewards_cpu = Array(rewards)
        rewards_cpu[1:success_trials, :] .= 1.0f0
        rewards = MtlArray(rewards_cpu)
        
        estimated_params = Metal.zeros(Float32, n_arms, n_agents)
        
        # Run kernel
        kernel = Metal.@metal launch=false batch_mle_estimation_kernel!(
            estimated_params, actions, rewards, n_arms, n_agents, n_trials, batch_size
        )
        total_elements = n_arms * n_agents
        kernel(estimated_params, actions, rewards, n_arms, n_agents, n_trials, batch_size;
               threads=min(1024, total_elements), groups=cld(total_elements, 1024))
        
        estimated_params_cpu = Array(estimated_params)
        
        # Arm 1 should have estimate around 0.8 (with Laplace smoothing)
        expected_arm1 = (success_trials + 1.0f0) / (n_trials + 2.0f0)
        @test all(estimated_params_cpu[1, :] .≈ expected_arm1)
        
        # Arm 2 should have estimate around 0.5 (no selections, pure smoothing)
        expected_arm2 = 1.0f0 / 2.0f0
        @test all(estimated_params_cpu[2, :] .≈ expected_arm2)
    end
end