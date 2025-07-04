using Test
using Metal
using Statistics
using Random

# Add the parent directory to the path
push!(LOAD_PATH, "..")
include("../metal_bandit_simulator.jl")

@testset "MetalQLearningAgent Tests" begin
    
    @testset "Agent Creation" begin
        # Test basic agent creation
        n_arms, n_agents, n_trials = 5, 100, 1000
        agent = MetalQLearningAgent(n_arms, n_agents, n_trials)
        
        @test agent.n_arms == n_arms
        @test agent.n_agents == n_agents
        @test agent.n_trials == n_trials
        @test size(agent.alpha) == (n_arms, n_agents)
        @test size(agent.beta) == (n_arms, n_agents)
        @test size(agent.q_values) == (n_arms, n_agents)
        @test size(agent.arm_counts) == (n_arms, n_agents)
        @test size(agent.total_rewards) == (n_arms, n_agents)
        
        # Test with custom parameters
        alpha, beta = 0.05f0, 5.0f0
        agent_custom = MetalQLearningAgent(n_arms, n_agents, n_trials; alpha=alpha, beta=beta)
        
        alpha_cpu = Array(agent_custom.alpha)
        beta_cpu = Array(agent_custom.beta)
        
        @test all(alpha_cpu .≈ alpha)
        @test all(beta_cpu .≈ beta)
    end
    
    @testset "Initial Values" begin
        # Test initial Q-values
        agent = MetalQLearningAgent(3, 50, 500)
        q_values_cpu = Array(agent.q_values)
        
        @test all(q_values_cpu .≈ 0.5f0)
        
        # Test initial counts
        arm_counts_cpu = Array(agent.arm_counts)
        total_rewards_cpu = Array(agent.total_rewards)
        
        @test all(arm_counts_cpu .== 0)
        @test all(total_rewards_cpu .≈ 0.0f0)
    end
    
    @testset "Parameter Ranges" begin
        # Test reasonable parameter ranges
        agent = MetalQLearningAgent(4, 25, 100; alpha=0.2f0, beta=1.5f0)
        
        alpha_cpu = Array(agent.alpha)
        beta_cpu = Array(agent.beta)
        
        @test all(0.0 .< alpha_cpu .< 1.0)
        @test all(beta_cpu .> 0.0)
    end
    
    @testset "Memory Layout" begin
        # Test that arrays are properly allocated on GPU
        agent = MetalQLearningAgent(6, 75, 800)
        
        @test typeof(agent.alpha) <: MtlArray
        @test typeof(agent.beta) <: MtlArray
        @test typeof(agent.q_values) <: MtlArray
        @test typeof(agent.arm_counts) <: MtlArray
        @test typeof(agent.total_rewards) <: MtlArray
    end
    
    @testset "Edge Cases" begin
        # Test minimum viable sizes
        agent_min = MetalQLearningAgent(2, 1, 1)
        @test agent_min.n_arms == 2
        @test agent_min.n_agents == 1
        @test agent_min.n_trials == 1
        
        # Test different parameter combinations
        agent_extreme = MetalQLearningAgent(10, 10, 10; alpha=0.001f0, beta=10.0f0)
        alpha_cpu = Array(agent_extreme.alpha)
        beta_cpu = Array(agent_extreme.beta)
        
        @test all(alpha_cpu .≈ 0.001f0)
        @test all(beta_cpu .≈ 10.0f0)
    end
end