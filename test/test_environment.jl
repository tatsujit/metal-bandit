using Test
using Metal
using Statistics
using Random

# Add the parent directory to the path
push!(LOAD_PATH, "..")
include("../metal_bandit_simulator.jl")

@testset "MetalBernoulliEnvironment Tests" begin
    
    @testset "Environment Creation" begin
        # Test basic environment creation
        n_arms, n_trials, n_agents = 3, 100, 50
        env = MetalBernoulliEnvironment(n_arms, n_trials, n_agents)
        
        @test env.n_arms == n_arms
        @test env.n_trials == n_trials
        @test env.n_agents == n_agents
        @test size(env.true_params) == (n_arms, n_agents)
        @test size(env.rewards) == (n_trials, n_arms, n_agents)
        @test size(env.actions) == (n_trials, n_agents)
        @test size(env.actual_rewards) == (n_trials, n_agents)
        
        # Test with custom parameters
        true_params = rand(Float32, n_arms, n_agents)
        env_custom = MetalBernoulliEnvironment(n_arms, n_trials, n_agents; true_params=true_params)
        @test Array(env_custom.true_params) ≈ true_params
    end
    
    @testset "Environment Parameter Validation" begin
        # Test parameter ranges
        env = MetalBernoulliEnvironment(5, 1000, 100)
        true_params_cpu = Array(env.true_params)
        
        @test all(0.0 .<= true_params_cpu .<= 1.0)
        @test size(true_params_cpu) == (5, 100)
    end
    
    @testset "Memory Layout" begin
        # Test that arrays are properly allocated on GPU
        env = MetalBernoulliEnvironment(4, 200, 25)
        
        @test typeof(env.true_params) <: MtlArray
        @test typeof(env.rewards) <: MtlArray
        @test typeof(env.actions) <: MtlArray
        @test typeof(env.actual_rewards) <: MtlArray
    end
    
    @testset "Edge Cases" begin
        # Test minimum viable sizes
        env_min = MetalBernoulliEnvironment(2, 1, 1)
        @test env_min.n_arms == 2
        @test env_min.n_trials == 1
        @test env_min.n_agents == 1
        
        # Test larger sizes
        env_large = MetalBernoulliEnvironment(20, 5000, 1000)
        @test env_large.n_arms == 20
        @test env_large.n_trials == 5000
        @test env_large.n_agents == 1000
    end
end