#!/usr/bin/env julia
"""
Comprehensive test runner for MetalBandit package
"""

using Test
using Metal
using Statistics
using BenchmarkTools
using Random
using Pkg

# Set up the environment
println("🔧 Setting up test environment...")
cd(dirname(@__FILE__))

# Add parent directory to load path
push!(LOAD_PATH, "..")

# Check Metal availability
println("🔍 Checking Metal availability...")
if Metal.functional()
    println("✅ Metal available")
    try
        println("   Max buffer length: $(Metal.max_buffer_length() ÷ 1024^2) MB")
    catch
        println("   Buffer length info not available")
    end
else
    println("❌ Metal not available - tests will be limited")
    println("   Reason: Metal GPU not functional")
end

# Test configuration
const TEST_CONFIG = (
    run_performance_tests = Metal.functional(),
    run_stress_tests = Metal.functional(),
    verbose = true
)

println("\n📋 Test Configuration:")
println("   Performance tests: $(TEST_CONFIG.run_performance_tests)")
println("   Stress tests: $(TEST_CONFIG.run_stress_tests)")
println("   Verbose output: $(TEST_CONFIG.verbose)")

# Include the main simulator
println("\n📦 Loading MetalBandit simulator...")
try
    include("../metal_bandit_simulator.jl")
    println("✅ Simulator loaded successfully")
catch e
    println("❌ Failed to load simulator: $e")
    exit(1)
end

# Define test suites
const TEST_SUITES = [
    ("Environment Tests", "test_environment.jl", true),
    ("Agent Tests", "test_agent.jl", true),
    ("Kernel Tests", "test_kernels.jl", Metal.functional()),
    ("Integration Tests", "test_integration.jl", true),
    ("Performance Tests", "test_performance.jl", TEST_CONFIG.run_performance_tests)
]

# Function to run a test suite
function run_test_suite(name::String, filename::String, should_run::Bool)
    if !should_run
        println("⏭️  Skipping $name (conditions not met)")
        return true
    end
    
    println("\n🧪 Running $name...")
    
    try
        # Capture test output
        test_result = @testset "$name" begin
            include(filename)
        end
        
        # Simple check - if we get here without exception, tests passed
        println("✅ $name completed successfully")
        return true
    catch e
        println("💥 $name crashed with error: $e")
        return false
    end
end

# Main test execution
function main()
    println("\n🚀 Starting MetalBandit Test Suite")
    println("=" ^ 50)
    
    start_time = time()
    all_passed = true
    
    # Run all test suites
    for (name, filename, should_run) in TEST_SUITES
        success = run_test_suite(name, filename, should_run)
        all_passed = all_passed && success
    end
    
    # Summary
    elapsed_time = time() - start_time
    println("\n📊 Test Summary")
    println("=" ^ 30)
    println("Total time: $(round(elapsed_time, digits=2)) seconds")
    
    if all_passed
        println("🎉 All tests passed!")
        exit(0)
    else
        println("💔 Some tests failed!")
        exit(1)
    end
end

# Additional utility functions for testing
function test_metal_functionality()
    """Quick test of Metal functionality"""
    if !Metal.functional()
        return false
    end
    
    try
        # Test basic operations
        a = Metal.ones(Float32, 10, 10)
        b = Metal.zeros(Float32, 10, 10)
        c = a + b
        result = Array(c)
        
        return all(result .≈ 1.0f0)
    catch
        return false
    end
end

function print_system_info()
    """Print system information"""
    println("\n💻 System Information:")
    println("   Julia version: $(VERSION)")
    println("   OS: $(Sys.KERNEL)")
    println("   CPU: $(Sys.CPU_NAME)")
    println("   Memory: $(Sys.total_memory() ÷ 1024^3) GB")
    
    if Metal.functional()
        println("   GPU: Metal GPU available")
        try
            println("   GPU Memory: $(Metal.max_buffer_length() ÷ 1024^3) GB")
        catch
            println("   GPU Memory: Info not available")
        end
    end
end

# Run system info if in verbose mode
if TEST_CONFIG.verbose
    print_system_info()
end

# Quick Metal test
println("\n🧪 Quick Metal functionality test...")
if test_metal_functionality()
    println("✅ Metal basic operations working")
else
    println("❌ Metal basic operations failing")
end

# Run main test suite
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end