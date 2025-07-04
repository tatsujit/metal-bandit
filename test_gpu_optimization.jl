using Metal
using Statistics
using Random

# Include our optimization
include("gpu_optimized_simple.jl")

# Test the optimization
println("🚀 Testing GPU Optimization...")
result = test_gpu_acceleration()

# Display results
println("\n🏆 GPU Optimization Results:")
println("=" ^ 40)
gpu_time = result.gpu_result.total_time
original_time = result.original_time
speedup = result.speedup

println("GPU Time: $(round(gpu_time, digits=2))s")
if original_time !== nothing
    println("Original Time: $(round(original_time, digits=2))s")
    println("Speedup: $(round(speedup, digits=2))x")
    if speedup > 1.0
        println("✅ GPU IS FASTER!")
    else
        println("❌ GPU needs more optimization")
    end
else
    println("Original comparison not available")
end

println("\n📊 GPU Acceleration Summary:")
println("α correlation: $(round(result.gpu_result.alpha_correlation, digits=3))")
println("β correlation: $(round(result.gpu_result.beta_correlation, digits=3))")