using Metal
using Statistics
using Random

println("🏁 Final GPU vs CPU Comparison")
println("=" ^ 50)

# Test parameters
n_subjects = 1000
n_arms = 4
n_trials = 200

println("Test configuration:")
println("  Subjects: $n_subjects")
println("  Arms: $n_arms")
println("  Trials: $n_trials")
println()

# Test 1: Original CPU implementation
println("1️⃣ Testing Original CPU Implementation...")
include("q_parameter_recovery.jl")
time_original = @elapsed result_original = run_parameter_recovery_experiment(n_subjects, n_arms, n_trials)
println("   Original CPU time: $(round(time_original, digits=2))s")
# Calculate correlations for successful estimations
success_mask = result_original.estimation_success
if sum(success_mask) > 0
    α_correlation = cor(result_original.true_alpha[success_mask], result_original.estimated_alpha[success_mask])
    β_correlation = cor(result_original.true_beta[success_mask], result_original.estimated_beta[success_mask])
    println("   α correlation: $(round(α_correlation, digits=3))")
    println("   β correlation: $(round(β_correlation, digits=3))")
    println("   Success rate: $(round(sum(success_mask)/length(success_mask)*100, digits=1))%")
else
    println("   No successful estimations")
end
println()

# Test 2: GPU optimized implementation
println("2️⃣ Testing GPU Optimized Implementation...")
include("gpu_optimized_simple.jl")
time_gpu = @elapsed result_gpu = gpu_accelerated_recovery(n_subjects, n_arms, n_trials)
println("   GPU time: $(round(time_gpu, digits=2))s")
println("   α correlation: $(round(result_gpu.alpha_correlation, digits=3))")
println("   β correlation: $(round(result_gpu.beta_correlation, digits=3))")
println()

# Final comparison
speedup = time_original / time_gpu
println("🏆 FINAL RESULTS")
println("=" ^ 30)
println("Original (CPU): $(round(time_original, digits=2))s")
println("GPU Optimized:  $(round(time_gpu, digits=2))s")
println("Speedup:        $(round(speedup, digits=2))x")
println()

if speedup > 1.0
    println("✅ GPU IS FASTER! $(round(speedup, digits=1))x speedup achieved!")
    println("🎯 GPU optimization successful!")
else
    println("❌ CPU still faster by $(round(1/speedup, digits=1))x")
    println("⚡ More GPU optimization needed")
end