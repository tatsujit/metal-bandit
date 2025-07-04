using Metal
using Statistics
using Random
using Distributions
using Optim
using CSV
using DataFrames
using LinearAlgebra
using StatsBase
using BenchmarkTools

# Import CairoMakie explicitly to avoid plotting conflicts
import CairoMakie
using CairoMakie: Figure, Axis, scatter!, lines!, hist!, hlines!, heatmap!, text!, save, Colorbar, Label, axislegend, barplot!

# Include the comparison framework
include("gpu_vs_cpu_comparison.jl")

"""
Large-scale GPU vs CPU comparison experiment
"""
function run_large_scale_comparison(;
    n_subjects::Int = 2000,
    n_arms::Int = 8,
    n_trials::Int = 500,
    n_threads::Int = 8,
    seed::Int = 42
)
    
    println("🚀 LARGE-SCALE GPU vs CPU Q学習パラメータ回復比較実験")
    println("=" ^ 80)
    println("スケール: $(n_subjects) subjects × $(n_arms) arms × $(n_trials) trials")
    println("総決定数: $(n_subjects * n_trials) decisions")
    println("スレッド数: $(n_threads)")
    println("シード: $(seed)")
    println()
    
    # Verify threading
    println("利用可能スレッド数: $(Threads.nthreads())")
    if Threads.nthreads() < n_threads
        @warn "期待スレッド数 $(n_threads) より少ない $(Threads.nthreads()) スレッドで実行中"
    end
    println()
    
    # Memory usage estimation
    memory_per_subject_mb = (n_trials * (4 + 8) + n_arms * 8 + 100) / 1024 / 1024
    total_memory_mb = memory_per_subject_mb * n_subjects
    println("推定メモリ使用量: $(round(total_memory_mb, digits=1)) MB")
    println()
    
    # Run the comparison with large scale parameters
    comparison, timing_results = run_fair_gpu_vs_cpu_comparison(
        n_subjects, n_arms, n_trials; seed=seed
    )
    
    return comparison, timing_results
end

"""
Create large-scale performance analysis visualization
"""
function create_large_scale_visualization(comparison, timing_results, small_scale_results=nothing)
    println("🎨 大規模実験可視化作成中...")
    
    fig = Figure(size=(1800, 1400), fontsize=14)
    
    # 1. Performance comparison bar chart
    ax1 = Axis(fig[1, 1],
               xlabel="Method", ylabel="Time (seconds)",
               title="Large-Scale Performance Comparison")
    
    methods = ["GPU", "CPU (8 threads)", "CPU (1 thread)"]
    times = [timing_results.gpu_time, timing_results.cpu_threaded_time, timing_results.cpu_sequential_time]
    colors = [:blue, :red, :orange]
    
    barplot!(ax1, 1:3, times, color=colors, alpha=0.7)
    ax1.xticks = (1:3, methods)
    
    # Add timing labels
    for (i, time) in enumerate(times)
        text!(ax1, i, time + maximum(times)*0.02, 
              text="$(round(time, digits=1))s", 
              align=(:center, :bottom), fontsize=12)
    end
    
    # 2. Speedup analysis
    ax2 = Axis(fig[1, 2],
               xlabel="Comparison", ylabel="Speedup Factor",
               title="GPU Performance Analysis")
    
    speedup_labels = ["GPU vs\nCPU (8 threads)", "GPU vs\nCPU (1 thread)"]
    speedup_values = [timing_results.cpu_threaded_time / timing_results.gpu_time,
                     timing_results.cpu_sequential_time / timing_results.gpu_time]
    speedup_colors = [val > 1.0 ? :green : :red for val in speedup_values]
    
    barplot!(ax2, 1:2, speedup_values, color=speedup_colors, alpha=0.7)
    ax2.xticks = (1:2, speedup_labels)
    
    # Add speedup labels
    for (i, speedup) in enumerate(speedup_values)
        text!(ax2, i, speedup + maximum(speedup_values)*0.02,
              text="$(round(speedup, digits=2))x",
              align=(:center, :bottom), fontsize=12)
    end
    
    # Reference line at 1x
    hlines!(ax2, [1.0], color=:black, linestyle=:dash, alpha=0.5)
    
    # 3. Scale comparison (if small scale results provided)
    if small_scale_results !== nothing
        ax3 = Axis(fig[2, 1],
                   xlabel="Scale", ylabel="Time (seconds)",
                   title="Scaling Performance Analysis")
        
        # This would need small scale timing data
        # For now, create placeholder
        scales = ["Small\n(200×4×200)", "Large\n(2000×8×500)"]
        ax3.xticks = (1:2, scales)
        text!(ax3, 1, 0.5, text="Small scale\nresults needed", align=(:center, :center))
    end
    
    # 4. Parameter recovery quality
    ax4 = Axis(fig[2, 2],
               xlabel="Parameter", ylabel="Correlation",
               title="Parameter Recovery Quality")
    
    success_idx_gpu = comparison.gpu_estimation_success
    success_idx_cpu = comparison.cpu_estimation_success
    
    if sum(success_idx_gpu) > 0 && sum(success_idx_cpu) > 0
        gpu_α_corr = cor(comparison.true_alpha[success_idx_gpu], 
                         comparison.gpu_estimated_alpha[success_idx_gpu])
        gpu_β_corr = cor(comparison.true_beta[success_idx_gpu], 
                         comparison.gpu_estimated_beta[success_idx_gpu])
        cpu_α_corr = cor(comparison.true_alpha[success_idx_cpu], 
                         comparison.cpu_estimated_alpha[success_idx_cpu])
        cpu_β_corr = cor(comparison.true_beta[success_idx_cpu], 
                         comparison.cpu_estimated_beta[success_idx_cpu])
        
        barplot!(ax4, [1, 1.3], [gpu_α_corr, cpu_α_corr], 
                color=[:blue, :red], alpha=0.7, width=0.25)
        barplot!(ax4, [2, 2.3], [gpu_β_corr, cpu_β_corr], 
                color=[:blue, :red], alpha=0.7, width=0.25)
        
        ax4.xticks = ([1.15, 2.15], ["Learning Rate α", "Inverse Temp β"])
        
        # Legend
        scatter!(ax4, [0], [0], color=:blue, label="GPU", markersize=0)
        scatter!(ax4, [0], [0], color=:red, label="CPU", markersize=0)
        axislegend(ax4)
    end
    
    # 5. Success rate comparison
    ax5 = Axis(fig[3, 1],
               xlabel="Method", ylabel="Success Rate (%)",
               title="Parameter Estimation Success Rate")
    
    gpu_success_rate = mean(comparison.gpu_estimation_success) * 100
    cpu_success_rate = mean(comparison.cpu_estimation_success) * 100
    
    success_methods = ["GPU", "CPU"]
    success_rates = [gpu_success_rate, cpu_success_rate]
    
    barplot!(ax5, 1:2, success_rates, color=[:blue, :red], alpha=0.7)
    ax5.xticks = (1:2, success_methods)
    
    for (i, rate) in enumerate(success_rates)
        text!(ax5, i, rate + 2, text="$(round(rate, digits=1))%", 
              align=(:center, :bottom), fontsize=12)
    end
    
    # 6. Performance summary text
    Label(fig[3, 2], 
          "Large-Scale Performance Summary\\n\\n" *
          "Dataset Size:\\n" *
          "Subjects: $(comparison.n_subjects)\\n" *
          "Arms: $(comparison.n_arms)\\n" *
          "Trials: $(comparison.n_trials)\\n" *
          "Total Decisions: $(comparison.n_subjects * comparison.n_trials)\\n\\n" *
          "Execution Time:\\n" *
          "GPU: $(round(timing_results.gpu_time, digits=1))s\\n" *
          "CPU (8 threads): $(round(timing_results.cpu_threaded_time, digits=1))s\\n" *
          "CPU (1 thread): $(round(timing_results.cpu_sequential_time, digits=1))s\\n\\n" *
          "GPU Speedup:\\n" *
          "vs CPU (8 threads): $(round(timing_results.cpu_threaded_time/timing_results.gpu_time, digits=2))x\\n" *
          "vs CPU (1 thread): $(round(timing_results.cpu_sequential_time/timing_results.gpu_time, digits=2))x\\n\\n" *
          "Success Rate:\\n" *
          "GPU: $(round(mean(comparison.gpu_estimation_success)*100, digits=1))%\\n" *
          "CPU: $(round(mean(comparison.cpu_estimation_success)*100, digits=1))%",
          fontsize=11, halign=:left, valign=:top)
    
    return fig
end

"""
Save large-scale results with detailed analysis
"""
function save_large_scale_results(comparison, timing_results, 
                                filename_prefix::String="large_scale_gpu_vs_cpu")
    
    # Save detailed comparison results
    results_filename = "$(filename_prefix)_results.csv"
    timing_filename = "$(filename_prefix)_timing.csv"
    
    # Detailed results
    df = DataFrame(
        subject_id = 1:comparison.n_subjects,
        true_alpha = comparison.true_alpha,
        true_beta = comparison.true_beta,
        gpu_estimated_alpha = comparison.gpu_estimated_alpha,
        gpu_estimated_beta = comparison.gpu_estimated_beta,
        gpu_estimation_success = comparison.gpu_estimation_success,
        cpu_estimated_alpha = comparison.cpu_estimated_alpha,
        cpu_estimated_beta = comparison.cpu_estimated_beta,
        cpu_estimation_success = comparison.cpu_estimation_success,
        gpu_alpha_error = comparison.gpu_estimated_alpha - comparison.true_alpha,
        gpu_beta_error = comparison.gpu_estimated_beta - comparison.true_beta,
        cpu_alpha_error = comparison.cpu_estimated_alpha - comparison.true_alpha,
        cpu_beta_error = comparison.cpu_estimated_beta - comparison.true_beta
    )
    
    CSV.write(results_filename, df)
    println("📁 大規模比較結果を $(results_filename) に保存しました")
    
    # Timing analysis
    timing_df = DataFrame(
        method = ["GPU", "CPU_8_Threads", "CPU_1_Thread"],
        time_seconds = [timing_results.gpu_time, timing_results.cpu_threaded_time, timing_results.cpu_sequential_time],
        speedup_vs_gpu = [1.0, 
                         timing_results.gpu_time/timing_results.cpu_threaded_time, 
                         timing_results.gpu_time/timing_results.cpu_sequential_time],
        dataset_scale = ["$(comparison.n_subjects)×$(comparison.n_arms)×$(comparison.n_trials)" for _ in 1:3],
        total_decisions = [comparison.n_subjects * comparison.n_trials for _ in 1:3]
    )
    
    CSV.write(timing_filename, timing_df)
    println("📁 大規模タイミング結果を $(timing_filename) に保存しました")
    
    return df, timing_df
end

"""
Main large-scale experiment execution
"""
function main_large_scale_experiment()
    println("🔥 LARGE-SCALE GPU vs CPU Q学習パラメータ回復実験開始")
    println("=" ^ 90)
    
    # Check system resources
    println("システム情報:")
    println("  利用可能スレッド数: $(Threads.nthreads())")
    try
        # Try to get system memory info
        mem_info = read(`sysctl hw.memsize`, String)
        mem_gb = parse(Int, split(mem_info)[2]) / 1024^3
        println("  システムメモリ: $(round(mem_gb, digits=1)) GB")
    catch
        println("  システムメモリ: 情報取得不可")
    end
    println()
    
    # Run large-scale comparison
    comparison, timing_results = run_large_scale_comparison(
        n_subjects=2000,
        n_arms=8, 
        n_trials=500,
        n_threads=8,
        seed=42
    )
    
    # Create visualization
    fig = create_large_scale_visualization(comparison, timing_results)
    
    # Save results
    df, timing_df = save_large_scale_results(comparison, timing_results)
    
    # Save visualization
    save("large_scale_gpu_vs_cpu_visualization.png", fig, size=(1800, 1400))
    println("🎨 大規模比較可視化を large_scale_gpu_vs_cpu_visualization.png に保存しました")
    
    # Performance analysis
    println("\\n" * "=" * 80)
    println("🏆 LARGE-SCALE PERFORMANCE ANALYSIS")
    println("=" * 80)
    
    gpu_vs_cpu_8_speedup = timing_results.cpu_threaded_time / timing_results.gpu_time
    gpu_vs_cpu_1_speedup = timing_results.cpu_sequential_time / timing_results.gpu_time
    
    println("データセット規模:")
    println("  被験者数: $(comparison.n_subjects)")
    println("  アーム数: $(comparison.n_arms)")  
    println("  試行数: $(comparison.n_trials)")
    println("  総決定数: $(comparison.n_subjects * comparison.n_trials)")
    println()
    
    println("実行時間:")
    println("  GPU: $(round(timing_results.gpu_time, digits=2))s")
    println("  CPU (8スレッド): $(round(timing_results.cpu_threaded_time, digits=2))s")
    println("  CPU (1スレッド): $(round(timing_results.cpu_sequential_time, digits=2))s")
    println()
    
    println("GPU性能分析:")
    if gpu_vs_cpu_8_speedup > 1.0
        println("  🚀 GPU は 8スレッドCPU より $(round(gpu_vs_cpu_8_speedup, digits=2))x 高速")
    else
        println("  ⚡ 8スレッドCPU は GPU より $(round(1/gpu_vs_cpu_8_speedup, digits=2))x 高速")
    end
    
    if gpu_vs_cpu_1_speedup > 1.0
        println("  🚀 GPU は 1スレッドCPU より $(round(gpu_vs_cpu_1_speedup, digits=2))x 高速")
    else
        println("  ⚡ 1スレッドCPU は GPU より $(round(1/gpu_vs_cpu_1_speedup, digits=2))x 高速")
    end
    println()
    
    println("推定精度:")
    println("  GPU成功率: $(round(mean(comparison.gpu_estimation_success)*100, digits=1))%")
    println("  CPU成功率: $(round(mean(comparison.cpu_estimation_success)*100, digits=1))%")
    
    println("\\n✅ 大規模GPU vs CPU比較実験完了！")
    
    return comparison, timing_results, fig, df, timing_df
end

# Export functions
export run_large_scale_comparison, create_large_scale_visualization,
       save_large_scale_results, main_large_scale_experiment