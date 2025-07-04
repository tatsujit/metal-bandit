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

# Include the Q-learning parameter recovery framework
include("q_parameter_recovery.jl")

"""
GPU vs CPU 比較実験のためのデータ構造
"""
struct GPUvsCPUComparison
    n_subjects::Int
    n_arms::Int
    n_trials::Int
    
    # 共通の真のパラメータ
    true_alpha::Vector{Float64}
    true_beta::Vector{Float64}
    
    # GPU結果
    gpu_estimated_alpha::Vector{Float64}
    gpu_estimated_beta::Vector{Float64}
    gpu_estimation_success::Vector{Bool}
    gpu_simulation_time::Float64
    gpu_estimation_time::Float64
    gpu_total_time::Float64
    
    # CPU結果
    cpu_estimated_alpha::Vector{Float64}
    cpu_estimated_beta::Vector{Float64}
    cpu_estimation_success::Vector{Bool}
    cpu_simulation_time::Float64
    cpu_estimation_time::Float64
    cpu_total_time::Float64
    
    # 行動データ（GPU/CPU共通）
    actions::Matrix{Int}
    rewards::Matrix{Float64}
    true_reward_probs::Matrix{Float64}
end

"""
CPU版Q学習行動データ生成（非並列版）
"""
function generate_q_learning_behavior_cpu(true_alpha::Float64, true_beta::Float64, 
                                         reward_probs::Vector{Float64}, n_trials::Int)
    n_arms = length(reward_probs)
    actions = zeros(Int, n_trials)
    rewards = zeros(Float64, n_trials)
    q_values = fill(0.5, n_arms)  # Q値初期化
    
    for trial in 1:n_trials
        # Softmax行動選択
        action_probs = softmax(true_beta .* q_values)
        actions[trial] = StatsBase.sample(1:n_arms, Weights(action_probs))
        
        # 報酬生成
        rewards[trial] = rand() < reward_probs[actions[trial]] ? 1.0 : 0.0
        
        # Q値更新
        prediction_error = rewards[trial] - q_values[actions[trial]]
        q_values[actions[trial]] += true_alpha * prediction_error
    end
    
    return actions, rewards
end

"""
CPU版パラメータ回復実験（並列化なし）
"""
function run_cpu_parameter_recovery_experiment(n_subjects::Int, n_arms::Int, 
                                             n_trials::Int; seed::Int = 42,
                                             use_threading::Bool = false)
    Random.seed!(seed)
    
    println("🖥️  CPU版Q学習パラメータ回復実験開始")
    println("被験者数: $n_subjects, アーム数: $n_arms, 試行数: $n_trials")
    println("並列化: $(use_threading ? "有効" : "無効")")
    
    # 同じパラメータサンプリング（再現性のため）
    true_alpha = rand(n_subjects)
    true_beta = rand(n_subjects) * 10
    
    # 結果格納用配列
    estimated_alpha = zeros(n_subjects)
    estimated_beta = zeros(n_subjects)
    actions = zeros(Int, n_trials, n_subjects)
    rewards = zeros(Float64, n_trials, n_subjects)
    true_reward_probs = rand(n_arms, n_subjects) * 0.6 .+ 0.2
    estimation_success = fill(false, n_subjects)
    
    # シミュレーション時間測定
    simulation_time = @elapsed begin
        if use_threading
            # 並列化版
            println("データ生成（並列）実行中...")
            Threads.@threads for subject in 1:n_subjects
                if subject % max(1, n_subjects ÷ 20) == 0
                    println("  シミュレーション進捗: $subject / $n_subjects")
                end
                
                subject_actions, subject_rewards = generate_q_learning_behavior_cpu(
                    true_alpha[subject], 
                    true_beta[subject],
                    true_reward_probs[:, subject],
                    n_trials
                )
                
                actions[:, subject] = subject_actions
                rewards[:, subject] = subject_rewards
            end
        else
            # 非並列版
            println("データ生成（逐次）実行中...")
            for subject in 1:n_subjects
                if subject % max(1, n_subjects ÷ 20) == 0
                    println("  シミュレーション進捗: $subject / $n_subjects")
                end
                
                subject_actions, subject_rewards = generate_q_learning_behavior_cpu(
                    true_alpha[subject], 
                    true_beta[subject],
                    true_reward_probs[:, subject],
                    n_trials
                )
                
                actions[:, subject] = subject_actions
                rewards[:, subject] = subject_rewards
            end
        end
    end
    
    # パラメータ推定時間測定
    estimation_time = @elapsed begin
        if use_threading
            # 並列化版
            println("パラメータ推定（並列）実行中...")
            Threads.@threads for subject in 1:n_subjects
                if subject % max(1, n_subjects ÷ 20) == 0
                    println("  推定進捗: $subject / $n_subjects")
                end
                
                estimated_params, _, success = estimate_q_parameters(
                    actions[:, subject], rewards[:, subject], n_arms
                )
                
                if success
                    estimated_alpha[subject] = estimated_params[1]
                    estimated_beta[subject] = estimated_params[2]
                    estimation_success[subject] = true
                else
                    estimated_alpha[subject] = NaN
                    estimated_beta[subject] = NaN
                    estimation_success[subject] = false
                end
            end
        else
            # 非並列版
            println("パラメータ推定（逐次）実行中...")
            for subject in 1:n_subjects
                if subject % max(1, n_subjects ÷ 20) == 0
                    println("  推定進捗: $subject / $n_subjects")
                end
                
                estimated_params, _, success = estimate_q_parameters(
                    actions[:, subject], rewards[:, subject], n_arms
                )
                
                if success
                    estimated_alpha[subject] = estimated_params[1]
                    estimated_beta[subject] = estimated_params[2]
                    estimation_success[subject] = true
                else
                    estimated_alpha[subject] = NaN
                    estimated_beta[subject] = NaN
                    estimation_success[subject] = false
                end
            end
        end
    end
    
    success_rate = mean(estimation_success)
    total_time = simulation_time + estimation_time
    
    println("✅ CPU実験完了！")
    println("  シミュレーション時間: $(round(simulation_time, digits=2))s")
    println("  推定時間: $(round(estimation_time, digits=2))s") 
    println("  総時間: $(round(total_time, digits=2))s")
    println("  推定成功率: $(round(success_rate*100, digits=1))%")
    
    return (
        true_alpha = true_alpha,
        true_beta = true_beta,
        estimated_alpha = estimated_alpha,
        estimated_beta = estimated_beta,
        actions = actions,
        rewards = rewards,
        true_reward_probs = true_reward_probs,
        estimation_success = estimation_success,
        simulation_time = simulation_time,
        estimation_time = estimation_time,
        total_time = total_time
    )
end

"""
公平なGPU vs CPU比較実験
"""
function run_fair_gpu_vs_cpu_comparison(n_subjects::Int = 200, n_arms::Int = 4, 
                                       n_trials::Int = 200; seed::Int = 42)
    println("⚖️  GPU vs CPU 公平比較実験")
    println("=" ^ 60)
    println("被験者数: $n_subjects, アーム数: $n_arms, 試行数: $n_trials")
    println("シード: $seed")
    println()
    
    # 同じ乱数シードで同じパラメータを生成
    Random.seed!(seed)
    true_alpha = rand(n_subjects)
    true_beta = rand(n_subjects) * 10
    true_reward_probs = rand(n_arms, n_subjects) * 0.6 .+ 0.2
    
    println("🔥 GPU版実験実行...")
    
    # GPU版実験
    gpu_time = @elapsed begin
        # GPU用にパラメータを再設定
        Random.seed!(seed)
        gpu_experiment = run_parameter_recovery_experiment(n_subjects, n_arms, n_trials; seed=seed)
    end
    
    println("🖥️  CPU版実験実行...")
    
    # CPU版実験（並列化あり）
    cpu_threaded_time = @elapsed begin
        Random.seed!(seed)
        cpu_threaded_result = run_cpu_parameter_recovery_experiment(
            n_subjects, n_arms, n_trials; seed=seed, use_threading=true
        )
    end
    
    println("🖥️  CPU版実験実行（逐次処理）...")
    
    # CPU版実験（並列化なし）- 小規模サンプルで測定
    small_n = min(50, n_subjects)  # 時間短縮のため小さなサンプル
    cpu_sequential_time = @elapsed begin
        Random.seed!(seed)
        cpu_sequential_result = run_cpu_parameter_recovery_experiment(
            small_n, n_arms, n_trials; seed=seed, use_threading=false
        )
    end
    
    # 逐次処理時間を全サンプルに外挿
    cpu_sequential_extrapolated = cpu_sequential_time * (n_subjects / small_n)
    
    # 結果比較
    gpu_success_rate = mean(gpu_experiment.estimation_success)
    cpu_threaded_success_rate = mean(cpu_threaded_result.estimation_success)
    
    # GPU vs CPU比較データ構造作成
    comparison = GPUvsCPUComparison(
        n_subjects, n_arms, n_trials,
        true_alpha, true_beta,
        gpu_experiment.estimated_alpha, gpu_experiment.estimated_beta, gpu_experiment.estimation_success,
        0.0, 0.0, gpu_time,  # GPUは分離測定困難なので総時間のみ
        cpu_threaded_result.estimated_alpha, cpu_threaded_result.estimated_beta, cpu_threaded_result.estimation_success,
        cpu_threaded_result.simulation_time, cpu_threaded_result.estimation_time, cpu_threaded_result.total_time,
        Array(gpu_experiment.actions), Array(gpu_experiment.rewards), Array(gpu_experiment.true_reward_probs)
    )
    
    # 結果サマリー
    println("\n📊 GPU vs CPU 比較結果")
    println("=" ^ 50)
    
    println("⏱️  実行時間比較:")
    println("  GPU総時間:        $(round(gpu_time, digits=2))s")
    println("  CPU並列総時間:    $(round(cpu_threaded_result.total_time, digits=2))s")
    println("  CPU逐次総時間:    $(round(cpu_sequential_extrapolated, digits=2))s (外挿)")
    
    gpu_speedup_vs_threaded = cpu_threaded_result.total_time / gpu_time
    gpu_speedup_vs_sequential = cpu_sequential_extrapolated / gpu_time
    
    println("  GPU高速化率 (vs 並列CPU): $(round(gpu_speedup_vs_threaded, digits=2))x")
    println("  GPU高速化率 (vs 逐次CPU): $(round(gpu_speedup_vs_sequential, digits=2))x")
    
    println("\n🎯 推定精度比較:")
    println("  GPU成功率:        $(round(gpu_success_rate*100, digits=1))%")
    println("  CPU成功率:        $(round(cpu_threaded_success_rate*100, digits=1))%")
    
    # GPU vs CPU の推定精度比較
    if gpu_success_rate > 0 && cpu_threaded_success_rate > 0
        gpu_success_idx = gpu_experiment.estimation_success
        cpu_success_idx = cpu_threaded_result.estimation_success
        
        if sum(gpu_success_idx) > 0 && sum(cpu_success_idx) > 0
            gpu_alpha_corr = cor(true_alpha[gpu_success_idx], gpu_experiment.estimated_alpha[gpu_success_idx])
            gpu_beta_corr = cor(true_beta[gpu_success_idx], gpu_experiment.estimated_beta[gpu_success_idx])
            
            cpu_alpha_corr = cor(true_alpha[cpu_success_idx], cpu_threaded_result.estimated_alpha[cpu_success_idx])
            cpu_beta_corr = cor(true_beta[cpu_success_idx], cpu_threaded_result.estimated_beta[cpu_success_idx])
            
            println("  GPU α回復相関:    $(round(gpu_alpha_corr, digits=3))")
            println("  CPU α回復相関:    $(round(cpu_alpha_corr, digits=3))")
            println("  GPU β回復相関:    $(round(gpu_beta_corr, digits=3))")
            println("  CPU β回復相関:    $(round(cpu_beta_corr, digits=3))")
        end
    end
    
    return comparison, (
        gpu_time = gpu_time,
        cpu_threaded_time = cpu_threaded_result.total_time,
        cpu_sequential_time = cpu_sequential_extrapolated,
        speedup_vs_threaded = gpu_speedup_vs_threaded,
        speedup_vs_sequential = gpu_speedup_vs_sequential
    )
end

"""
GPU vs CPU 比較可視化
"""
function create_gpu_vs_cpu_visualization(comparison::GPUvsCPUComparison, timing_results)
    println("🎨 GPU vs CPU 比較可視化作成中...")
    
    # 成功したデータのみ抽出
    gpu_success_idx = comparison.gpu_estimation_success
    cpu_success_idx = comparison.cpu_estimation_success
    
    gpu_true_α = comparison.true_alpha[gpu_success_idx]
    gpu_est_α = comparison.gpu_estimated_alpha[gpu_success_idx]
    gpu_true_β = comparison.true_beta[gpu_success_idx]
    gpu_est_β = comparison.gpu_estimated_beta[gpu_success_idx]
    
    cpu_true_α = comparison.true_alpha[cpu_success_idx]
    cpu_est_α = comparison.cpu_estimated_alpha[cpu_success_idx]
    cpu_true_β = comparison.true_beta[cpu_success_idx]
    cpu_est_β = comparison.cpu_estimated_beta[cpu_success_idx]
    
    fig = Figure(size=(1600, 1200), fontsize=12)
    
    # 1. パラメータ回復比較 (α)
    ax1 = Axis(fig[1, 1], 
               xlabel="True Learning Rate α", ylabel="Estimated Learning Rate α",
               title="Learning Rate α Recovery: GPU vs CPU")
    
    if length(gpu_true_α) > 0
        scatter!(ax1, gpu_true_α, gpu_est_α, 
                color=:blue, alpha=0.6, markersize=8, label="GPU")
    end
    
    if length(cpu_true_α) > 0
        scatter!(ax1, cpu_true_α, cpu_est_α, 
                color=:red, alpha=0.6, markersize=8, label="CPU")
    end
    
    # Perfect recovery line
    lines!(ax1, [0, 1], [0, 1], color=:black, linestyle=:dash, linewidth=2)
    axislegend(ax1)
    
    # 2. パラメータ回復比較 (β)
    ax2 = Axis(fig[1, 2],
               xlabel="True Inverse Temperature β", ylabel="Estimated Inverse Temperature β",
               title="Inverse Temperature β Recovery: GPU vs CPU")
    
    if length(gpu_true_β) > 0
        scatter!(ax2, gpu_true_β, gpu_est_β,
                color=:blue, alpha=0.6, markersize=8, label="GPU")
    end
    
    if length(cpu_true_β) > 0
        scatter!(ax2, cpu_true_β, cpu_est_β,
                color=:red, alpha=0.6, markersize=8, label="CPU")
    end
    
    # Perfect recovery line
    if length(gpu_true_β) > 0 || length(cpu_true_β) > 0
        all_true_β = vcat(gpu_true_β, cpu_true_β)
        all_est_β = vcat(gpu_est_β, cpu_est_β)
        max_β = max(maximum(all_true_β), maximum(all_est_β))
        lines!(ax2, [0, max_β], [0, max_β], color=:black, linestyle=:dash, linewidth=2)
    end
    axislegend(ax2)
    
    # 3. 実行時間比較
    ax3 = Axis(fig[2, 1], 
               xlabel="Computation Method", ylabel="Time (seconds)",
               title="Execution Time Comparison")
    
    methods = ["GPU", "CPU\n(Parallel)", "CPU\n(Sequential)"]
    times = [timing_results.gpu_time, timing_results.cpu_threaded_time, timing_results.cpu_sequential_time]
    colors = [:blue, :red, :orange]
    
    barplot!(ax3, 1:3, times, color=colors, alpha=0.7)
    ax3.xticks = (1:3, methods)
    
    # 時間を棒グラフ上に表示
    for (i, time) in enumerate(times)
        text!(ax3, i, time + maximum(times)*0.02, text="$(round(time, digits=1))s", 
              align=(:center, :bottom), fontsize=10)
    end
    
    # 4. 高速化率表示
    ax4 = Axis(fig[2, 2],
               xlabel="Comparison", ylabel="Speedup Factor",
               title="GPU Speedup Analysis")
    
    speedup_labels = ["GPU vs\nCPU Parallel", "GPU vs\nCPU Sequential"]
    speedup_values = [timing_results.speedup_vs_threaded, timing_results.speedup_vs_sequential]
    speedup_colors = [:green, :darkgreen]
    
    barplot!(ax4, 1:2, speedup_values, color=speedup_colors, alpha=0.7)
    ax4.xticks = (1:2, speedup_labels)
    
    # 高速化率を棒グラフ上に表示
    for (i, speedup) in enumerate(speedup_values)
        text!(ax4, i, speedup + maximum(speedup_values)*0.02, text="$(round(speedup, digits=1))x", 
              align=(:center, :bottom), fontsize=10)
    end
    
    # 水平線で1x（同等）を示す
    hlines!(ax4, [1.0], color=:black, linestyle=:dash, alpha=0.5)
    
    # 5. 推定精度比較
    ax5 = Axis(fig[3, 1],
               xlabel="Method", ylabel="Success Rate (%)",
               title="Parameter Estimation Success Rate")
    
    gpu_success_rate = mean(comparison.gpu_estimation_success) * 100
    cpu_success_rate = mean(comparison.cpu_estimation_success) * 100
    
    success_methods = ["GPU", "CPU"]
    success_rates = [gpu_success_rate, cpu_success_rate]
    success_colors = [:blue, :red]
    
    barplot!(ax5, 1:2, success_rates, color=success_colors, alpha=0.7)
    ax5.xticks = (1:2, success_methods)
    
    for (i, rate) in enumerate(success_rates)
        text!(ax5, i, rate + 2, text="$(round(rate, digits=1))%", 
              align=(:center, :bottom), fontsize=10)
    end
    
    # 6. 相関比較
    ax6 = Axis(fig[3, 2], 
               xlabel="Parameter", ylabel="Correlation",
               title="Parameter Recovery Correlation")
    
    if length(gpu_true_α) > 0 && length(cpu_true_α) > 0
        gpu_α_corr = cor(gpu_true_α, gpu_est_α)
        gpu_β_corr = cor(gpu_true_β, gpu_est_β)
        cpu_α_corr = cor(cpu_true_α, cpu_est_α)
        cpu_β_corr = cor(cpu_true_β, cpu_est_β)
        
        correlations = [gpu_α_corr gpu_β_corr; cpu_α_corr cpu_β_corr]
        
        barplot!(ax6, [1, 1.3], [gpu_α_corr, cpu_α_corr], 
                color=[:blue, :red], alpha=0.7, width=0.25)
        barplot!(ax6, [2, 2.3], [gpu_β_corr, cpu_β_corr], 
                color=[:blue, :red], alpha=0.7, width=0.25)
        
        ax6.xticks = ([1.15, 2.15], ["Learning Rate α", "Inverse Temp β"])
        
        # 凡例用のダミープロット
        scatter!(ax6, [0], [0], color=:blue, label="GPU", markersize=0)
        scatter!(ax6, [0], [0], color=:red, label="CPU", markersize=0)
        axislegend(ax6)
    end
    
    # 統計サマリーテキスト
    Label(fig[1:2, 3], 
          "Performance Summary\n\n" *
          "Execution Time:\n" *
          "GPU: $(round(timing_results.gpu_time, digits=1))s\n" *
          "CPU (Parallel): $(round(timing_results.cpu_threaded_time, digits=1))s\n" *
          "CPU (Sequential): $(round(timing_results.cpu_sequential_time, digits=1))s\n\n" *
          "Speedup:\n" *
          "GPU vs CPU (||): $(round(timing_results.speedup_vs_threaded, digits=1))x\n" *
          "GPU vs CPU (seq): $(round(timing_results.speedup_vs_sequential, digits=1))x\n\n" *
          "Success Rate:\n" *
          "GPU: $(round(mean(comparison.gpu_estimation_success)*100, digits=1))%\n" *
          "CPU: $(round(mean(comparison.cpu_estimation_success)*100, digits=1))%\n\n" *
          "Sample Size: $(comparison.n_subjects) subjects\n" *
          "Trials: $(comparison.n_trials) per subject\n" *
          "Arms: $(comparison.n_arms)",
          fontsize=11, halign=:left, valign=:top)
    
    return fig
end

"""
結果をCSVファイルに保存
"""
function save_comparison_results(comparison::GPUvsCPUComparison, timing_results, 
                               filename::String = "gpu_vs_cpu_comparison_results.csv")
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
    
    CSV.write(filename, df)
    println("📁 比較結果を $filename に保存しました")
    
    # タイミング結果も別途保存
    timing_df = DataFrame(
        method = ["GPU", "CPU_Parallel", "CPU_Sequential"],
        time_seconds = [timing_results.gpu_time, timing_results.cpu_threaded_time, timing_results.cpu_sequential_time],
        speedup_vs_gpu = [1.0, timing_results.gpu_time/timing_results.cpu_threaded_time, timing_results.gpu_time/timing_results.cpu_sequential_time]
    )
    
    timing_filename = replace(filename, ".csv" => "_timing.csv")
    CSV.write(timing_filename, timing_df)
    println("📁 タイミング結果を $timing_filename に保存しました")
    
    return df, timing_df
end

"""
メイン比較実験実行関数
"""
function main_gpu_vs_cpu_comparison_experiment()
    println("🚀 GPU vs CPU Q学習パラメータ回復比較実験開始")
    println("=" ^ 70)
    
    # 比較実験実行
    comparison, timing_results = run_fair_gpu_vs_cpu_comparison(200, 4, 200)
    
    # 可視化
    fig = create_gpu_vs_cpu_visualization(comparison, timing_results)
    
    # 結果保存
    df, timing_df = save_comparison_results(comparison, timing_results)
    
    # 図を保存
    save("gpu_vs_cpu_comparison_visualization.png", fig, size=(1600, 1200))
    println("🎨 比較可視化を gpu_vs_cpu_comparison_visualization.png に保存しました")
    
    println("\n✅ GPU vs CPU 比較実験完了！")
    
    return comparison, timing_results, fig, df, timing_df
end

# エクスポート
export GPUvsCPUComparison, run_fair_gpu_vs_cpu_comparison, 
       create_gpu_vs_cpu_visualization, save_comparison_results,
       main_gpu_vs_cpu_comparison_experiment