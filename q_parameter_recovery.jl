using Metal
using Statistics
using Random
using Distributions
using Optim
using CSV
using DataFrames
using LinearAlgebra
using StatsBase

# Import CairoMakie explicitly to avoid plotting conflicts
import CairoMakie
using CairoMakie: Figure, Axis, scatter!, lines!, hist!, hlines!, heatmap!, text!, save, Colorbar, Label, axislegend

# Include the main simulator
include("metal_bandit_simulator.jl")

"""
Q学習パラメータ回復実験のためのデータ構造
"""
struct ParameterRecoveryExperiment
    n_subjects::Int                    # 被験者数（パラメータ組み合わせ数）
    n_arms::Int                       # バンディットアーム数
    n_trials::Int                     # 試行数
    true_alpha::Vector{Float64}       # 真の学習率
    true_beta::Vector{Float64}        # 真の逆温度
    estimated_alpha::Vector{Float64}  # 推定学習率
    estimated_beta::Vector{Float64}   # 推定逆温度
    actions::Matrix{Int}              # 行動データ (n_trials × n_subjects)
    rewards::Matrix{Float64}          # 報酬データ (n_trials × n_subjects)
    true_reward_probs::Matrix{Float64} # 真の報酬確率 (n_arms × n_subjects)
    log_likelihoods::Vector{Float64}  # 対数尤度
    estimation_success::Vector{Bool}  # 推定成功フラグ
end

"""
単一被験者のQ学習行動データ生成
"""
function generate_q_learning_behavior(true_alpha::Float64, true_beta::Float64, 
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
Softmax関数（数値安定版）
"""
function softmax(x::Vector{Float64})
    max_x = maximum(x)
    exp_x = exp.(x .- max_x)
    return exp_x ./ sum(exp_x)
end


"""
Q学習モデルの負対数尤度計算
"""
function negative_log_likelihood(params::Vector{Float64}, actions::Vector{Int}, 
                                rewards::Vector{Float64}, n_arms::Int)
    alpha, beta = params
    
    # パラメータ境界チェック
    if alpha <= 0 || alpha >= 1 || beta <= 0 || beta >= 20
        return Inf
    end
    
    n_trials = length(actions)
    q_values = fill(0.5, n_arms)
    nll = 0.0
    
    for trial in 1:n_trials
        # 現在のQ値に基づく行動確率
        action_probs = softmax(beta .* q_values)
        
        # 負対数尤度に追加
        prob = action_probs[actions[trial]]
        nll -= log(max(prob, 1e-10))  # 数値安定化
        
        # Q値更新
        prediction_error = rewards[trial] - q_values[actions[trial]]
        q_values[actions[trial]] += alpha * prediction_error
    end
    
    return nll
end

"""
単一被験者のパラメータ推定
"""
function estimate_q_parameters(actions::Vector{Int}, rewards::Vector{Float64}, n_arms::Int;
                              n_restarts::Int = 10)
    best_result = nothing
    best_nll = Inf
    
    for restart in 1:n_restarts
        # ランダム初期値
        initial_alpha = rand() * 0.8 + 0.1  # [0.1, 0.9]
        initial_beta = rand() * 8 + 1       # [1, 9]
        
        try
            result = optimize(
                params -> negative_log_likelihood(params, actions, rewards, n_arms),
                [initial_alpha, initial_beta],
                BFGS(),
                Optim.Options(iterations=1000, g_tol=1e-6)
            )
            
            if Optim.converged(result) && result.minimum < best_nll
                best_nll = result.minimum
                best_result = result
            end
        catch e
            # 最適化失敗時は次のリスタートへ
            continue
        end
    end
    
    if best_result === nothing
        return [NaN, NaN], Inf, false
    end
    
    estimated_params = Optim.minimizer(best_result)
    return estimated_params, best_nll, true
end

"""
メイン実験実行関数
"""
function run_parameter_recovery_experiment(n_subjects::Int = 1000, n_arms::Int = 4, 
                                         n_trials::Int = 200; seed::Int = 42)
    Random.seed!(seed)
    
    println("🧠 Q学習パラメータ回復実験開始")
    println("被験者数: $n_subjects, アーム数: $n_arms, 試行数: $n_trials")
    
    # パラメータサンプリング
    true_alpha = rand(n_subjects)                    # [0, 1]
    true_beta = rand(n_subjects) * 10               # [0, 10]
    
    # 結果格納用配列
    estimated_alpha = zeros(n_subjects)
    estimated_beta = zeros(n_subjects)
    actions = zeros(Int, n_trials, n_subjects)
    rewards = zeros(Float64, n_trials, n_subjects)
    true_reward_probs = rand(n_arms, n_subjects) * 0.6 .+ 0.2  # [0.2, 0.8]
    log_likelihoods = zeros(n_subjects)
    estimation_success = fill(false, n_subjects)
    
    # 各被験者について実験実行
    println("データ生成・パラメータ推定実行中...")
    progress_interval = max(1, n_subjects ÷ 20)
    
    Threads.@threads for subject in 1:n_subjects
        if subject % progress_interval == 0
            println("  進捗: $subject / $n_subjects ($(round(subject/n_subjects*100, digits=1))%)")
        end
        
        # 行動データ生成
        subject_actions, subject_rewards = generate_q_learning_behavior(
            true_alpha[subject], 
            true_beta[subject],
            true_reward_probs[:, subject],
            n_trials
        )
        
        actions[:, subject] = subject_actions
        rewards[:, subject] = subject_rewards
        
        # パラメータ推定
        estimated_params, nll, success = estimate_q_parameters(
            subject_actions, subject_rewards, n_arms
        )
        
        if success
            estimated_alpha[subject] = estimated_params[1]
            estimated_beta[subject] = estimated_params[2]
            log_likelihoods[subject] = -nll
            estimation_success[subject] = true
        else
            estimated_alpha[subject] = NaN
            estimated_beta[subject] = NaN
            log_likelihoods[subject] = NaN
            estimation_success[subject] = false
        end
    end
    
    success_rate = mean(estimation_success)
    println("✅ 実験完了！推定成功率: $(round(success_rate*100, digits=1))%")
    
    return ParameterRecoveryExperiment(
        n_subjects, n_arms, n_trials,
        true_alpha, true_beta,
        estimated_alpha, estimated_beta,
        actions, rewards, true_reward_probs,
        log_likelihoods, estimation_success
    )
end

"""
実験結果の詳細分析
"""
function analyze_recovery_results(experiment::ParameterRecoveryExperiment)
    success_idx = experiment.estimation_success
    
    if sum(success_idx) == 0
        println("❌ 推定成功したデータがありません")
        return nothing
    end
    
    # 成功したデータのみ分析
    true_α = experiment.true_alpha[success_idx]
    true_β = experiment.true_beta[success_idx]
    est_α = experiment.estimated_alpha[success_idx]
    est_β = experiment.estimated_beta[success_idx]
    
    # αの回復統計
    α_correlation = cor(true_α, est_α)
    α_mae = mean(abs.(true_α - est_α))
    α_rmse = sqrt(mean((true_α - est_α).^2))
    α_r2 = α_correlation^2
    
    # βの回復統計
    β_correlation = cor(true_β, est_β)
    β_mae = mean(abs.(true_β - est_β))
    β_rmse = sqrt(mean((true_β - est_β).^2))
    β_r2 = β_correlation^2
    
    # バイアス分析
    α_bias = mean(est_α - true_α)
    β_bias = mean(est_β - true_β)
    
    println("\n📊 パラメータ回復分析結果")
    println("=" ^ 50)
    println("成功率: $(round(mean(experiment.estimation_success)*100, digits=1))%")
    println("成功データ数: $(sum(success_idx)) / $(experiment.n_subjects)")
    println()
    println("【学習率 α の回復】")
    println("  相関係数: $(round(α_correlation, digits=4))")
    println("  R²: $(round(α_r2, digits=4))")
    println("  MAE: $(round(α_mae, digits=4))")
    println("  RMSE: $(round(α_rmse, digits=4))")
    println("  バイアス: $(round(α_bias, digits=4))")
    println()
    println("【逆温度 β の回復】")
    println("  相関係数: $(round(β_correlation, digits=4))")
    println("  R²: $(round(β_r2, digits=4))")
    println("  MAE: $(round(β_mae, digits=4))")
    println("  RMSE: $(round(β_rmse, digits=4))")
    println("  バイアス: $(round(β_bias, digits=4))")
    
    return (
        alpha_stats = (correlation=α_correlation, mae=α_mae, rmse=α_rmse, r2=α_r2, bias=α_bias),
        beta_stats = (correlation=β_correlation, mae=β_mae, rmse=β_rmse, r2=β_r2, bias=β_bias),
        n_success = sum(success_idx)
    )
end

"""
CairoMakieを用いた包括的可視化
"""
function create_recovery_visualization(experiment::ParameterRecoveryExperiment)
    println("🎨 可視化作成中...")
    
    success_idx = experiment.estimation_success
    failed_idx = .!success_idx
    
    # 成功・失敗データ分離
    true_α_success = experiment.true_alpha[success_idx]
    true_β_success = experiment.true_beta[success_idx]
    est_α_success = experiment.estimated_alpha[success_idx]
    est_β_success = experiment.estimated_beta[success_idx]
    
    true_α_failed = experiment.true_alpha[failed_idx]
    true_β_failed = experiment.true_beta[failed_idx]
    
    # 図のセットアップ
    fig = Figure(size=(1400, 1000), fontsize=12)
    
    # 1. Parameter recovery scatter plots
    ax1 = Axis(fig[1, 1], 
               xlabel="True Learning Rate α", ylabel="Estimated Learning Rate α",
               title="Learning Rate α Recovery")
    
    if sum(success_idx) > 0
        scatter!(ax1, true_α_success, est_α_success, 
                color=:blue, alpha=0.6, markersize=8)
        
        # Perfect recovery line (y=x)
        lines!(ax1, [0, 1], [0, 1], color=:red, linestyle=:dash, linewidth=2)
        
        # Regression line
        if length(true_α_success) > 1
            α_fit = hcat(ones(length(true_α_success)), true_α_success) \ est_α_success
            x_line = range(minimum(true_α_success), maximum(true_α_success), length=100)
            y_line = α_fit[1] .+ α_fit[2] .* x_line
            lines!(ax1, x_line, y_line, color=:green, linewidth=2)
        end
    end
    
    ax2 = Axis(fig[1, 2],
               xlabel="True Inverse Temperature β", ylabel="Estimated Inverse Temperature β", 
               title="Inverse Temperature β Recovery")
    
    if sum(success_idx) > 0
        scatter!(ax2, true_β_success, est_β_success,
                color=:orange, alpha=0.6, markersize=8)
        
        # 理想的な回復線
        max_β = max(maximum(true_β_success), maximum(est_β_success))
        lines!(ax2, [0, max_β], [0, max_β], color=:red, linestyle=:dash, linewidth=2)
        
        # 回帰線
        if length(true_β_success) > 1
            β_fit = hcat(ones(length(true_β_success)), true_β_success) \ est_β_success
            x_line = range(minimum(true_β_success), maximum(true_β_success), length=100)
            y_line = β_fit[1] .+ β_fit[2] .* x_line
            lines!(ax2, x_line, y_line, color=:green, linewidth=2)
        end
    end
    
    # 2. Parameter distributions
    ax3 = Axis(fig[2, 1], xlabel="Learning Rate α", ylabel="Density", title="True α vs Estimated α Distribution")
    
    if sum(success_idx) > 0
        hist!(ax3, experiment.true_alpha, bins=30, color=(:blue, 0.5), 
              label="True", normalization=:pdf)
        hist!(ax3, est_α_success, bins=30, color=(:red, 0.5),
              label="Estimated", normalization=:pdf)
        axislegend(ax3)
    end
    
    ax4 = Axis(fig[2, 2], xlabel="Inverse Temperature β", ylabel="Density", title="True β vs Estimated β Distribution")
    
    if sum(success_idx) > 0
        hist!(ax4, experiment.true_beta, bins=30, color=(:blue, 0.5),
              label="True", normalization=:pdf)
        hist!(ax4, est_β_success, bins=30, color=(:red, 0.5),
              label="Estimated", normalization=:pdf)
        axislegend(ax4)
    end
    
    # 3. Error analysis
    ax5 = Axis(fig[3, 1], xlabel="True Learning Rate α", ylabel="Estimation Error (Est - True)",
               title="α Estimation Error")
    
    if sum(success_idx) > 0
        α_errors = est_α_success - true_α_success
        scatter!(ax5, true_α_success, α_errors, color=:purple, alpha=0.6, markersize=6)
        hlines!(ax5, [0], color=:black, linestyle=:dash)
    end
    
    ax6 = Axis(fig[3, 2], xlabel="True Inverse Temperature β", ylabel="Estimation Error (Est - True)",
               title="β Estimation Error")
    
    if sum(success_idx) > 0
        β_errors = est_β_success - true_β_success
        scatter!(ax6, true_β_success, β_errors, color=:brown, alpha=0.6, markersize=6)
        hlines!(ax6, [0], color=:black, linestyle=:dash)
    end
    
    # 4. Success/failure distribution
    ax7 = Axis(fig[4, 1], xlabel="True Learning Rate α", ylabel="True Inverse Temperature β",
               title="Estimation Success/Failure Distribution")
    
    if sum(success_idx) > 0
        scatter!(ax7, true_α_success, true_β_success, 
                color=:green, alpha=0.7, markersize=8, label="Success")
    end
    
    if sum(failed_idx) > 0
        scatter!(ax7, true_α_failed, true_β_failed,
                color=:red, alpha=0.7, markersize=8, label="Failure")
    end
    
    if sum(success_idx) > 0 || sum(failed_idx) > 0
        axislegend(ax7)
    end
    
    # 5. Correlation matrix heatmap
    ax8 = Axis(fig[4, 2], title="Parameter Correlation")
    
    if sum(success_idx) > 0
        corr_data = [true_α_success true_β_success est_α_success est_β_success]
        corr_matrix = cor(corr_data)
        
        hm = heatmap!(ax8, corr_matrix, colormap=:RdBu, colorrange=(-1, 1))
        
        # Labels
        ax8.xticks = (1:4, ["True α", "True β", "Est α", "Est β"])
        ax8.yticks = (1:4, ["True α", "True β", "Est α", "Est β"])
        
        # Display correlation values as text
        for i in 1:4, j in 1:4
            text!(ax8, i, j, text="$(round(corr_matrix[j,i], digits=2))", 
                  align=(:center, :center), fontsize=10,
                  color = abs(corr_matrix[j,i]) > 0.5 ? :white : :black)
        end
        
        Colorbar(fig[4, 3], hm, label="Correlation")
    end
    
    # Statistics summary text
    stats = analyze_recovery_results(experiment)
    if stats !== nothing
        Label(fig[1:2, 3], 
              "Statistics Summary\n\n" *
              "Success Rate: $(round(mean(experiment.estimation_success)*100, digits=1))%\n\n" *
              "α Recovery:\n" *
              "R² = $(round(stats.alpha_stats.r2, digits=3))\n" *
              "MAE = $(round(stats.alpha_stats.mae, digits=3))\n" *
              "Bias = $(round(stats.alpha_stats.bias, digits=3))\n\n" *
              "β Recovery:\n" *
              "R² = $(round(stats.beta_stats.r2, digits=3))\n" *
              "MAE = $(round(stats.beta_stats.mae, digits=3))\n" *
              "Bias = $(round(stats.beta_stats.bias, digits=3))",
              fontsize=11, halign=:left, valign=:top)
    end
    
    return fig
end

"""
結果をCSVファイルに保存
"""
function save_results_to_csv(experiment::ParameterRecoveryExperiment, filename::String="q_parameter_recovery_results.csv")
    df = DataFrame(
        subject_id = 1:experiment.n_subjects,
        true_alpha = experiment.true_alpha,
        true_beta = experiment.true_beta,
        estimated_alpha = experiment.estimated_alpha,
        estimated_beta = experiment.estimated_beta,
        log_likelihood = experiment.log_likelihoods,
        estimation_success = experiment.estimation_success,
        alpha_error = experiment.estimated_alpha - experiment.true_alpha,
        beta_error = experiment.estimated_beta - experiment.true_beta
    )
    
    CSV.write(filename, df)
    println("📁 結果を $filename に保存しました")
    return df
end

"""
メイン実行関数
"""
function main_parameter_recovery_experiment()
    println("🚀 Q学習パラメータ回復実験開始")
    println("=" ^ 60)
    
    # 実験実行
    experiment = run_parameter_recovery_experiment(1000, 4, 200)
    
    # 分析
    stats = analyze_recovery_results(experiment)
    
    # 可視化
    fig = create_recovery_visualization(experiment)
    
    # 結果保存
    df = save_results_to_csv(experiment)
    
    # 図を保存
    save("q_parameter_recovery_visualization.png", fig, resolution=(1400, 1000))
    println("🎨 可視化を q_parameter_recovery_visualization.png に保存しました")
    
    println("\n✅ Q学習パラメータ回復実験完了！")
    
    return experiment, fig, df
end

# エクスポート
export ParameterRecoveryExperiment, run_parameter_recovery_experiment, 
       analyze_recovery_results, create_recovery_visualization,
       save_results_to_csv, main_parameter_recovery_experiment