using Pkg
Pkg.add("Metal")
using Metal
using Statistics
using BenchmarkTools
using Random

# Apple Silicon GPU用のシンプルなMLE実装
struct AppleSiliconBanditMLE{T}
    n_bandits::Int
    n_arms::Int
    n_samples::Int
    # Metal arrays for GPU computation
    observations::MtlArray{T, 3}  # (n_samples, n_arms, n_bandits)
    rewards::MtlArray{T, 3}       # (n_samples, n_arms, n_bandits)
end

function AppleSiliconBanditMLE(observations::Array{T, 3}, rewards::Array{T, 3}) where T
    return AppleSiliconBanditMLE{T}(
        size(observations, 3),  # n_bandits
        size(observations, 2),  # n_arms
        size(observations, 1),  # n_samples
        MtlArray(observations),
        MtlArray(rewards)
    )
end

# Metal kernel for statistics computation
function compute_statistics_kernel!(successes, trials, observations, rewards)
    # Metal kernel function
    bandit_idx = thread_position_in_grid_1d()

    if bandit_idx <= length(successes)
        success_count = 0.0f0
        trial_count = 0.0f0

        # 各サンプルについて成功・試行回数を集計
        for sample in 1:size(observations, 1)
            if observations[sample, bandit_idx] > 0
                trial_count += 1.0f0
                if rewards[sample, bandit_idx] > 0.5f0
                    success_count += 1.0f0
                end
            end
        end

        successes[bandit_idx] = success_count
        trials[bandit_idx] = trial_count
    end

    return nothing
end

# MLE推定のメイン関数（Metal使用）
function mle_estimate_metal(bandit_mle::AppleSiliconBanditMLE{T}) where T
    n_bandits = bandit_mle.n_bandits
    n_arms = bandit_mle.n_arms

    # 結果格納用配列
    estimated_params = Metal.zeros(T, n_arms, n_bandits)

    # 各アームごとに並列でMLE推定
    for arm in 1:n_arms
        # 成功回数と試行回数を並列計算
        successes = Metal.zeros(T, n_bandits)
        trials = Metal.zeros(T, n_bandits)

        # GPU kernelで統計量計算
        kernel = Metal.@metal launch=false compute_statistics_kernel!(
            successes, trials,
            view(bandit_mle.observations, :, arm, :),
            view(bandit_mle.rewards, :, arm, :)
        )

        # Launch the kernel
        kernel(successes, trials,
               view(bandit_mle.observations, :, arm, :),
               view(bandit_mle.rewards, :, arm, :);
               threads=256, groups=cld(n_bandits, 256))

        # MLE推定値計算（単純な成功率）
        estimated_params[arm, :] = successes ./ max.(trials, 1.0f0)
    end

    return Array(estimated_params)
end

# より簡単な実装（Metal array operations使用）
function mle_estimate_simple(bandit_mle::AppleSiliconBanditMLE{T}) where T
    n_bandits = bandit_mle.n_bandits
    n_arms = bandit_mle.n_arms

    estimated_params = Metal.zeros(T, n_arms, n_bandits)

    for arm in 1:n_arms
        arm_observations = view(bandit_mle.observations, :, arm, :)
        arm_rewards = view(bandit_mle.rewards, :, arm, :)

        # 成功回数と試行回数を計算
        successes = Metal.sum(arm_observations .* arm_rewards, dims=1)
        trials = Metal.sum(arm_observations, dims=1)

        # MLE推定値
        estimated_params[arm, :] = dropdims(successes ./ max.(trials, 1.0f0), dims=1)
    end

    return Array(estimated_params)
end

# CPU版MLE実装（比較用）
function mle_estimate_cpu(observations::Array{T, 3}, rewards::Array{T, 3}) where T
    n_samples, n_arms, n_bandits = size(observations)
    estimated_params = zeros(T, n_arms, n_bandits)

    Threads.@threads for b in 1:n_bandits
        for a in 1:n_arms
            successes = 0
            trials = 0
            for s in 1:n_samples
                if observations[s, a, b] > 0
                    trials += 1
                    if rewards[s, a, b] > 0.5
                        successes += 1
                    end
                end
            end
            estimated_params[a, b] = trials > 0 ? successes / trials : 0.5
        end
    end

    return estimated_params
end

# テストデータ生成
function generate_test_data(n_bandits::Int, n_arms::Int, n_samples::Int;
                           observation_prob::Float32 = 0.8f0)
    # 真のパラメータ
    true_params = rand(Float32, n_arms, n_bandits)

    # 観測データ
    observations = zeros(Float32, n_samples, n_arms, n_bandits)
    rewards = zeros(Float32, n_samples, n_arms, n_bandits)

    for b in 1:n_bandits, a in 1:n_arms, s in 1:n_samples
        if rand() < observation_prob
            observations[s, a, b] = 1.0f0
            rewards[s, a, b] = rand() < true_params[a, b] ? 1.0f0 : 0.0f0
        end
    end

    return true_params, observations, rewards
end

# パフォーマンス比較
function benchmark_apple_silicon_mle(n_bandits_list = [100, 500, 1000])
    results = Dict()

    for n_bandits in n_bandits_list
        println("Testing with $n_bandits bandits...")

        # テストデータ生成
        n_arms = 5
        n_samples = 1000
        true_params, observations, rewards = generate_test_data(n_bandits, n_arms, n_samples)

        # CPU版
        cpu_time = @elapsed cpu_estimates = mle_estimate_cpu(observations, rewards)

        if Metal.functional()
            # Metal GPU版
            bandit_mle = AppleSiliconBanditMLE(observations, rewards)

            # シンプルMLE
            gpu_time = @elapsed gpu_estimates = mle_estimate_simple(bandit_mle)

            # 精度評価
            cpu_error = mean(abs.(cpu_estimates - true_params))
            gpu_error = mean(abs.(gpu_estimates - true_params))

            # 高速化率
            speedup = cpu_time / gpu_time

            results[n_bandits] = (
                cpu_time = cpu_time,
                gpu_time = gpu_time,
                cpu_error = cpu_error,
                gpu_error = gpu_error,
                speedup = speedup
            )

            println("  CPU:  Time=$(cpu_time:.3f)s, Error=$(cpu_error:.4f)")
            println("  GPU:  Time=$(gpu_time:.3f)s, Error=$(gpu_error:.4f), Speedup=$(speedup:.1f)x")
        else
            results[n_bandits] = (
                cpu_time = cpu_time,
                gpu_time = NaN,
                cpu_error = mean(abs.(cpu_estimates - true_params)),
                gpu_error = NaN,
                speedup = NaN
            )
            println("  CPU:  Time=$(cpu_time:.3f)s, Error=$(results[n_bandits].cpu_error:.4f)")
            println("  GPU:  Not available")
        end
    end

    return results
end

# 最もシンプルな使用API
function simple_bandit_mle(observations::Array{T, 3}, rewards::Array{T, 3}) where T
    """
    シンプルなバンディット問題MLE推定

    Parameters:
    - observations: (n_samples, n_arms, n_bandits) 観測フラグ
    - rewards: (n_samples, n_arms, n_bandits) 報酬

    Returns:
    - estimated_params: (n_arms, n_bandits) 推定パラメータ
    """
    if Metal.functional()
        # GPU版
        bandit_mle = AppleSiliconBanditMLE(observations, rewards)
        return mle_estimate_simple(bandit_mle)
    else
        # CPU版フォールバック
        return mle_estimate_cpu(observations, rewards)
    end
end

# 実際の使用例
function run_apple_silicon_benchmark()
    println("Apple Silicon GPU 並列化バンディット問題MLE")
    println("=" * 50)

    if Metal.functional()
        println("✓ Metal が利用可能です！")
        try
            println("Max buffer length: $(Metal.max_buffer_length())")
        catch
            println("Device info not available")
        end
        println()

        # 簡単なテスト
        println("簡単なテスト実行中...")
        true_params, observations, rewards = generate_test_data(100, 3, 500)

        # CPU vs GPU比較
        cpu_time = @elapsed cpu_result = mle_estimate_cpu(observations, rewards)
        gpu_time = @elapsed gpu_result = simple_bandit_mle(observations, rewards)

        println("CPU time: $(cpu_time:.3f)s")
        println("GPU time: $(gpu_time:.3f)s")
        println("Speedup: $(cpu_time/gpu_time:.1f)x")
        println("Error difference: $(mean(abs.(cpu_result - gpu_result)):.6f)")
        println()

        # ベンチマーク実行
        println("詳細ベンチマーク実行中...")
        results = benchmark_apple_silicon_mle([100, 500, 1000])

        # 結果の要約
        println("\n=== パフォーマンス要約 ===")
        for (n_bandits, result) in sort(collect(results))
            if !isnan(result.speedup)
                println("$n_bandits bandits: $(result.speedup:.1f)x speedup")
            end
        end

        return results
    else
        println("✗ Metal が利用できません")
        println("原因の可能性:")
        println("  - Rosetta環境で実行中")
        println("  - Intel Macで実行中")
        println("  - Metal.jl が正しくインストールされていない")
        println()
        println("CPU版のみでテスト実行...")

        # CPU版のみ実行
        results = benchmark_apple_silicon_mle([100, 500])
        return results
    end
end

# 使用例とテスト
function example_usage()
    println("使用例:")
    println("=" * 30)

    # 1. 簡単な例
    println("1. 基本的な使用法")
    n_bandits, n_arms, n_samples = 50, 3, 200
    true_params, observations, rewards = generate_test_data(n_bandits, n_arms, n_samples)

    println("  真のパラメータ例: $(true_params[1, 1:5])")

    estimated_params = simple_bandit_mle(observations, rewards)
    println("  推定パラメータ例: $(estimated_params[1, 1:5])")

    error = mean(abs.(estimated_params - true_params))
    println("  平均絶対誤差: $(error:.4f)")
    println()

    # 2. 実際のベンチマーク
    println("2. パフォーマンステスト")
    return run_apple_silicon_benchmark()
end

# Load the new advanced simulator
include("metal_bandit_simulator.jl")

# 実行
if abspath(PROGRAM_FILE) == @__FILE__
    example_usage()
end
#+end_
