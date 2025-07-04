#!/usr/bin/env julia

"""
Q学習パラメータ回復実験実行スクリプト

このスクリプトは以下を実行します：
1. 1000組の(α, β)パラメータをランダムサンプリング
2. 各パラメータでQ学習バンディットシミュレーション実行
3. 行動データからパラメータを逆推定（MLE）
4. CairoMakieによる包括的可視化
5. 結果をCSVファイルに保存
"""

using Pkg

# 必要なパッケージのインストール
println("📦 依存パッケージをインストール中...")
Pkg.instantiate()

# Q学習パラメータ回復実験をロード
include("q_parameter_recovery.jl")

function main()
    try
        println("🧠 Q学習パラメータ回復実験を開始します")
        println("実験設定:")
        println("  - 被験者数（パラメータ組み合わせ）: 1000")
        println("  - 学習率α範囲: [0, 1]")
        println("  - 逆温度β範囲: [0, 10]")
        println("  - バンディットアーム数: 4")
        println("  - 試行数: 200")
        println()
        
        # メイン実験実行
        experiment, fig, df = main_parameter_recovery_experiment()
        
        println("\n📊 実験結果サマリー:")
        println("  - 推定成功率: $(round(mean(experiment.estimation_success)*100, digits=1))%")
        println("  - 成功データ数: $(sum(experiment.estimation_success))")
        
        # 成功したデータの統計
        if sum(experiment.estimation_success) > 0
            success_idx = experiment.estimation_success
            α_correlation = cor(experiment.true_alpha[success_idx], 
                              experiment.estimated_alpha[success_idx])
            β_correlation = cor(experiment.true_beta[success_idx], 
                              experiment.estimated_beta[success_idx])
            
            println("  - α回復相関: $(round(α_correlation, digits=3))")
            println("  - β回復相関: $(round(β_correlation, digits=3))")
        end
        
        println("\n📁 出力ファイル:")
        println("  - 可視化: q_parameter_recovery_visualization.png")
        println("  - データ: q_parameter_recovery_results.csv")
        
        println("\n✅ 実験が正常に完了しました！")
        
        return experiment, fig, df
        
    catch e
        println("❌ 実験中にエラーが発生しました:")
        println(e)
        rethrow(e)
    end
end

# スクリプトが直接実行された場合のみ実行
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end