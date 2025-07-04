#!/usr/bin/env julia

"""
Q学習パラメータ回復実験の小規模テスト
"""

using Pkg
Pkg.instantiate()

include("q_parameter_recovery.jl")

function test_small_experiment()
    println("🧪 小規模テスト実行中...")
    
    try
        # 小規模実験（10被験者）
        experiment = run_parameter_recovery_experiment(10, 4, 50; seed=42)
        
        println("✅ 小規模実験成功！")
        println("成功率: $(round(mean(experiment.estimation_success)*100, digits=1))%")
        
        # 分析
        stats = analyze_recovery_results(experiment)
        
        # 簡単な可視化テスト
        println("🎨 可視化テスト中...")
        fig = create_recovery_visualization(experiment)
        save("test_q_recovery_small.png", fig)
        
        println("✅ テスト完了！")
        return experiment
        
    catch e
        println("❌ テスト失敗: $e")
        rethrow(e)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    test_small_experiment()
end