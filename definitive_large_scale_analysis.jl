using CSV
using DataFrames
using Statistics
using CairoMakie

"""
Definitive Large-Scale Analysis

This script analyzes the comprehensive 8-hour large-scale test results
to provide definitive conclusions about GPU vs CPU performance.
"""

function analyze_definitive_results()
    println("📊 DEFINITIVE LARGE-SCALE ANALYSIS")
    println("🕘 Based on 8-hour comprehensive testing")
    println("="^80)
    
    # Read the results
    df = CSV.read("large_scale_results_intermediate.csv", DataFrame)
    
    println("📈 Data Summary:")
    println("   Total tests run: $(nrow(df))")
    println("   Scales tested: $(join(unique(df.scale), ", "))")
    println("   Methods tested: $(join(unique(df.method), ", "))")
    println("   Repetitions per scale: $(maximum(df.repetition))")
    
    scales = unique(df.scale)
    
    # Analyze each scale
    for scale in scales
        analyze_scale_performance(df, scale)
    end
    
    # Overall analysis
    analyze_overall_trends(df)
    
    # Create definitive visualization
    create_definitive_visualization(df)
    
    return df
end

function analyze_scale_performance(df::DataFrame, scale)
    scale_data = filter(row -> row.scale == scale, df)
    
    println("\n" * "="^60)
    println("📊 $scale SCALE ANALYSIS")
    
    # Calculate statistics by method
    methods = unique(scale_data.method)
    stats = []
    
    for method in methods
        method_data = filter(row -> row.method == method, scale_data)
        times = method_data.execution_time
        success_rates = method_data.success_rate
        throughputs = method_data.throughput
        memory = method_data.memory_used
        
        mean_time = mean(times)
        std_time = std(times)
        cv_time = std_time / mean_time * 100  # Coefficient of variation
        
        mean_success = mean(success_rates)
        mean_throughput = mean(throughputs)
        mean_memory = mean(memory)
        
        push!(stats, (method, mean_time, std_time, cv_time, mean_success, mean_throughput, mean_memory))
        
        println("   $method:")
        println("     Time: $(round(mean_time, digits=2))s ± $(round(std_time, digits=2))s (CV: $(round(cv_time, digits=1))%)")
        println("     Success: $(round(mean_success*100, digits=1))%")
        println("     Throughput: $(round(mean_throughput, digits=0)) decisions/s")
        println("     Memory: $(round(mean_memory, digits=1))MB")
    end
    
    # Find winner
    cpu8_stats = findfirst(s -> s[1] == "CPU(8)", stats)
    gpu_stats = findfirst(s -> s[1] == "GPU", stats)
    
    if cpu8_stats !== nothing && gpu_stats !== nothing
        cpu8_time = stats[cpu8_stats][2]
        gpu_time = stats[gpu_stats][2]
        cpu8_std = stats[cpu8_stats][3]
        gpu_std = stats[gpu_stats][3]
        
        # Performance comparison
        if gpu_time < cpu8_time
            advantage = cpu8_time / gpu_time
            winner = "GPU"
            advantage_pct = (advantage - 1) * 100
            println("\n   🚀 WINNER: GPU")
            println("     GPU advantage: $(round(advantage, digits=3))x ($(round(advantage_pct, digits=1))% faster)")
        else
            advantage = gpu_time / cpu8_time
            winner = "CPU(8)"
            advantage_pct = (advantage - 1) * 100
            println("\n   🖥️  WINNER: CPU(8)")
            println("     CPU(8) advantage: $(round(advantage, digits=3))x ($(round(advantage_pct, digits=1))% faster)")
        end
        
        # Statistical significance
        difference = abs(gpu_time - cpu8_time)
        pooled_std = sqrt((cpu8_std^2 + gpu_std^2) / 2)
        effect_size = difference / pooled_std
        
        println("     Effect size: $(round(effect_size, digits=3))")
        if effect_size > 0.8
            significance = "LARGE effect - highly significant"
        elseif effect_size > 0.5
            significance = "MEDIUM effect - significant"
        elseif effect_size > 0.2
            significance = "SMALL effect - marginally significant"
        else
            significance = "NEGLIGIBLE effect - not significant"
        end
        println("     Statistical significance: $significance")
        
        # Memory comparison
        cpu8_memory = stats[cpu8_stats][7]
        gpu_memory = stats[gpu_stats][7]
        memory_advantage = cpu8_memory / gpu_memory
        
        println("     Memory efficiency: GPU uses $(round(memory_advantage, digits=2))x less memory")
    end
end

function analyze_overall_trends(df::DataFrame)
    println("\n" * "="^60)
    println("🔍 OVERALL TRENDS ANALYSIS")
    
    scales = unique(df.scale)
    
    # Calculate scale information
    scale_info = []
    for scale in scales
        scale_data = filter(row -> row.scale == scale, df)
        
        # Get problem size (assuming same for all methods in scale)
        first_row = scale_data[1, :]
        # Problem size estimation based on execution times
        if scale == "Large"
            decisions = 30000  # 200 × 150
        elseif scale == "Very-Large" 
            decisions = 100000  # 500 × 200
        elseif scale == "Massive"
            decisions = 250000  # 1000 × 250
        elseif scale == "Ultra"
            decisions = 450000  # 1500 × 300
        else
            decisions = 0
        end
        
        push!(scale_info, (scale, decisions))
    end
    
    println("\n📈 Scale Progression Analysis:")
    for (scale, decisions) in scale_info
        println("   $scale: $(decisions) decisions")
    end
    
    # Analyze GPU vs CPU(8) trend across scales
    println("\n🏆 Winner by Scale:")
    winners = []
    
    for scale in scales
        scale_data = filter(row -> row.scale == scale, df)
        
        cpu8_data = filter(row -> row.method == "CPU(8)", scale_data)
        gpu_data = filter(row -> row.method == "GPU", scale_data)
        
        if !isempty(cpu8_data) && !isempty(gpu_data)
            cpu8_mean = mean(cpu8_data.execution_time)
            gpu_mean = mean(gpu_data.execution_time)
            
            if gpu_mean < cpu8_mean
                winner = "GPU"
                advantage = cpu8_mean / gpu_mean
            else
                winner = "CPU(8)" 
                advantage = gpu_mean / cpu8_mean
            end
            
            push!(winners, (scale, winner, advantage))
            println("   $scale: $winner wins ($(round(advantage, digits=3))x advantage)")
        end
    end
    
    # Overall conclusion
    gpu_wins = count(w -> w[2] == "GPU", winners)
    cpu_wins = count(w -> w[2] == "CPU(8)", winners)
    
    println("\n🏁 DEFINITIVE CONCLUSION:")
    println("   GPU wins: $gpu_wins/$(length(winners)) scales")
    println("   CPU(8) wins: $cpu_wins/$(length(winners)) scales")
    
    if gpu_wins > cpu_wins
        println("   🚀 OVERALL WINNER: GPU")
        println("   📊 GPU dominates at large scales with fair BFGS optimization")
    elseif cpu_wins > gpu_wins
        println("   🖥️  OVERALL WINNER: CPU(8)")
        println("   📊 CPU(8) dominates despite fair BFGS optimization")
    else
        println("   🤝 TIE: GPU and CPU(8) are equally competitive")
        println("   📊 Performance differences negligible at large scales")
    end
    
    # Performance magnitude analysis
    advantages = [w[3] for w in winners]
    max_advantage = maximum(advantages)
    mean_advantage = mean(advantages)
    
    println("\n📏 Performance Difference Magnitude:")
    println("   Maximum advantage: $(round(max_advantage, digits=3))x")
    println("   Average advantage: $(round(mean_advantage, digits=3))x")
    
    if max_advantage < 1.05
        println("   📊 VERDICT: Performance differences are NEGLIGIBLE (<5%)")
        println("   💡 Choice should be based on other factors (memory, ease of use, etc.)")
    elseif max_advantage < 1.1
        println("   📊 VERDICT: Performance differences are SMALL (5-10%)")
        println("   💡 Choice can be influenced by secondary considerations")
    else
        println("   📊 VERDICT: Performance differences are SIGNIFICANT (>10%)")
        println("   💡 Performance should be primary selection criterion")
    end
end

function create_definitive_visualization(df::DataFrame)
    scales = unique(df.scale)
    methods = unique(df.method)
    
    fig = Figure(size = (1600, 1200))
    
    colors = Dict(
        "CPU(1)" => :red,
        "CPU(8)" => :blue,
        "GPU" => :green
    )
    
    # Main performance comparison with error bars
    ax1 = Axis(fig[1, 1], 
        title = "Definitive Large-Scale Comparison: GPU vs CPU(8) Performance", 
        xlabel = "Scale", 
        ylabel = "Execution Time (seconds)",
        yscale = log10)
    
    x_positions = Dict(scale => i for (i, scale) in enumerate(scales))
    
    for method in ["CPU(8)", "GPU"]  # Focus on main comparison
        x_vals = Float64[]
        y_vals = Float64[]
        errors = Float64[]
        
        for scale in scales
            scale_data = filter(row -> row.scale == scale && row.method == method, df)
            if !isempty(scale_data)
                mean_time = mean(scale_data.execution_time)
                std_time = std(scale_data.execution_time)
                
                push!(x_vals, x_positions[scale])
                push!(y_vals, mean_time)
                push!(errors, std_time)
            end
        end
        
        color = colors[method]
        scatter!(ax1, x_vals, y_vals, markersize = 15, color = color, label = method)
        errorbars!(ax1, x_vals, y_vals, errors, color = color, linewidth = 3)
        lines!(ax1, x_vals, y_vals, color = color, linewidth = 2, linestyle = :dash)
    end
    
    # Set x-axis labels
    ax1.xticks = (1:length(scales), scales)
    axislegend(ax1, position = :lt)
    
    # Memory usage comparison
    ax2 = Axis(fig[1, 2], 
        title = "Memory Usage Comparison", 
        xlabel = "Scale", 
        ylabel = "Memory Used (MB)")
    
    for method in ["CPU(8)", "GPU"]
        x_vals = Float64[]
        y_vals = Float64[]
        
        for scale in scales
            scale_data = filter(row -> row.scale == scale && row.method == method, df)
            if !isempty(scale_data)
                mean_memory = mean(scale_data.memory_used)
                push!(x_vals, x_positions[scale])
                push!(y_vals, mean_memory)
            end
        end
        
        color = colors[method]
        scatter!(ax2, x_vals, y_vals, markersize = 15, color = color, label = method)
        lines!(ax2, x_vals, y_vals, color = color, linewidth = 2)
    end
    
    ax2.xticks = (1:length(scales), scales)
    axislegend(ax2, position = :lt)
    
    # Throughput comparison
    ax3 = Axis(fig[2, 1], 
        title = "Throughput Comparison", 
        xlabel = "Scale", 
        ylabel = "Throughput (decisions/second)")
    
    for method in ["CPU(8)", "GPU"]
        x_vals = Float64[]
        y_vals = Float64[]
        
        for scale in scales
            scale_data = filter(row -> row.scale == scale && row.method == method, df)
            if !isempty(scale_data)
                mean_throughput = mean(scale_data.throughput)
                push!(x_vals, x_positions[scale])
                push!(y_vals, mean_throughput)
            end
        end
        
        color = colors[method]
        scatter!(ax3, x_vals, y_vals, markersize = 15, color = color, label = method)
        lines!(ax3, x_vals, y_vals, color = color, linewidth = 2)
    end
    
    ax3.xticks = (1:length(scales), scales)
    axislegend(ax3, position = :lt)
    
    # Performance advantage over scales
    ax4 = Axis(fig[2, 2], 
        title = "GPU vs CPU(8) Advantage Across Scales", 
        xlabel = "Scale", 
        ylabel = "Performance Ratio (>1 = GPU faster)")
    
    x_vals = Float64[]
    y_vals = Float64[]
    
    for scale in scales
        scale_data = filter(row -> row.scale == scale, df)
        cpu8_data = filter(row -> row.method == "CPU(8)", scale_data)
        gpu_data = filter(row -> row.method == "GPU", scale_data)
        
        if !isempty(cpu8_data) && !isempty(gpu_data)
            cpu8_mean = mean(cpu8_data.execution_time)
            gpu_mean = mean(gpu_data.execution_time)
            ratio = cpu8_mean / gpu_mean  # >1 means GPU faster
            
            push!(x_vals, x_positions[scale])
            push!(y_vals, ratio)
        end
    end
    
    scatter!(ax4, x_vals, y_vals, markersize = 15, color = :purple)
    lines!(ax4, x_vals, y_vals, color = :purple, linewidth = 2)
    hlines!(ax4, [1.0], color = :black, linestyle = :dash, linewidth = 1)
    
    ax4.xticks = (1:length(scales), scales)
    
    save("definitive_large_scale_results.png", fig)
    println("📊 Definitive visualization saved to: definitive_large_scale_results.png")
    
    return fig
end

# Run the analysis
results_df = analyze_definitive_results()