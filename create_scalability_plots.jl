using CairoMakie
using Statistics

# Scalability test results data
scales = ["Small", "Medium", "Large", "Extra-Large"]
decisions = [400_000, 1_800_000, 6_000_000, 16_000_000]

# Execution times (seconds)
cpu1_times = [17.59, 93.46, 253.09, 581.53]
cpu8_times = [9.83, 50.03, 83.32, 181.24]
gpu_times = [8.54, 8.24, 7.96, 8.14]

# Memory usage (MB)
cpu1_memory = [32.6, 77.2, 72.2, 103.0]
cpu8_memory = [18.3, 58.8, 88.8, 140.2]
gpu_memory = [22.9, 32.1, 30.6, 35.6]

# Throughput (decisions/second)
cpu1_throughput = [5685, 4815, 5927, 6878]
cpu8_throughput = [10169, 8995, 18002, 22070]
gpu_throughput = [11704, 54644, 188391, 491289]

function create_comprehensive_scalability_plots()
    # Create the main figure
    fig = Figure(size = (1600, 1200), fontsize = 14)
    
    # Colors for consistency
    cpu1_color = :red
    cpu8_color = :blue
    gpu_color = :green
    
    # Plot 1: Execution Time vs Scale (Log scale)
    ax1 = Axis(fig[1, 1], 
        title = "Execution Time by Scale", 
        xlabel = "Scale", 
        ylabel = "Execution Time (seconds)",
        yscale = log10,
        xticks = (1:4, scales))
    
    lines!(ax1, 1:4, cpu1_times, color = cpu1_color, linewidth = 3, label = "CPU (1-thread)")
    scatter!(ax1, 1:4, cpu1_times, color = cpu1_color, markersize = 12)
    
    lines!(ax1, 1:4, cpu8_times, color = cpu8_color, linewidth = 3, label = "CPU (8-threads)")
    scatter!(ax1, 1:4, cpu8_times, color = cpu8_color, markersize = 12)
    
    lines!(ax1, 1:4, gpu_times, color = gpu_color, linewidth = 3, label = "GPU")
    scatter!(ax1, 1:4, gpu_times, color = gpu_color, markersize = 12)
    
    axislegend(ax1, position = :lt)
    
    # Plot 2: Throughput vs Scale (Log scale)
    ax2 = Axis(fig[1, 2], 
        title = "Throughput by Scale", 
        xlabel = "Scale", 
        ylabel = "Throughput (decisions/second)",
        yscale = log10,
        xticks = (1:4, scales))
    
    lines!(ax2, 1:4, cpu1_throughput, color = cpu1_color, linewidth = 3, label = "CPU (1-thread)")
    scatter!(ax2, 1:4, cpu1_throughput, color = cpu1_color, markersize = 12)
    
    lines!(ax2, 1:4, cpu8_throughput, color = cpu8_color, linewidth = 3, label = "CPU (8-threads)")
    scatter!(ax2, 1:4, cpu8_throughput, color = cpu8_color, markersize = 12)
    
    lines!(ax2, 1:4, gpu_throughput, color = gpu_color, linewidth = 3, label = "GPU")
    scatter!(ax2, 1:4, gpu_throughput, color = gpu_color, markersize = 12)
    
    axislegend(ax2, position = :lt)
    
    # Plot 3: Memory Usage vs Scale
    ax3 = Axis(fig[2, 1], 
        title = "Memory Usage by Scale", 
        xlabel = "Scale", 
        ylabel = "Memory Used (MB)",
        xticks = (1:4, scales))
    
    lines!(ax3, 1:4, cpu1_memory, color = cpu1_color, linewidth = 3, label = "CPU (1-thread)")
    scatter!(ax3, 1:4, cpu1_memory, color = cpu1_color, markersize = 12)
    
    lines!(ax3, 1:4, cpu8_memory, color = cpu8_color, linewidth = 3, label = "CPU (8-threads)")
    scatter!(ax3, 1:4, cpu8_memory, color = cpu8_color, markersize = 12)
    
    lines!(ax3, 1:4, gpu_memory, color = gpu_color, linewidth = 3, label = "GPU")
    scatter!(ax3, 1:4, gpu_memory, color = gpu_color, markersize = 12)
    
    axislegend(ax3, position = :lt)
    
    # Plot 4: GPU Speedup vs Scale
    ax4 = Axis(fig[2, 2], 
        title = "GPU Speedup vs CPU Methods", 
        xlabel = "Scale", 
        ylabel = "Speedup Factor",
        yscale = log10,
        xticks = (1:4, scales))
    
    gpu_vs_cpu1 = cpu1_times ./ gpu_times
    gpu_vs_cpu8 = cpu8_times ./ gpu_times
    
    lines!(ax4, 1:4, gpu_vs_cpu1, color = cpu1_color, linewidth = 3, label = "GPU vs CPU(1)")
    scatter!(ax4, 1:4, gpu_vs_cpu1, color = cpu1_color, markersize = 12)
    
    lines!(ax4, 1:4, gpu_vs_cpu8, color = cpu8_color, linewidth = 3, label = "GPU vs CPU(8)")
    scatter!(ax4, 1:4, gpu_vs_cpu8, color = cpu8_color, markersize = 12)
    
    # Add speedup annotations
    for i in 1:4
        text!(ax4, i, gpu_vs_cpu1[i] * 1.1, text = "$(round(gpu_vs_cpu1[i], digits=1))x", 
              align = (:center, :bottom), fontsize = 12, color = cpu1_color)
        text!(ax4, i, gpu_vs_cpu8[i] * 0.9, text = "$(round(gpu_vs_cpu8[i], digits=1))x", 
              align = (:center, :top), fontsize = 12, color = cpu8_color)
    end
    
    axislegend(ax4, position = :lt)
    
    # Save the plot
    save("comprehensive_scalability_results.png", fig)
    println("📊 Comprehensive scalability plot saved to: comprehensive_scalability_results.png")
    
    return fig
end

function create_summary_table_plot()
    # Create summary table visualization
    fig_table = Figure(size = (1400, 800), fontsize = 12)
    
    # Prepare data for table
    scale_labels = ["Small\n(400K decisions)", "Medium\n(1.8M decisions)", 
                   "Large\n(6M decisions)", "Extra-Large\n(16M decisions)"]
    
    # Create table data
    table_data = [
        scale_labels,
        ["$(cpu1_times[i])s" for i in 1:4],
        ["$(cpu8_times[i])s" for i in 1:4], 
        ["$(gpu_times[i])s" for i in 1:4],
        ["$(round(cpu1_times[i]/gpu_times[i], digits=1))x" for i in 1:4],
        ["$(round(cpu8_times[i]/gpu_times[i], digits=1))x" for i in 1:4]
    ]
    
    headers = ["Scale", "CPU(1)", "CPU(8)", "GPU", "GPU vs CPU(1)", "GPU vs CPU(8)"]
    
    # Create table plot
    ax_table = Axis(fig_table[1, 1], 
        title = "Scalability Test Results Summary",
        aspect = DataAspect())
    
    # Hide axes
    hidedecorations!(ax_table)
    hidespines!(ax_table)
    
    # Draw table
    for (i, header) in enumerate(headers)
        text!(ax_table, i, 5, text = header, align = (:center, :center), 
              fontsize = 16)
    end
    
    for i in 1:4
        for j in 1:6
            color = j > 3 ? :lightgreen : :white
            text!(ax_table, j, 4-i, text = table_data[j][i], 
                  align = (:center, :center), fontsize = 12,
                  color = j > 3 ? :darkgreen : :black)
        end
    end
    
    xlims!(ax_table, 0.5, 6.5)
    ylims!(ax_table, 0, 5.5)
    
    save("scalability_summary_table.png", fig_table)
    println("📊 Summary table saved to: scalability_summary_table.png")
    
    return fig_table
end

# Run the plotting functions
println("Creating comprehensive scalability visualizations...")
fig1 = create_comprehensive_scalability_plots()
fig2 = create_summary_table_plot()

println("✅ All visualizations created successfully!")