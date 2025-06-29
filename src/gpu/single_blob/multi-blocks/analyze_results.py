#!/usr/bin/env python3
"""
Advanced analysis of CUDA flood fill benchmark results.
Provides detailed insights and recommendations for optimal configurations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_and_analyze_results():
    """Load benchmark results and provide comprehensive analysis"""
    
    # Load the CSV file
    results_file = './benchmark_results/cuda_flood_fill_benchmark.csv'
    df = pd.read_csv(results_file)
    
    print("🔍 CUDA FLOOD FILL BENCHMARK ANALYSIS")
    print("=" * 60)
    print(f"📊 Total configurations tested: {len(df)}")
    print(f"📈 Scene size: 2000x2000 pixels (4M pixels total)")
    print(f"🎯 Target blob: 1000x1000 pixels (1M pixels)")
    print()
    
    # Key metrics analysis
    print("📊 PERFORMANCE METRICS OVERVIEW:")
    print(f"   • Pixels/second range: {df['pixels_per_second'].min():,.0f} - {df['pixels_per_second'].max():,.0f}")
    print(f"   • Execution time range: {df['execution_time_ms'].min():.1f} - {df['execution_time_ms'].max():.1f} ms")
    print(f"   • Block utilization range: {df['block_utilization_pct'].min():.1f} - {df['block_utilization_pct'].max():.1f}%")
    print(f"   • Thread utilization range: {df['thread_utilization_pct'].min():.1f} - {df['thread_utilization_pct'].max():.1f}%")
    print()
    
    # TOP CONFIGURATIONS
    print("🏆 TOP PERFORMING CONFIGURATIONS:")
    print("-" * 50)
    
    # 1. Fastest overall
    fastest = df.loc[df['pixels_per_second'].idxmax()]
    print(f"\n🚀 FASTEST CONFIGURATION:")
    print(f"   • {fastest['blocks_per_grid']:.0f} blocks × {fastest['threads_per_block']:.0f} threads, chunk={fastest['chunk_size']:.0f}")
    print(f"   • {fastest['pixels_per_second']:,.0f} pixels/second")
    print(f"   • {fastest['execution_time_ms']:.2f}ms execution time")
    print(f"   • GPU utilization: {fastest['block_utilization_pct']:.1f}% blocks, {fastest['thread_utilization_pct']:.1f}% threads")
    print(f"   • {fastest['time_per_iteration_ms']:.3f}ms per iteration")
    
    # 2. Best utilization
    best_util = df.loc[df['block_utilization_pct'].idxmax()]
    print(f"\n🎯 BEST GPU UTILIZATION:")
    print(f"   • {best_util['blocks_per_grid']:.0f} blocks × {best_util['threads_per_block']:.0f} threads, chunk={best_util['chunk_size']:.0f}")
    print(f"   • {best_util['block_utilization_pct']:.1f}% blocks, {best_util['thread_utilization_pct']:.1f}% threads")
    print(f"   • {best_util['pixels_per_second']:,.0f} pixels/second")
    print(f"   • {best_util['execution_time_ms']:.2f}ms execution time")
    
    # 3. Best efficiency (speed × utilization)
    df['efficiency_score'] = df['pixels_per_second'] * df['block_utilization_pct'] / 100
    best_efficiency = df.loc[df['efficiency_score'].idxmax()]
    print(f"\n⚖️  BEST OVERALL EFFICIENCY:")
    print(f"   • {best_efficiency['blocks_per_grid']:.0f} blocks × {best_efficiency['threads_per_block']:.0f} threads, chunk={best_efficiency['chunk_size']:.0f}")
    print(f"   • {best_efficiency['pixels_per_second']:,.0f} pixels/second")
    print(f"   • {best_efficiency['block_utilization_pct']:.1f}% blocks, {best_efficiency['thread_utilization_pct']:.1f}% threads")
    print(f"   • Efficiency score: {best_efficiency['efficiency_score']:,.0f}")
    
    # PARAMETER ANALYSIS
    print(f"\n📈 PARAMETER IMPACT ANALYSIS:")
    print("-" * 40)
    
    # Blocks per grid analysis
    blocks_analysis = df.groupby('blocks_per_grid').agg({
        'pixels_per_second': ['mean', 'max', 'std'],
        'block_utilization_pct': 'mean',
        'execution_time_ms': 'mean'
    }).round(0)
    
    print(f"\n📦 BLOCKS PER GRID ANALYSIS:")
    for blocks in sorted(df['blocks_per_grid'].unique()):
        subset = df[df['blocks_per_grid'] == blocks]
        avg_speed = subset['pixels_per_second'].mean()
        max_speed = subset['pixels_per_second'].max()
        avg_util = subset['block_utilization_pct'].mean()
        print(f"   • {blocks:.0f} blocks: {avg_speed:,.0f} avg, {max_speed:,.0f} max pixels/s, {avg_util:.1f}% avg util")
    
    # Threads per block analysis
    print(f"\n🧵 THREADS PER BLOCK ANALYSIS:")
    for threads in sorted(df['threads_per_block'].unique()):
        subset = df[df['threads_per_block'] == threads]
        avg_speed = subset['pixels_per_second'].mean()
        max_speed = subset['pixels_per_second'].max()
        avg_util = subset['thread_utilization_pct'].mean()
        print(f"   • {threads:.0f} threads: {avg_speed:,.0f} avg, {max_speed:,.0f} max pixels/s, {avg_util:.1f}% avg util")
    
    # Chunk size analysis
    print(f"\n📐 CHUNK SIZE ANALYSIS:")
    for chunk in sorted(df['chunk_size'].unique()):
        subset = df[df['chunk_size'] == chunk]
        avg_speed = subset['pixels_per_second'].mean()
        max_speed = subset['pixels_per_second'].max()
        avg_time = subset['execution_time_ms'].mean()
        print(f"   • Chunk {chunk:.0f}: {avg_speed:,.0f} avg, {max_speed:,.0f} max pixels/s, {avg_time:.1f}ms avg time")
    
    # RECOMMENDATIONS
    print(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
    print("=" * 50)
    
    # Find optimal for different use cases
    top_configs = df.nlargest(5, 'pixels_per_second')
    high_util_configs = df[df['block_utilization_pct'] >= 80].nlargest(3, 'pixels_per_second')
    
    print(f"\n🎯 FOR MAXIMUM SPEED:")
    for i, (_, config) in enumerate(top_configs.iterrows(), 1):
        print(f"   {i}. {config['blocks_per_grid']:.0f}×{config['threads_per_block']:.0f}, chunk={config['chunk_size']:.0f} → {config['pixels_per_second']:,.0f} pixels/s")
    
    print(f"\n⚡ FOR HIGH UTILIZATION + SPEED:")
    for i, (_, config) in enumerate(high_util_configs.iterrows(), 1):
        print(f"   {i}. {config['blocks_per_grid']:.0f}×{config['threads_per_block']:.0f}, chunk={config['chunk_size']:.0f} → {config['pixels_per_second']:,.0f} pixels/s, {config['block_utilization_pct']:.1f}% util")
    
    # Hardware utilization insights
    print(f"\n🔧 HARDWARE UTILIZATION INSIGHTS:")
    print(f"   • RTX 4060 has 24 SMs - optimal blocks: 24, 32, 40, 48")
    print(f"   • Max 1024 threads/block - tested: 64, 128, 256, 512")
    print(f"   • Warp size: 32 threads - thread counts should be multiples")
    print(f"   • Best chunk sizes: 64-128 for large scenes")
    
    # Memory analysis
    total_pixels = df.iloc[0]['pixels_processed'] if not df.empty else 0
    if total_pixels > 0:
        memory_per_pixel = 128000008 / total_pixels  # From debug output
        print(f"\n💾 MEMORY EFFICIENCY:")
        print(f"   • Queue memory: 128MB for 16M queue slots")
        print(f"   • Scene memory: ~{total_pixels * 4 / 1024**2:.1f}MB (image + visited)")
        print(f"   • Effective memory per pixel: {memory_per_pixel:.2f} bytes")
    
    return df

def create_visualizations(df):
    """Create performance visualization charts"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create output directory
        Path('./benchmark_results/plots').mkdir(exist_ok=True)
        
        # 1. Performance heatmap by blocks and threads
        plt.figure(figsize=(12, 8))
        pivot_data = df.pivot_table(values='pixels_per_second', 
                                   index='threads_per_block', 
                                   columns='blocks_per_grid', 
                                   aggfunc='mean')
        
        sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='viridis', 
                   cbar_kws={'label': 'Pixels/Second'})
        plt.title('Performance Heatmap: Pixels/Second by Configuration')
        plt.xlabel('Blocks per Grid')
        plt.ylabel('Threads per Block')
        plt.tight_layout()
        plt.savefig('./benchmark_results/plots/performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Utilization scatter plot
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(df['block_utilization_pct'], df['pixels_per_second'], 
                            c=df['chunk_size'], s=60, alpha=0.7, cmap='plasma')
        plt.colorbar(scatter, label='Chunk Size')
        plt.xlabel('Block Utilization (%)')
        plt.ylabel('Pixels per Second')
        plt.title('Performance vs GPU Utilization')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('./benchmark_results/plots/utilization_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Parameter analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Blocks analysis
        blocks_perf = df.groupby('blocks_per_grid')['pixels_per_second'].mean()
        axes[0,0].bar(blocks_perf.index, blocks_perf.values)
        axes[0,0].set_title('Average Performance by Blocks per Grid')
        axes[0,0].set_ylabel('Pixels/Second')
        axes[0,0].set_xlabel('Blocks per Grid')
        
        # Threads analysis
        threads_perf = df.groupby('threads_per_block')['pixels_per_second'].mean()
        axes[0,1].bar(threads_perf.index, threads_perf.values)
        axes[0,1].set_title('Average Performance by Threads per Block')
        axes[0,1].set_ylabel('Pixels/Second')
        axes[0,1].set_xlabel('Threads per Block')
        
        # Chunk analysis
        chunk_perf = df.groupby('chunk_size')['pixels_per_second'].mean()
        axes[1,0].bar(chunk_perf.index, chunk_perf.values)
        axes[1,0].set_title('Average Performance by Chunk Size')
        axes[1,0].set_ylabel('Pixels/Second')
        axes[1,0].set_xlabel('Chunk Size')
        
        # Execution time distribution
        axes[1,1].hist(df['execution_time_ms'], bins=20, alpha=0.7)
        axes[1,1].set_title('Execution Time Distribution')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_xlabel('Execution Time (ms)')
        
        plt.tight_layout()
        plt.savefig('./benchmark_results/plots/parameter_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Visualization plots saved to ./benchmark_results/plots/")
        
    except ImportError:
        print("⚠️  Matplotlib/Seaborn not available for visualizations")

def main():
    """Main analysis function"""
    print("🔍 Starting comprehensive benchmark analysis...")
    
    try:
        df = load_and_analyze_results()
        
        # Create visualizations if possible
        create_visualizations(df)
        
        # Save detailed analysis
        analysis_file = './benchmark_results/detailed_analysis.txt'
        with open(analysis_file, 'w') as f:
            f.write("CUDA Flood Fill Benchmark Analysis\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total configurations: {len(df)}\n")
            f.write(f"Best performance: {df['pixels_per_second'].max():,.0f} pixels/s\n")
            f.write(f"Best utilization: {df['block_utilization_pct'].max():.1f}%\n")
            
            # Top configurations
            f.write("\nTop 5 configurations by speed:\n")
            top_5 = df.nlargest(5, 'pixels_per_second')
            for i, (_, config) in enumerate(top_5.iterrows(), 1):
                f.write(f"{i}. {config['blocks_per_grid']:.0f}×{config['threads_per_block']:.0f}, chunk={config['chunk_size']:.0f} → {config['pixels_per_second']:,.0f} pixels/s\n")
        
        print(f"💾 Detailed analysis saved to: {analysis_file}")
        print("✅ Analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
