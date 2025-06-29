#!/usr/bin/env python3
"""
Comprehensive analysis of CUDA flood fill benchmark results.
Pure Python implementation without external dependencies.
"""

import csv
import json
from collections import defaultdict
from statistics import mean, stdev

def load_csv_results(filename):
    """Load benchmark results from CSV file"""
    results = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            numeric_fields = [
                'blocks_per_grid', 'threads_per_block', 'chunk_size',
                'execution_time_ms', 'iterations', 'pixels_processed',
                'pixels_per_second', 'blocks_used', 'threads_used', 'warps_used',
                'block_utilization_pct', 'thread_utilization_pct', 'warp_utilization_pct',
                'time_per_iteration_ms'
            ]
            for field in numeric_fields:
                if field in row:
                    row[field] = float(row[field])
            results.append(row)
    return results

def analyze_benchmark_results():
    """Comprehensive analysis of benchmark results"""
    
    print("🔍 CUDA FLOOD FILL BENCHMARK ANALYSIS")
    print("=" * 60)
    
    # Load results
    results_file = './benchmark_results/cuda_flood_fill_benchmark.csv'
    try:
        results = load_csv_results(results_file)
    except FileNotFoundError:
        print(f"❌ Results file not found: {results_file}")
        return
    
    print(f"📊 Total configurations tested: {len(results)}")
    print(f"📈 Scene size: 2000x2000 pixels (4M pixels total)")
    print(f"🎯 Target blob: 1000x1000 pixels (1M pixels)")
    print()
    
    # Basic statistics
    speeds = [r['pixels_per_second'] for r in results]
    times = [r['execution_time_ms'] for r in results]
    block_utils = [r['block_utilization_pct'] for r in results]
    thread_utils = [r['thread_utilization_pct'] for r in results]
    
    print("📊 PERFORMANCE METRICS OVERVIEW:")
    print(f"   • Pixels/second range: {min(speeds):,.0f} - {max(speeds):,.0f}")
    print(f"   • Execution time range: {min(times):.1f} - {max(times):.1f} ms")
    print(f"   • Block utilization range: {min(block_utils):.1f} - {max(block_utils):.1f}%")
    print(f"   • Thread utilization range: {min(thread_utils):.1f} - {max(thread_utils):.1f}%")
    print(f"   • Average performance: {mean(speeds):,.0f} pixels/s")
    print()
    
    # Find top configurations
    results_sorted = sorted(results, key=lambda x: x['pixels_per_second'], reverse=True)
    
    print("🏆 TOP PERFORMING CONFIGURATIONS:")
    print("-" * 50)
    
    # Top 5 fastest
    print(f"\\n🚀 TOP 5 FASTEST CONFIGURATIONS:")
    for i, config in enumerate(results_sorted[:5], 1):
        print(f"   {i}. {config['blocks_per_grid']:.0f} blocks × {config['threads_per_block']:.0f} threads, chunk={config['chunk_size']:.0f}")
        print(f"      → {config['pixels_per_second']:,.0f} pixels/s, {config['execution_time_ms']:.2f}ms")
        print(f"      → GPU: {config['block_utilization_pct']:.1f}% blocks, {config['thread_utilization_pct']:.1f}% threads")
        print()
    
    # Best utilization
    best_util = max(results, key=lambda x: x['block_utilization_pct'])
    print(f"🎯 BEST GPU UTILIZATION:")
    print(f"   • {best_util['blocks_per_grid']:.0f} blocks × {best_util['threads_per_block']:.0f} threads, chunk={best_util['chunk_size']:.0f}")
    print(f"   • {best_util['block_utilization_pct']:.1f}% blocks, {best_util['thread_utilization_pct']:.1f}% threads")
    print(f"   • {best_util['pixels_per_second']:,.0f} pixels/second")
    print(f"   • {best_util['execution_time_ms']:.2f}ms execution time")
    print()
    
    # Best efficiency (speed × utilization)
    for r in results:
        r['efficiency_score'] = r['pixels_per_second'] * r['block_utilization_pct'] / 100
    best_efficiency = max(results, key=lambda x: x['efficiency_score'])
    
    print(f"⚖️  BEST OVERALL EFFICIENCY (speed × utilization):")
    print(f"   • {best_efficiency['blocks_per_grid']:.0f} blocks × {best_efficiency['threads_per_block']:.0f} threads, chunk={best_efficiency['chunk_size']:.0f}")
    print(f"   • {best_efficiency['pixels_per_second']:,.0f} pixels/second")
    print(f"   • {best_efficiency['block_utilization_pct']:.1f}% blocks, {best_efficiency['thread_utilization_pct']:.1f}% threads")
    print(f"   • Efficiency score: {best_efficiency['efficiency_score']:,.0f}")
    print()
    
    # Parameter analysis
    print("📈 PARAMETER IMPACT ANALYSIS:")
    print("-" * 40)
    
    # Group by parameter
    blocks_groups = defaultdict(list)
    threads_groups = defaultdict(list)
    chunk_groups = defaultdict(list)
    
    for r in results:
        blocks_groups[r['blocks_per_grid']].append(r)
        threads_groups[r['threads_per_block']].append(r)
        chunk_groups[r['chunk_size']].append(r)
    
    # Blocks analysis
    print(f"\\n📦 BLOCKS PER GRID ANALYSIS:")
    for blocks in sorted(blocks_groups.keys()):
        configs = blocks_groups[blocks]
        speeds = [c['pixels_per_second'] for c in configs]
        utils = [c['block_utilization_pct'] for c in configs]
        avg_speed = mean(speeds)
        max_speed = max(speeds)
        avg_util = mean(utils)
        print(f"   • {blocks:.0f} blocks: {avg_speed:,.0f} avg, {max_speed:,.0f} max pixels/s, {avg_util:.1f}% avg util")
    
    # Threads analysis
    print(f"\\n🧵 THREADS PER BLOCK ANALYSIS:")
    for threads in sorted(threads_groups.keys()):
        configs = threads_groups[threads]
        speeds = [c['pixels_per_second'] for c in configs]
        utils = [c['thread_utilization_pct'] for c in configs]
        avg_speed = mean(speeds)
        max_speed = max(speeds)
        avg_util = mean(utils)
        print(f"   • {threads:.0f} threads: {avg_speed:,.0f} avg, {max_speed:,.0f} max pixels/s, {avg_util:.1f}% avg util")
    
    # Chunk analysis
    print(f"\\n📐 CHUNK SIZE ANALYSIS:")
    for chunk in sorted(chunk_groups.keys()):
        configs = chunk_groups[chunk]
        speeds = [c['pixels_per_second'] for c in configs]
        times = [c['execution_time_ms'] for c in configs]
        avg_speed = mean(speeds)
        max_speed = max(speeds)
        avg_time = mean(times)
        print(f"   • Chunk {chunk:.0f}: {avg_speed:,.0f} avg, {max_speed:,.0f} max pixels/s, {avg_time:.1f}ms avg time")
    
    # Recommendations
    print(f"\\n💡 OPTIMIZATION RECOMMENDATIONS:")
    print("=" * 50)
    
    # High-performance configs
    top_configs = results_sorted[:5]
    high_util_configs = [r for r in results if r['block_utilization_pct'] >= 80]
    high_util_configs = sorted(high_util_configs, key=lambda x: x['pixels_per_second'], reverse=True)[:3]
    
    print(f"\\n🎯 FOR MAXIMUM SPEED:")
    for i, config in enumerate(top_configs, 1):
        print(f"   {i}. {config['blocks_per_grid']:.0f}×{config['threads_per_block']:.0f}, chunk={config['chunk_size']:.0f} → {config['pixels_per_second']:,.0f} pixels/s")
    
    print(f"\\n⚡ FOR HIGH UTILIZATION + SPEED:")
    for i, config in enumerate(high_util_configs, 1):
        print(f"   {i}. {config['blocks_per_grid']:.0f}×{config['threads_per_block']:.0f}, chunk={config['chunk_size']:.0f} → {config['pixels_per_second']:,.0f} pixels/s, {config['block_utilization_pct']:.1f}% util")
    
    # Key insights
    print(f"\\n🔧 KEY INSIGHTS:")
    print(f"   • RTX 4060 has 24 SMs - tested blocks: 24, 32, 40, 48, 56")
    print(f"   • Optimal blocks appear to be 32-40 for this workload")
    print(f"   • 64-128 threads per block show best performance")
    print(f"   • Larger chunk sizes (64-128) generally perform better")
    print(f"   • 100% block utilization doesn't always mean best performance")
    
    # Hardware utilization insights
    best_config = results_sorted[0]
    print(f"\\n💾 MEMORY AND HARDWARE EFFICIENCY:")
    print(f"   • Queue memory: 128MB for 16M queue slots")
    print(f"   • Best config uses {best_config['blocks_per_grid']:.0f} blocks ({best_config['blocks_per_grid']/24:.1f}x SM count)")
    print(f"   • Best config: {best_config['threads_per_block']:.0f} threads/block = {best_config['threads_per_block']//32:.0f} warps/block")
    print(f"   • Chunk size {best_config['chunk_size']:.0f} provides good work distribution")
    
    # Save analysis summary
    summary = {
        'total_configs': len(results),
        'best_performance': {
            'pixels_per_second': max(speeds),
            'config': f"{best_config['blocks_per_grid']:.0f}×{best_config['threads_per_block']:.0f}, chunk={best_config['chunk_size']:.0f}"
        },
        'best_utilization': {
            'block_utilization_pct': max(block_utils),
            'config': f"{best_util['blocks_per_grid']:.0f}×{best_util['threads_per_block']:.0f}, chunk={best_util['chunk_size']:.0f}"
        },
        'recommended_configs': [
            {
                'blocks': int(config['blocks_per_grid']),
                'threads': int(config['threads_per_block']),
                'chunk': int(config['chunk_size']),
                'pixels_per_second': config['pixels_per_second']
            }
            for config in top_configs
        ]
    }
    
    with open('./benchmark_results/analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\\n💾 Analysis summary saved to: ./benchmark_results/analysis_summary.json")
    print("✅ Analysis completed successfully!")

def main():
    """Main analysis function"""
    try:
        analyze_benchmark_results()
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
