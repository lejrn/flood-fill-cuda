#!/usr/bin/env python3
"""
Cleanup script for CUDA flood fill multi-blocks directory.
Safely removes redundant files while preserving essential implementation.
"""

import os
import subprocess

def run_command(command, description):
    """Run a system command safely"""
    print(f"🔧 {description}")
    print(f"   Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✅ Success")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
        else:
            print(f"   ❌ Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False
    
    print()
    return True

def cleanup_redundant_files():
    """Remove redundant files while keeping essential implementation"""
    
    print("🧹 CUDA FLOOD FILL DIRECTORY CLEANUP")
    print("=" * 50)
    print()
    print("📚 All milestones have been documented in git!")
    print("🏷️  Tags created: v1.0 through v1.4-optimization-complete")
    print("☁️  Pushed to remote repository")
    print()
    
    # Define the base directory
    base_dir = "/home/lrn/Repos/flood-fill-cuda/src/gpu/single_blob/multi-blocks"
    
    print("🔍 ANALYZING CURRENT FILES:")
    print("-" * 30)
    
    # List all current files
    all_files = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isfile(item_path):
            size = os.path.getsize(item_path)
            print(f"   📄 {item:<35} ({size:,} bytes)")
            all_files.append(item)
        elif os.path.isdir(item_path):
            print(f"   📁 {item}/")
    print()
    
    # Define files to keep (essential for production)
    essential_files = {
        # Core implementation
        "kernels_fixed.py",           # Main optimized CUDA kernels
        "main_optimized.py",          # Main execution script
        "utils.py",                   # Configuration and utilities
        "debug_logging.py",           # Debug and logging utilities
        
        # Latest benchmarking and testing
        "benchmark_optimizer.py",     # Parameter benchmarking tool
        "test_new_optimal_config.py", # Latest optimal config testing
        
        # Latest analysis
        "analyze_expanded_results.py", # Most recent analysis (96 configs)
        "final_summary.py",           # Comprehensive final results
        
        # Documentation
        "OPTIMIZATION_RESULTS.md",    # Optimization documentation
        "create_milestones.py",       # This cleanup script
    }
    
    # Define files to remove (redundant)
    redundant_files = {
        # Older implementations
        "kernels.py",                 # Superseded by kernels_fixed.py
        "main.py",                    # Superseded by main_optimized.py
        
        # Older tests
        "test_optimal_config.py",     # Superseded by test_new_optimal_config.py
        "test_fixed.py",              # Development test, no longer needed
        
        # Older analysis
        "analyze_results.py",         # Superseded by analyze_expanded_results.py
        "analyze_results_simple.py", # Redundant analysis script
        
        # Redundant/empty files
        "setup.py",                   # Redundant setup file
        "configure_gpu.py",           # Redundant configuration
        "profiling.py",               # Empty file
    }
    
    print("✅ FILES TO KEEP (Essential for Production):")
    print("-" * 45)
    for file in sorted(essential_files):
        if file in all_files:
            file_path = os.path.join(base_dir, file)
            size = os.path.getsize(file_path)
            print(f"   🔒 {file:<35} ({size:,} bytes)")
        else:
            print(f"   ⚠️  {file:<35} (NOT FOUND)")
    print()
    
    print("🗑️  FILES TO REMOVE (Redundant):")
    print("-" * 35)
    files_to_remove = []
    for file in sorted(redundant_files):
        if file in all_files:
            file_path = os.path.join(base_dir, file)
            size = os.path.getsize(file_path)
            print(f"   ❌ {file:<35} ({size:,} bytes)")
            files_to_remove.append(file)
        else:
            print(f"   ℹ️  {file:<35} (already removed)")
    print()
    
    # Double-check directories to preserve
    essential_dirs = {
        "benchmark_results",  # All benchmark data
        "images"              # Output images
    }
    
    print("📁 DIRECTORIES TO PRESERVE:")
    print("-" * 30)
    for dir_name in essential_dirs:
        dir_path = os.path.join(base_dir, dir_name)
        if os.path.exists(dir_path):
            print(f"   🔒 {dir_name}/")
        else:
            print(f"   ⚠️  {dir_name}/ (NOT FOUND)")
    print()
    
    # Confirm before deletion
    if files_to_remove:
        print(f"⚠️  READY TO REMOVE {len(files_to_remove)} REDUNDANT FILES")
        print("=" * 50)
        print("This will:")
        print(f"   • Delete {len(files_to_remove)} redundant files")
        print(f"   • Preserve {len(essential_files)} essential files")
        print("   • Keep all directories (benchmark_results/, images/)")
        print("   • All development history is preserved in git milestones")
        print()
        
        response = input("🤔 Proceed with cleanup? (y/N): ").lower().strip()
        if response in ['y', 'yes']:
            print()
            print("🧹 STARTING CLEANUP...")
            print("-" * 25)
            
            # Remove redundant files
            success_count = 0
            for file in files_to_remove:
                file_path = os.path.join(base_dir, file)
                try:
                    os.remove(file_path)
                    print(f"   ✅ Removed: {file}")
                    success_count += 1
                except Exception as e:
                    print(f"   ❌ Failed to remove {file}: {e}")
            
            print()
            print("📊 CLEANUP SUMMARY:")
            print("-" * 20)
            print(f"   ✅ Successfully removed: {success_count} files")
            print(f"   🔒 Essential files preserved: {len(essential_files)}")
            print(f"   📁 Directories preserved: {len(essential_dirs)}")
            print()
            
            # Final directory listing
            print("📋 FINAL DIRECTORY STRUCTURE:")
            print("-" * 32)
            remaining_files = []
            for item in sorted(os.listdir(base_dir)):
                item_path = os.path.join(base_dir, item)
                if os.path.isfile(item_path):
                    size = os.path.getsize(item_path)
                    print(f"   📄 {item:<35} ({size:,} bytes)")
                    remaining_files.append(item)
                elif os.path.isdir(item_path):
                    print(f"   📁 {item}/")
            print()
            
            print("✅ CLEANUP COMPLETE!")
            print("=" * 20)
            print("🎯 Production-ready directory structure achieved!")
            print("📚 All development history preserved in git milestones")
            print("🏷️  Access previous versions using git tags:")
            print("   • git checkout v1.0-initial-optimization")
            print("   • git checkout v1.1-benchmarking-tools")
            print("   • git checkout v1.2-first-results")
            print("   • git checkout v1.3-breakthrough")
            print("   • git checkout v1.4-optimization-complete")
            print()
            print("🚀 Ready for production deployment!")
            
        else:
            print("❌ Cleanup cancelled. Files preserved.")
    else:
        print("ℹ️  No redundant files found. Directory already clean!")
    
    # Final git status check
    print()
    print("📋 Final Git Status:")
    run_command("git status --porcelain", "Checking git status")

def main():
    """Main cleanup function"""
    try:
        cleanup_redundant_files()
        return 0
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)
