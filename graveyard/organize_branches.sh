#!/bin/bash

# Quick Branch Organization - Keep Your "/" Naming Style!
# This script organizes your existing branches following your evolution path
# while preserving the clean "/" hierarchy you prefer

set -e

echo "🔄 Organizing Evolution Path Branches (Keeping / Style)"
echo "======================================================"

# Function to check if branch exists
branch_exists() {
    git show-ref --verify --quiet refs/heads/$1
}

# Function to safely rename branch
safe_rename() {
    local old_name=$1
    local new_name=$2
    
    if branch_exists "$old_name"; then
        echo "✅ Renaming: $old_name → $new_name"
        git branch -m "$old_name" "$new_name"
        return 0
    else
        echo "⚠️  Branch '$old_name' not found, skipping..."
        return 1
    fi
}

echo "📋 Current branches:"
git branch -a | grep -E "(cpu|gpu)" | head -10

echo ""
echo "🎯 Organizing Evolution Path (Keeping / Hierarchy)..."
echo ""

# Core Evolution Path Organization - Keep the "/" style you like!
echo "1️⃣ STEP 1: CPU BFS Flood Fill"
safe_rename "cpu/squential" "cpu/bfs-flood-fill"  # Fix the typo: squential → bfs-flood-fill

echo ""
echo "2️⃣ STEP 2: CPU BFS Flood Fill + Scanning"
# This will be implemented - combines BFS with scanning optimization
echo "⚠️  cpu/bfs-flood-fill-and-scanning needs implementation"

echo ""
echo "3️⃣ STEP 3: GPU BFS Flood Fill" 
safe_rename "gpu/basic-flood-fill-using-BFS" "gpu/bfs-flood-fill"  # Simplify the long name

echo ""
echo "4️⃣ STEP 4: GPU BFS Flood Fill + Scanning"
safe_rename "gpu/add-parallel-scanning-blobs" "gpu/bfs-flood-fill-and-scanning"  # Combines BFS with scanning

echo ""
echo "✅ Organization complete!"
echo ""
echo "🎯 EVOLUTION PATH STATUS (with / hierarchy):"
echo "============================================"
echo "✅ Step 1: cpu/bfs-flood-fill                (renamed from cpu/squential)"
echo "❌ Step 2: cpu/bfs-flood-fill-and-scanning   (needs implementation)"  
echo "✅ Step 3: gpu/bfs-flood-fill                (renamed from gpu/basic-flood-fill-using-BFS)"
echo "📦 Step 4: gpu/bfs-flood-fill-and-scanning   (renamed, stashed work ready to apply)"
echo ""
echo "🌟 Your \"/\" naming style benefits:"
echo "   - Clear hierarchy: cpu/* vs gpu/*"
echo "   - Logical grouping by platform"
echo "   - Easy to see evolution steps within each platform"
echo "   - Standard Git branch organization pattern"
echo ""
echo "📋 Updated branches:"
git branch | grep -E "(cpu|gpu)" || echo "No evolution branches found"
echo ""
echo "⚠️  Next steps:"
echo "   1. Apply stashed work to gpu/bfs-flood-fill-and-scanning"
echo "   2. Implement cpu/bfs-flood-fill-and-scanning (Step 2)"  
echo "   3. Update remote tracking for renamed branches"
echo "   4. Create documentation showing the complete evolution"
echo ""
echo "🎯 Perfect Evolution Path Structure:"
echo "   cpu/bfs-flood-fill                ← Step 1: Simple BFS flood fill"
echo "   cpu/bfs-flood-fill-and-scanning   ← Step 2: BFS + scanning optimization"
echo "   gpu/bfs-flood-fill                ← Step 3: BFS flood fill on GPU"  
echo "   gpu/bfs-flood-fill-and-scanning   ← Step 4: BFS + scanning on GPU"
