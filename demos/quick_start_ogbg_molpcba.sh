#!/bin/bash
# Quick start script for ogbg-molpcba integration
# Run this to verify the integration and demonstrate Transform DAG

set -e  # Exit on error

echo "================================================================================================="
echo "OGBG-MOLPCBA TRANSFORM DAG DEMO - QUICK START"
echo "================================================================================================="

# Check if OGB is installed
echo ""
echo "1️⃣  Checking dependencies..."
python -c "import ogb" 2>/dev/null && echo "   ✓ OGB installed" || {
    echo "   ❌ OGB not found. Installing..."
    pip install ogb
    echo "   ✓ OGB installed"
}

python -c "import topobench" 2>/dev/null && echo "   ✓ TopoBench installed" || {
    echo "   ❌ TopoBench not found. Please install with: pip install -e ."
    exit 1
}

echo "   ✓ All dependencies satisfied"

# Show available options
echo ""
echo "================================================================================================="
echo "AVAILABLE OPTIONS"
echo "================================================================================================="
echo ""
echo "Option 1: CONCEPT DEMO (no preprocessing, instant)"
echo "  $ python demos/ogbg_molpcba_transform_dag_demo.py --skip-preprocessing"
echo "  Duration: < 1 minute"
echo "  Shows: Conceptual explanation of Transform DAG benefits"
echo ""
echo "Option 2: QUICK VALIDATION (10K subset, ~30 min)"
echo "  $ python demos/ogbg_molpcba_transform_dag_demo.py --subset 10000"
echo "  Duration: ~30 minutes"
echo "  Shows: Full preprocessing + demo on 10K graphs"
echo ""
echo "Option 3: SCALE TEST (50K subset, ~2 hours)"
echo "  $ python demos/ogbg_molpcba_transform_dag_demo.py --subset 50000"
echo "  Duration: ~2 hours"
echo "  Shows: Scalability proof on 50K graphs"
echo ""
echo "Option 4: PRODUCTION SCALE (full 437K, ~6 hours)"
echo "  $ python demos/ogbg_molpcba_transform_dag_demo.py"
echo "  Duration: ~6 hours"
echo "  Shows: Full-scale production demo"
echo ""
echo "================================================================================================="

# Ask user which option to run
echo ""
read -p "Which option would you like to run? (1/2/3/4) [1]: " choice
choice=${choice:-1}

case $choice in
    1)
        echo ""
        echo "Running CONCEPT DEMO (instant)..."
        python demos/ogbg_molpcba_transform_dag_demo.py --skip-preprocessing
        ;;
    2)
        echo ""
        echo "Running QUICK VALIDATION (10K subset, ~30 min)..."
        echo "This will download ~35 MB and process 10,000 graphs"
        read -p "Continue? (y/n) [y]: " confirm
        confirm=${confirm:-y}
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            python demos/ogbg_molpcba_transform_dag_demo.py --subset 10000
        else
            echo "Cancelled."
            exit 0
        fi
        ;;
    3)
        echo ""
        echo "Running SCALE TEST (50K subset, ~2 hours)..."
        echo "This will process 50,000 graphs (~500 MB disk space)"
        read -p "Continue? (y/n) [y]: " confirm
        confirm=${confirm:-y}
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            python demos/ogbg_molpcba_transform_dag_demo.py --subset 50000
        else
            echo "Cancelled."
            exit 0
        fi
        ;;
    4)
        echo ""
        echo "Running PRODUCTION SCALE (full 437K, ~6 hours)..."
        echo "This will process 437,929 graphs (~3-4 GB disk space)"
        read -p "Are you sure? This will take ~6 hours! (y/n) [n]: " confirm
        confirm=${confirm:-n}
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            python demos/ogbg_molpcba_transform_dag_demo.py
        else
            echo "Cancelled. Consider running with --subset 10000 first!"
            exit 0
        fi
        ;;
    *)
        echo "Invalid option. Please run manually with desired arguments."
        exit 1
        ;;
esac

echo ""
echo "================================================================================================="
echo "✅ DEMO COMPLETE!"
echo "================================================================================================="
echo ""
echo "Next steps:"
echo "  - Review the output above to understand Transform DAG benefits"
echo "  - Check processed data in: ./data/ogbg_molpcba/processed_subset_*"
echo "  - Read more: demos/README_OGBG_MOLPCBA.md"
echo "  - Train a model: python main.py experiment=ogbg_molpcba_dag_demo"
echo ""
echo "🏆 Transform DAG: Solving the Lifting Bottleneck in TDL!"
echo "================================================================================================="
