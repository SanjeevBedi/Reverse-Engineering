#!/bin/bash

# Test multiple seeds for solid reconstruction
# Usage: ./test_multiple_seeds.sh

SEEDS="0 10 20 30 40 50 60 70 80 90 100"
PYTHON="/opt/anaconda3/envs/pyocc/bin/python"

echo "==============================================="
echo "Testing Solid Reconstruction: Seeds 0-100"
echo "==============================================="
echo ""

for seed in $SEEDS; do
    echo "---------- SEED $seed ----------"
    
    # Check if input file exists
    if [ ! -f "Solid_${seed}.json" ]; then
        echo "WARNING: Solid_${seed}.json not found, skipping..."
        echo ""
        continue
    fi
    
    # Run reconstruction and extract key metrics
    $PYTHON Reconstruct_Solid.py --seed $seed --no-occ-viewer 2>&1 | \
        grep -E "(STEP 7.4.5|STEP 7.4.6|STEP 7.5|STEP 8.2|Total boundary edges|boundary edges after|Iteration [0-9]+: Created|splits found|Number of free edges|Solid created|SUCCESS|FAILED)" | \
        grep -v "DEBUG" | \
        sed 's/^/  /'
    
    EXIT_CODE=${PIPESTATUS[0]}
    if [ $EXIT_CODE -ne 0 ]; then
        echo "  ERROR: Reconstruction failed with exit code $EXIT_CODE"
    fi
    
    echo ""
done

echo "==============================================="
echo "Test Complete"
echo "==============================================="
