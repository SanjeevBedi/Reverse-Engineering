#!/bin/bash
cd /Users/sbedi/Nextcloud/Python/Solid/random_solids
/opt/anaconda3/envs/pyocc/bin/python Reconstruct_Solid.py --seed 113 --no-occ-viewer 2>&1 | grep -A 80 "Pruning Iteration 2" | head -120
