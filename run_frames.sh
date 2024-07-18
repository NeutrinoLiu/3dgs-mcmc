#!/bin/bash

for f in {0..9}
do
    echo "Running frame $f"
    python train.py --source_path data/actor06_all --cap_max 100000 --scale_reg 1e-4 --opacity_reg 1e-4 --noise_lr 5e4 --init_type random --eval --frame $f -m "a06_all/$f" > log_frame_$f.log 2>&1
done
