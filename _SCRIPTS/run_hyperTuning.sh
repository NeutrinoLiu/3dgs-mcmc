#!/bin/bash

# Function to run the python command with given parameters a, b, c and redirect output to a log file
run_train() {
    local a=$1
    local b=$2
    local c=$3
    echo "Running with parameters: a=${a}, b=${b}, c=${c}"
    python train_swin.py --source_path data/DyNeRF/cook_spinach --cap_max ${a} --scale_reg ${b} --opacity_reg ${b} --noise_lr ${c} \
    --sh_degree 1 --swin_size 1 --eval -m test_dynerf_${a}_${b}_${c} --iterations 8010 --first_frame_only > hyperTuning/${a}_${b}_${c}.log 2>&1
    mv result.txt hyperTuning/${a}_${b}_${c}_result.txt
}

# Run the function with different sets of parameters

run_train 100000 1e-2 5e5
run_train 150000 1e-2 5e5
run_train 200000 1e-2 5e5
run_train 250000 1e-2 5e5
run_train 300000 1e-2 5e5
run_train 350000 1e-2 5e5
run_train 400000 1e-2 5e5