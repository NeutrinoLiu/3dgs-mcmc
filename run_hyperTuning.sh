#!/bin/bash

# Function to run the python command with given parameters a, b, c and redirect output to a log file
run_train() {
    local a=$1
    local b=$2
    local c=$3
    echo "Running with parameters: a=${a}, b=${b}, c=${c}"
    python train_swin.py --source_path data/DyNeRF/cook_spinach --cap_max ${a} --scale_reg ${b} --opacity_reg ${b} --noise_lr ${c} \
    --sh_degree 1 --swin_size 1 --eval -m test_dynerf_${a}_${b}_${c} --iterations 5000 --first_frame_only > hyperTuning/${a}_${b}_${c}.log 2>&1
    mv result.txt hyperTuning/${a}_${b}_${c}_result.txt
}

# Run the function with different sets of parameters

run_train 320000 1e-2 5e3
run_train 320000 1e-3 5e3
run_train 320000 1e-4 5e3
run_train 320000 1e-5 5e3
run_train 320000 1e-6 5e3

run_train 320000 1e-2 5e4
run_train 320000 1e-3 5e4
run_train 320000 1e-4 5e4
run_train 320000 1e-5 5e4
run_train 320000 1e-6 5e4

run_train 320000 1e-2 5e5
run_train 320000 1e-3 5e5
run_train 320000 1e-4 5e5
run_train 320000 1e-5 5e5
run_train 320000 1e-6 5e5