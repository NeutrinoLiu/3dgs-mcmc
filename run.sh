#!/bin/bash

# Function to run the python command with given parameters a, b, c and redirect output to a log file
run_train() {
    local a=$1
    local b=$2
    local c=$3
    local d=$4
    echo "Running with parameters: a=${a}, b=${b}, c=${c}, d=${d}"
    python train.py --source_path data/actor06_frame${d} --cap_max ${a} --scale_reg ${b} --opacity_reg ${b} --noise_lr ${c} --init_type random --eval -m test_a6_${a}_${b}_${c}_${d} > ${a}_${b}_${c}_${d}.log 2>&1
}

# Run the function with different sets of parameters

# best for actor06
# 100000, 1e-4, 5e4

run_train 100000 1e-2 5e3 2
run_train 100000 1e-3 5e3 2
run_train 100000 1e-4 5e3 2
run_train 100000 1e-5 5e3 2
run_train 100000 1e-6 5e3 2

run_train 100000 1e-2 5e4 2
run_train 100000 1e-3 5e4 2
run_train 100000 1e-4 5e4 2
run_train 100000 1e-5 5e4 2
run_train 100000 1e-6 5e4 2

run_train 100000 1e-2 5e5 2
run_train 100000 1e-3 5e5 2
run_train 100000 1e-4 5e5 2
run_train 100000 1e-5 5e5 2
run_train 100000 1e-6 5e5 2

run_train 150000 1e-2 5e4 2
run_train 150000 1e-3 5e4 2
run_train 150000 1e-4 5e4 2
run_train 150000 1e-5 5e4 2
run_train 150000 1e-6 5e4 2