#!/bin/bash

# Function to run the python command with given parameters a, b, c and redirect output to a log file
run_train() {
    local a=$1
    local b=$2
    echo "Running with parameters: a=${a}, b=${b}"
    python train_swin.py --source_path data/actorshq_02 --scale_reg 1e-2 --opacity_reg 2e-2 --noise_lr 5e5\
    --cap_max ${a} --swin_size ${b} --max_frame 220 --min_frame 160 \
    --sh_degree 1 --eval -m test_a02_${a}_${b} --iterations 20010 --genesis_iterations 30010 > exp1/${a}_${b}.log 2>&1
    mv result.txt exp1/${a}_${b}_result.txt
}

# Run the function with different sets of parameters
mkdir exp1


run_train 40000 4
run_train 80000 4
run_train 120000 4
run_train 40000 8
run_train 80000 8
run_train 120000 8
run_train 40000 12
run_train 80000 12
run_train 120000 12

run_train 160000 4
run_train 160000 8
run_train 160000 12
run_train 40000 16
run_train 80000 16
run_train 120000 16
run_train 160000 16

run_train 200000 4
run_train 200000 8
run_train 200000 12
run_train 200000 16
run_train 40000 20
run_train 80000 20
run_train 120000 20
run_train 160000 20
run_train 200000 20
