#!/bin/bash

# Function to run the command with parameter x
run() {
    echo "Running with parameters: v_lr=$1 rvec_lr=$2 rcen_lr=$3"

    # Execute the training command
    python train_swin.py --source_path data/actor06_all \
    --cap_max 100000 --eval --swin_size 10 --iterations 20000 \
    --detect_anomaly --first_frame_only \
    --rigid_v_lr $1 --rigid_rotvec_lr $2 --rigid_rotcen_lr $3 > /dev/null 2>&1

    # Move the result file to the destination path
    mv result.txt "a06_deform/result_${1}_${2}_${3}.txt"
}

cleanup() {
    echo "Script interrupted. Exiting..."
    rm result.txt
    exit 1
}

# Trap keyboard interrupt (Ctrl+C)
trap cleanup SIGINT


all_values=(1e-4 1e-5 1e-6)

for lr1 in ${all_values[@]}; do
    for lr2 in ${all_values[@]}; do
        for lr3 in ${all_values[@]}; do
            run $lr1 $lr2 $lr3
        done
    done
done