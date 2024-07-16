#!/bin/bash

# Define the root directory
ROOT_DIR=$1

# Check if the root directory exists
if [ ! -d "$ROOT_DIR" ]; then
  echo "Root directory does not exist: $ROOT_DIR"
  exit 1
fi

# Iterate through each subdirectory within the root directory
for dir in "$ROOT_DIR"/*; do
  if [ -d "$dir" ]; then
    # Echo the directory name
    echo "Processing directory: $dir"
    
    # Extract the directory name to create the log file
    DIR_NAME=$(basename "$dir")
    
    # Define the log file path
    LOG_FILE="$ROOT_DIR/$DIR_NAME.log"
    
    # Run the render.py script and redirect output to the log file
    echo "Running render.py for $dir" >> "$LOG_FILE"
    python render.py -m "$dir" >> "$LOG_FILE" 2>&1
    
    # Run the evaluate.py script and append output to the log file
    echo "Running metrics.py for $dir" >> "$LOG_FILE"
    python metrics.py -m "$dir" >> "$LOG_FILE" 2>&1
  fi
done
