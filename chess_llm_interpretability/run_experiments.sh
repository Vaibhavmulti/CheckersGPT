#!/bin/bash

# my_checkpoints=(100 200 300 400 500 600 700 800 900 \
#                 1000 2000 3000 4000 5000 6000 7000 8000 9000 \
#                        10000 20000 30000 40000 50000 60000 70000 80000 90000 \
#                        100000 200000 300000 400000 500000 600000)
my_checkpoints=(20)

source_base_path="/DATA3/vaibhav/experiments/Checkers_human_ckpts/CheckersHuman"
destination_base_path="/home/vaibhav/nanogpt/CheckersGPT/chess_llm_interpretability/models"

for item in "${my_checkpoints[@]}"
do
    echo "Processing $item"
    source_file="${source_base_path}${item}.pt"
    destination_file="${source_base_path}.pt"
    mv "$source_file" "$destination_file"
    source_file="${source_base_path}.pt"
    destination_file="${destination_base_path}/CheckersHuman.pt"
    mv "$source_file" "$destination_file"

    python3 model_setup.py
    python3 train_test_chess.py


done

# Uncomment to move files back if needed
# for item in "${my_checkpoints[@]}"
# do
#     source_file="${destination_base_path}/CheckersHuman${item}.pt"
#     destination_file="${source_base_path}${item}.pt"
#     mv "$source_file" "$destination_file"
# done
