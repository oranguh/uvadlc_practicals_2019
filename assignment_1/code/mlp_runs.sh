#!/bin/bash
# mkdir results_torch
CMD="python train_mlp_pytorch.py --max_steps 3000 --batch_size 200 --eval_freq 100 "
DIR="results_torch"
declare -a LR=("0.002" "0.001" "0.01")
declare -a HIDDEN=('1000,1000,200,200' '1000,1000,200,200,100,100' '1000,1000,200,100' '200,200,200' '100,200')
declare -a OPT=("adam")

for lr in "${LR[@]}"; do
  for hidden in "${HIDDEN[@]}"; do
    FULL="$CMD --dnn_hidden_units $hidden --learning_rate $lr --optim $opt --op_loc $DIR/lr_${lr}_hidden_${hidden}_opt_${opt}"
    echo $FULL; eval $FULL
  done
done
