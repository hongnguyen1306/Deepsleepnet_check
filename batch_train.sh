#!/bin/bash

data=$1
output=$2
nfolds=$3
start=$4
end=$5
gpu=$6
train=$7
val=$8


if [[ -n "$data" ]] && [[ -n "$start" ]] && [[ -n "$end" ]] && [[ -n "$gpu" ]]; then
    for i in $(eval echo {$start..$end})
    do
       CUDA_VISIBLE_DEVICES=$gpu python train.py --data_dir=$data --train_dir=data_sleep_npz/train --val_dir=data_sleep_npz/val --output_dir=$output --n_folds=$nfolds --pretrain_epochs=100 --finetune_epochs=200 --fold_idx=$i --resume=True
    done
else
    echo "argument error"
fi

