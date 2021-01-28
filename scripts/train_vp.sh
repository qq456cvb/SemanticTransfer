#!/usr/bin/env bash

outdir=./output/marrnet3

if [ $# -lt 2 ]; then
    echo "Usage: $0 gpu class[ ...]"
    exit 1
fi
gpu="$1"
class="$2"
shift # shift the remaining arguments
shift

set -e

python train.py \
    --net vpnet \
    --pred_depth_minmax \
    --dataset shapenet \
    --classes "$class" \
    --batch_size 16 \
    --epoch_batches 625 \
    --eval_batches 2 \
    --log_time \
    --optim adam \
    --lr 1e-3 \
    --epoch 1000 \
    --vis_batches_vali 1 \
    --gpu "$gpu" \
    --save_net 10 \
    --workers 4 \
    --logdir "$outdir" \
    --suffix '{classes}' \
    --tensorboard \
    --vis_every_vali 1 \
    $*
