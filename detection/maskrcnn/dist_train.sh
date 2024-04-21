#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun --nproc_per_node=$GPUS --master_port=6666 \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
