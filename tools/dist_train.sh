#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-12620}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py --validate --test-last --test-best $CONFIG --launcher pytorch ${@:3}
# Any arguments from the third one are captured by ${@:3}
