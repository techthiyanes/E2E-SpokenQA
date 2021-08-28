#!/bin/bash

# $1 : experiment name
# $2 : upstream name

DIR="/home/daniel094144/Daniel/SpokenQA"

python3 main.py --config config.yaml \
                --name $1 \
                --ckptdir ${DIR}/ckpt/ \
                --upstream $2 \
            