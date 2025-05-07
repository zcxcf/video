#!/usr/bin/env sh


export PYTHONPATH=$PYTHONPATH:$(pwd)


torchrun --nproc_per_node=8 ./train/video/train_stage1_dis.py