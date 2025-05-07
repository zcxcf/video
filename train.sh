#!/bin/bash

NUM_GPUS=8

torchrun --master_port 29507 --nproc_per_node=$NUM_GPUS train_stage1_dis_v2.py

