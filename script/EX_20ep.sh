#!/bin/bash

CUDA_VISIBLE_DEVICES=1,2,3 python ../src/model/base_segmentation.py --type EX --epochs 20