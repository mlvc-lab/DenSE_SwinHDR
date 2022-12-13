#!/bin/bash
for i in {0..300..8}
do
    CUDA_VISIBLE_DEVICES=0 python3 main_test_swinir.py --index $i
done