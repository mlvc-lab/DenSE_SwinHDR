#!/bin/bash
for i in {1..300..8}
do
    CUDA_VISIBLE_DEVICES=1 python3 main_test_swinir.py --index $i
done