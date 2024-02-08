#!/bin/bash

dataset=fair1m

python eval.py \
    --root_dir /mnt/ddisk/boux/code/data/fair1m/val/preprocessed/images \
    --annotations_file /mnt/ddisk/boux/code/data/fair1m/val/preprocessed/fair1m_val_2.0.json \
    --dataset ${dataset} \
    --save_dir "run/eval/${dataset}/N10_results" \
    --prototypes_path /mnt/ddisk/boux/code/ovdsat/run/train/fair1m/dinov2_no_neg/prototypes.pth \
    --bg_prototypes_path /mnt/ddisk/boux/code/ovdsat/run/train/fair1m/dinov2_no_neg/bg_prototypes.pth \
    --backbone_type dinov2 \
    --target_size 602 602\
    --batch_size 1 \
    --num_workers 8 \
    --iou_thr 0.2 \
    --conf_thres 0.001 \
    --scale_factor 1 

    