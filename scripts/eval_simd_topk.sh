#!/bin/bash

dataset=simd

python eval.py \
    --root_dir /mnt/ddisk/boux/code/data/simd/validation \
    --annotations_file /mnt/ddisk/boux/code/data/simd/val_coco.json \
    --dataset ${dataset} \
    --save_dir "run/eval/${dataset}/topk_results" \
    --prototypes_path /mnt/ddisk/boux/code/ovdsat/run/train/learned_prototype_dinov2_top10_aggregation/prototypes.pth \
    --bg_prototypes_path /mnt/ddisk/boux/code/ovdsat/run/train/learned_prototype_dinov2_top10_aggregation/bg_prototypes.pth \
    --backbone_type dinov2 \
    --target_size 602 602\
    --aggregation topk \
    --batch_size 1 \
    --num_workers 8 \
    --iou_thr 0.2 \
    --conf_thres 0.001 \
    --scale_factor 1 

    