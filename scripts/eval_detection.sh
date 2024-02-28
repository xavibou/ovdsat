#!/bin/bash

DATA_DIR=/mnt/ddisk/boux/code/data
backbone=dinov2
dataset=simd
finetune_type=boxes     # masks or boxes
N=10

python eval_detection.py \
    --dataset ${dataset} \
    --val_root_dir /mnt/ddisk/boux/code/data/${dataset}/val \
    --save_dir run/eval/detection/${dataset}/backbone_${backbone}_${finetune_type}/N${N} \
    --val_annotations_file /mnt/ddisk/boux/code/data/${dataset}/val_coco.json \
    --prototypes_path run/train/${finetune_type}/${dataset}_N${N}/prototypes.pth \
    --bg_prototypes_path /mnt/ddisk/boux/code/ovdsat/prototypes/simcd_subset_N10/simd_bg_prototypes.pth \
    --backbone_type ${backbone} \
    --classification box \
    --target_size 602 602 \
    --batch_size 1 \
    --num_workers 8 \
    --conf_thres 0.3 \
    --scale_factor 1 
