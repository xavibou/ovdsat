#!/bin/bash

# Define the values of N you want to iterate over
#for backbone in clip-14 clip-32 georsclip-32 remoteclip-14 remoteclip-32
backbone=dinov2
dataset=simd
N=10

python eval_detection.py \
    --dataset ${dataset} \
    --val_root_dir /mnt/ddisk/boux/code/data/simd/validation \
    --save_dir run/eval/detection/${dataset}/backbone_${backbone}_segmentation/N${N} \
    --val_annotations_file /mnt/ddisk/boux/code/data/simd/val_coco.json \
    --prototypes_path /mnt/ddisk/boux/code/ovdsat/run/train/segmentation/learned_prototype_dinov2_mean_aggr/prototypes.pth \
    --bg_prototypes_path /mnt/ddisk/boux/code/ovdsat/run/backbone_prototypes/simd_N10/bg_prototypes_dinov2.pt \
    --backbone_type ${backbone} \
    --target_size 602 602 \
    --batch_size 16 \
    --num_workers 8 \
    --scale_factor 1 
