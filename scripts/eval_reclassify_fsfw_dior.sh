#!/bin/bash

# Define the values of N you want to iterate over
#for backbone in clip-14 clip-32 georsclip-32 remoteclip-14 remoteclip-32
backbone=dinov2
dataset=dior
N=30

python eval_detection_reclassify.py \
    --dataset ${dataset} \
    --val_root_dir /mnt/ddisk/boux/code/data/${dataset}/val \
    --save_dir run/eval/detection/${dataset}/backbone_${backbone}_reclassified/N${N} \
    --val_annotations_file /mnt/ddisk/boux/code/data/${dataset}/val_coco.json \
    --prototypes_path /mnt/ddisk/boux/code/ovdsat/run/train/boxes/${dataset}_N${N}/prototypes.pth \
    --box_pred_dir /mnt/ddisk/boux/code/Few-shot-Object-Detection-via-Feature-Reweighting/results/${dataset}_${N}shot_novel0_neg0/ene000005 \
    --backbone_type ${backbone} \
    --target_size 602 602 \
    --batch_size 1 \
    --num_workers 8 \
    --conf_thres 0.3 \
    --scale_factor 1 
