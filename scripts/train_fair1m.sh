#!/bin/bash

# Define the values of N you want to iterate over
#for backbone in clip-14 clip-32 georsclip-32 remoteclip-14 remoteclip-32
dataset=fair1m
backbone=dinov2

echo "Training ${backbone} on ${dataset} dataset"
python train.py \
    --root_dir /mnt/ddisk/boux/code/data/fair1m/train/preprocessed/images \
    --save_dir "run/train/${dataset}/${backbone}_no_neg" \
    --annotations_file /mnt/ddisk/boux/code/data/fair1m/train/preprocessed/fair1m_train_2.0_subset_N10.json \
    --val_annotations_file /mnt/ddisk/boux/code/data/fair1m/train/preprocessed/subset_small_val.json \
    --prototypes_path /mnt/ddisk/boux/code/ovdsat/run/backbone_prototypes/fair1m/N10/prototypes_dinov2.pt \
    --bg_prototypes_path /mnt/ddisk/boux/code/ovdsat/run/backbone_prototypes/fair1m/N10/bg_prototypes_dinov2.pt \
    --backbone_type ${backbone} \
    --num_epochs 200 \
    --lr 2e-4 \
    --target_size 602 602 \
    --batch_size 1 \
    --num_neg 0 \
    --num_workers 8 \
    --iou_thr 0.1 \
    --conf_thres 0.2 \
    --scale_factor 1 \
    --only_train_prototypes