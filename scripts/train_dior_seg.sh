#!/bin/bash

# Define the values of N you want to iterate over
#for backbone in clip-14 clip-32 georsclip-32 remoteclip-14 remoteclip-32
dataset=dior
backbone=dinov2
N=10
echo "Training ${backbone}"
python train.py \
    --root_dir /mnt/ddisk/boux/code/data/dior/DIOR/Annotations/subsets/train/images \
    --save_dir "run/train/segmentation/${dataset}/learned_prototype_${backbone}_mean_aggr" \
    --annotations_file /mnt/ddisk/boux/code/devit/datasets/sam_dior_N10/modified_annotations.json \
    --val_annotations_file /mnt/ddisk/boux/code/devit/datasets/sam_dior_for_val/modified_annotations_validation.json \
    --prototypes_path /mnt/ddisk/boux/code/ovdsat/run/backbone_prototypes/dior_N${N}/prototypes_${backbone}.pt \
    --backbone_type ${backbone} \
    --num_epochs 200 \
    --lr 2e-4 \
    --target_size 602 602 \
    --batch_size 8 \
    --num_neg 0 \
    --num_workers 8 \
    --iou_thr 0.1 \
    --conf_thres 0.2 \
    --scale_factor 1 \
    --use_segmentation 