#!/bin/bash

# Define the values of N you want to iterate over
#for backbone in clip-14 clip-32 georsclip-32 remoteclip-14 remoteclip-32
for backbone in dinov2
do
    echo "Training ${backbone}"
    python train.py \
        --root_dir /mnt/ddisk/boux/code/data/simd/training \
        --save_dir "run/train/segmentation/learned_prototype_${backbone}_mean_aggr_weighted_cosim" \
        --annotations_file /mnt/ddisk/boux/code/devit/datasets/sam_simd_N10/modified_annotations.json \
        --val_annotations_file /mnt/ddisk/boux/code/devit/datasets/sam_simd_validation_finetuning/modified_annotations_validation.json \
        --prototypes_path "/mnt/ddisk/boux/code/ovdsat/run/classification_benchmark_exp/prototypes_dinov2.pt" \
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
done