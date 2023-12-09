#!/bin/bash

# Define the values of N you want to iterate over
for backbone in clip-14 clip-32 georsclip-32 remoteclip-14 remoteclip-32
do
    echo "Training ${backbone}"
    python train.py \
        --root_dir /mnt/ddisk/boux/code/data/simd/training \
        --save_dir "run/train/learned_prototype_${backbone}_neg_samples" \
        --annotations_file /mnt/ddisk/boux/code/data/simd/train_coco_subset_N10.json \
        --val_annotations_file /mnt/ddisk/boux/code/data/simd/train_coco_subset_2.json \
        --prototypes_path "/mnt/ddisk/boux/code/ovdsat/run/backbone_prototypes/prototypes_${backbone}.pt" \
        --bg_prototypes_path "/mnt/ddisk/boux/code/ovdsat/run/backbone_prototypes/bg_prototypes_${backbone}.pt" \
        --backbone_type ${backbone} \
        --num_epochs 200 \
        --lr 2e-5 \
        --target_size 602 602 \
        --batch_size 1 \
        --num_neg 1 \
        --num_workers 8 \
        --iou_thr 0.1 \
        --conf_thres 0.2 \
        --scale_factor 1 
done