#!/bin/bash

# Define the values of N you want to iterate over
#for backbone in clip-14 clip-32 georsclip-32 remoteclip-14 remoteclip-32
for backbone in dinov2
do
    echo "Training ${backbone}"
    python train.py \
        --root_dir /mnt/ddisk/boux/code/data/simd/training \
        --save_dir "run/train/learned_prototype_${backbone}_max_aggregation" \
        --annotations_file /mnt/ddisk/boux/code/data/simd/train_coco_subset_N10.json \
        --val_annotations_file /mnt/ddisk/boux/code/data/simd/train_coco_subset_2.json \
        --prototypes_path "/mnt/ddisk/boux/code/ovdsat/run/classification_benchmark_exp/prototypes_dinov2.pt" \
        --bg_prototypes_path "/mnt/ddisk/boux/code/ovdsat/run/classification_benchmark_exp/bg_prototypes_dinov2.pt" \
        --backbone_type ${backbone} \
        --aggregation topk \
        --num_epochs 200 \
        --lr 2e-4 \
        --target_size 602 602 \
        --batch_size 4 \
        --num_neg 1 \
        --num_workers 8 \
        --iou_thr 0.1 \
        --conf_thres 0.2 \
        --scale_factor 1 
done