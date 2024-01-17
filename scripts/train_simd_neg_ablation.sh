#!/bin/bash

# Define the values of N you want to iterate over
#for backbone in clip-14 clip-32 georsclip-32 remoteclip-14 remoteclip-32
backbone=dinov2
for N in 5 10 30
do
    for neg in 1 5 10
    do
        echo "Training ${backbone} for N=${N} and neg=${neg}"
        python train.py \
            --root_dir /mnt/ddisk/boux/code/data/simd/training \
            --save_dir "run/train/learned_prototype_simd_ablation${backbone}_N${N}/neg_${neg}" \
            --annotations_file /mnt/ddisk/boux/code/data/simd/train_coco_subset_N${N}.json \
            --val_annotations_file /mnt/ddisk/boux/code/data/simd/train_coco_subset_2.json \
            --prototypes_path "/mnt/ddisk/boux/code/ovdsat/prototypes/simcd_subset_N${N}/simd_prototypes.pth" \
            --bg_prototypes_path "/mnt/ddisk/boux/code/ovdsat/prototypes/simcd_subset_N${N}/simd_bg_prototypes.pth" \
            --backbone_type ${backbone} \
            --num_epochs 200 \
            --lr 2e-5 \
            --target_size 602 602 \
            --batch_size 1 \
            --num_neg ${neg} \
            --num_workers 8 \
            --iou_thr 0.1 \
            --conf_thres 0.2 \
            --scale_factor 1 
    done
done