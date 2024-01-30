#!/bin/bash

for backbone in dinov2
do
    echo "Evaluating prototypes for ${backbone}"

    python eval.py \
        --root_dir /mnt/ddisk/boux/code/data/simd/validation \
        --annotations_file /mnt/ddisk/boux/code/data/simd/val_coco.json \
        --save_dir "run/eval/prototypes" \
        --prototypes_path /mnt/ddisk/boux/code/ovdsat/run/backbone_prototypes/simd_N10/prototypes_dinov2.pt \
        --bg_prototypes_path /mnt/ddisk/boux/code/ovdsat/run/backbone_prototypes/simd_N10/bg_prototypes_dinov2.pt \
        --backbone_type ${backbone} \
        --target_size 602 602\
        --batch_size 1 \
        --num_workers 8 \
        --iou_thr 0.2 \
        --conf_thres 0.001 \
        --scale_factor 1 
done

    