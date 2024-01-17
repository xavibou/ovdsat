#!/bin/bash

for backbone in dinov2
do
    echo "Evaluating prototypes for ${backbone}"
    num_neg=10
    N=30

    python eval.py \
        --root_dir /mnt/ddisk/boux/code/data/simd/validation \
        --dataset simd \
        --annotations_file /mnt/ddisk/boux/code/data/simd/val_coco.json \
        --save_dir "run/eval/learned_prototypes_neg_samples_N${N}_num_neg${num_neg}" \
        --prototypes_path /mnt/ddisk/boux/code/ovdsat/run/train/learned_prototype_simd_ablationdinov2_N${N}/neg_${num_neg}/prototypes.pth \
        --bg_prototypes_path /mnt/ddisk/boux/code/ovdsat/run/train/learned_prototype_simd_ablationdinov2_N${N}/neg_${num_neg}/bg_prototypes.pth \
        --backbone_type ${backbone} \
        --target_size 602 602\
        --batch_size 1 \
        --num_workers 8 \
        --iou_thr 0.2 \
        --conf_thres 0.001 \
        --scale_factor 1 
done

    