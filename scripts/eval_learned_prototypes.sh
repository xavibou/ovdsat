#!/bin/bash

for backbone in dinov2 clip-14 clip-32 georsclip-14 georsclip-32 remoteclip-14 remoteclip-32
do
    echo "Evaluating prototypes for ${backbone}"

    python eval.py \
        --root_dir /mnt/ddisk/boux/code/data/simd/validation \
        --annotations_file /mnt/ddisk/boux/code/data/simd/val_coco.json \
        --save_dir "run/eval/learned_prototypes_neg_samples" \
        --prototypes_path run/train/learned_prototype_${backbone}_neg_samples/prototypes.pth \
        --bg_prototypes_path run/train/learned_prototype_${backbone}_neg_samples/bg_prototypes.pth \
        --backbone_type ${backbone} \
        --target_size 602 602\
        --batch_size 1 \
        --num_workers 8 \
        --iou_thr 0.2 \
        --conf_thres 0.001 \
        --scale_factor 1 
done

    