#!/bin/bash

for backbone in dinov2 clip-14 clip-32 georsclip-14 georsclip-32 remoteclip-14 remoteclip-32
do
    echo "Evaluating prototypes for ${backbone}"

    python eval.py \
        --root_dir /mnt/ddisk/boux/code/data/simd/validation \
        --annotations_file /mnt/ddisk/boux/code/data/simd/val_coco.json \
        --save_dir "run/eval/prototypes" \
        --prototypes_path run/train/learned_prototype_${backbone}/prototypes.pth \
        --bg_prototypes_path /mnt/ddisk/boux/code/ovdsat/run/backbone_prototypes/bg_prototypes_${backbone}.pt \
        --backbone_type ${backbone} \
        --target_size 602 602\
        --batch_size 1 \
        --num_workers 8 \
        --iou_thr 0.2 \
        --conf_thres 0.001 \
        --scale_factor 1 
done

    