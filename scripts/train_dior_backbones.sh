#!/bin/bash

# Define the values of N you want to iterate over
for backbone in dinov2 clip-32 clip-14 georsclip-32 georsclip-14 remoteclip-32 remoteclip-14
do
    for N in 10 30 50
    do
        echo "Training ${backbone} with N=${N}"
        python train.py \
            --root_dir /mnt/ddisk/boux/code/data/dior/DIOR/Annotations/yolo_annotations/train/images \
            --save_dir "run/train/learned_prototypes/dior_${backbone}_neg_samples_${N}" \
            --annotations_file /mnt/ddisk/boux/code/data/dior/DIOR/Annotations/yolo_annotations/train/train_coco_subset_N${N}.json \
            --val_annotations_file /mnt/ddisk/boux/code/data/dior/DIOR/Annotations/yolo_annotations/train/train_coco_subset_N15_for_val.json \
            --prototypes_path "/mnt/ddisk/boux/code/ovdsat/run/backbone_prototypes/dior_N${N}/prototypes_${backbone}.pt" \
            --bg_prototypes_path "/mnt/ddisk/boux/code/ovdsat/run/backbone_prototypes/dior_N${N}/bg_prototypes_${backbone}.pt" \
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
done