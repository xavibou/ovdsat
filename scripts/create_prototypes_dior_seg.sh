#!/bin/bash

for backbone in dinov2
do
    for N in 10
    do
        echo "Creating prototypes for ${backbone} with N=${N}"

        python build_prototypes.py \
            --data_dir /mnt/ddisk/boux/code/devit/datasets/sam_dior_N${N} \
            --save_dir /mnt/ddisk/boux/code/ovdsat/run/backbone_prototypes/segmentation/dior_N${N} \
            --annotations_file /mnt/ddisk/boux/code/devit/datasets/sam_dior_N10/modified_annotations.json \
            --src_data_dir /mnt/ddisk/boux/code/data/dior/DIOR/Annotations/yolo_annotations/train/images \
            --backbone_type ${backbone} \
            --target_size 602 602 \
            --window_size 224 \
            --scale_factor 1 \
            --num_b 10 \
            --k 200 \
            --store_bg_prototypes
    done
done