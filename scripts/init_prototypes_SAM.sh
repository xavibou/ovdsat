#!/bin/bash

backbone=dinov2
for dataset in simd
do
    if [ $dataset == "simd" ]
    then
        src_img_dir=/mnt/ddisk/boux/code/data/simd/yolo/train/images
    else
        src_img_dir=/mnt/ddisk/boux/code/data/dior/DIOR/Annotations/subsets/train/images
    fi
    for N in 5 10 30
    do
        echo "Creating prototypes for the ${dataset} dataset using ${backbone} features with N=${N}"

        python build_prototypes.py \
            --data_dir /mnt/ddisk/boux/code/devit/datasets/sam_${dataset}_N${N} \
            --save_dir /mnt/ddisk/boux/code/ovdsat/run/segmentation_init_prototypes/${dataset}_N${N} \
            --annotations_file /mnt/ddisk/boux/code/devit/datasets/sam_${dataset}_N${N}/modified_annotations.json \
            --src_data_dir /mnt/ddisk/boux/code/data/dior/DIOR/Annotations/yolo_annotations/train/images \
            --backbone_type ${backbone} \
            --target_size 602 602 \
            --window_size 224 \
            --scale_factor 1 \
            --num_b 10 \
            --k 200 \
            #--store_bg_prototypes
    done
done