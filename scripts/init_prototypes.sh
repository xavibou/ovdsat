#!/bin/bash

DATA_DIR=data
backbone=dinov2
for dataset in simd
do
    for N in 5 10 30
    do
        echo "Creating prototypes for the ${dataset} dataset using ${backbone} features with N=${N}"

        python build_prototypes.py \
            --data_dir ${DATA_DIR}/${dataset}/train \
            --save_dir run/init_prototypes/boxes/${dataset}_N${N} \
            --annotations_file ${DATA_DIR}/${dataset}/train_coco_subset_N${N}.json \
            --backbone_type ${backbone} \
            --target_size 602 602 \
            --window_size 224 \
            --scale_factor 1 \
            --num_b 10 \
            --k 200 \
            --store_bg_prototypes
    done
done