#!/bin/bash

DATA_DIR=/mnt/ddisk/boux/code/data
backbone=dinov2
for dataset in dior
do
    for N in 5 10 30
    do
        echo "Creating prototypes for the ${dataset} dataset using ${backbone} features with N=${N}"

        python build_prototypes.py \
            --data_dir /mnt/ddisk/boux/code/devit/datasets/${dataset}_N${N}_obbs \
            --save_dir /mnt/ddisk/boux/code/ovdsat/run/init_prototypes_obbs/${dataset}_N${N} \
            --annotations_file ${DATA_DIR}/${dataset}/train_coco_subset_N${N}_with_obbs.json \
            --src_data_dir ${DATA_DIR}/${dataset}/train \
            --backbone_type ${backbone} \
            --target_size 602 602 \
            --scale_factor 1 \
            --num_b 10 \
            --k 200 \
            #--store_bg_prototypes
    done
done