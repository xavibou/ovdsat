#!/bin/bash

DATA_DIR=/mnt/ddisk/boux/code/data
INIT_PROTOTYPES_PATH=run/segmentation_init_prototypes
backbone=dinov2

for dataset in simd dior
do
    for N in 5 10 30
    do
        echo "Training mask classifier model for the ${dataset} dataset using ${backbone} features with N=${N}"
        python train.py \
            --train_root_dir  ${DATA_DIR}/${dataset}/train \
            --val_root_dir  ${DATA_DIR}/${dataset}/train \
            --save_dir "run/train/segmentation/${dataset}_N${N}" \
            --train_annotations_file ${DATA_DIR}/${dataset}/train_coco_subset_N${N}_seg.json \
            --val_annotations_file ${DATA_DIR}/${dataset}/train_coco_finetune_val_seg.json \
            --prototypes_path ${INIT_PROTOTYPES_PATH}/${dataset}_N${N}/prototypes_${backbone}.pt \
            --backbone_type ${backbone} \
            --num_epochs 200 \
            --lr 2e-4 \
            --target_size 602 602 \
            --batch_size 4 \
            --num_neg 0 \
            --num_workers 0 \
            --iou_thr 0.1 \
            --conf_thres 0.2 \
            --scale_factor 1 \
            --use_segmentation 
    done
done
