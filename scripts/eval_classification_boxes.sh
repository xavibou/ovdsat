#!/bin/bash

DATA_DIR=data
backbone=dinov2
annotations=box

for dataset in simd
do
    for N in 5 10 30
    do
        python eval_classification.py \
            --val_root_dir ${DATA_DIR}/${dataset}/val \
            --save_dir "run/eval/classification/boxes/${dataset}_N${N}" \
            --val_annotations_file ${DATA_DIR}/${dataset}/val_coco.json \
            --prototypes_path run/train/boxes/${dataset}_N${N}/prototypes.pth \
            --annotations ${annotations} \
            --backbone_type ${backbone} \
            --target_size 602 602 \
            --batch_size 8 \
            --num_workers 8 \
            --scale_factor 1 
    done
done

        
