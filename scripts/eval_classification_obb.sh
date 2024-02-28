#!/bin/bash

DATA_DIR=/mnt/ddisk/boux/code/data
backbone=dinov2
annotations=obb

for dataset in dior
do
    for N in 10 
    do
        python eval_classification.py \
            --val_root_dir ${DATA_DIR}/${dataset}/val \
            --save_dir "run/eval/classification/obbs/${dataset}_N${N}" \
            --val_annotations_file ${DATA_DIR}/${dataset}/val_coco_with_obbs.json \
            --prototypes_path run/train/obbs/${dataset}_N${N}/prototypes.pth \
            --annotations ${annotations} \
            --backbone_type ${backbone} \
            --target_size 602 602 \
            --batch_size 1 \
            --num_workers 8 \
            --scale_factor 1 
    done
done

        
