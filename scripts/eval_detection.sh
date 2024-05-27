#!/bin/bash
DATA_DIR=data
backbone=dinov2
dataset=simd
finetune_type=boxes     # masks or boxes
N=10
INIT_PROTOTYPES_PATH=run/init_prototypes

python eval_detection.py \
    --dataset ${dataset} \
    --val_root_dir ${DATA_DIR}/${dataset}/val \
    --save_dir run/eval/detection/${dataset}/backbone_${backbone}_${finetune_type}/N${N} \
    --val_annotations_file ${DATA_DIR}/${dataset}/val_coco.json \
    --prototypes_path run/train/${finetune_type}/${dataset}_N${N}/prototypes.pth \
    --bg_prototypes_path ${INIT_PROTOTYPES_PATH}/boxes/${dataset}_N${N}/bg_prototypes_${backbone}.pt \
    --backbone_type ${backbone} \
    --classification box \
    --target_size 602 602 \
    --batch_size 1 \
    --num_workers 8 \
    --conf_thres 0.3 \
    --scale_factor 1 
