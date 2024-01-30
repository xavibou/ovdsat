#!/bin/bash

for N in 5 10 30
do
    echo "Evaluating prototypes for ${N}"

    python eval.py \
        --root_dir /mnt/ddisk/boux/code/data/dior/DIOR/Annotations/subsets/val/images \
        --annotations_file /mnt/ddisk/boux/code/data/dior/DIOR/Annotations/subsets/val/val_coco_annotations.json \
        --save_dir "run/eval/dior_yolo/results_N${N}_no_freeze" \
        --dataset dior \
        --model_type yolo \
        --prototypes_path /mnt/ddisk/boux/code/ovdsat/run/backbone_prototypes/dior_N5/prototypes_dinov2.pt \
        --ckpt /mnt/ddisk/boux/code/yolov5/runs/train/dior_train_N${N}_no_freeze/weights/best.pt \
        --target_size 608 608\
        --batch_size 1 \
        --num_workers 8 \
        --iou_thr 0.2 \
        --conf_thres 0.001 \
        --scale_factor 1 

    python eval.py \
        --root_dir /mnt/ddisk/boux/code/data/dior/DIOR/Annotations/subsets/val/images \
        --annotations_file /mnt/ddisk/boux/code/data/dior/DIOR/Annotations/subsets/val/val_coco_annotations.json \
        --save_dir "run/eval/dior_yolo/results_N${N}_frozen" \
        --dataset dior \
        --model_type yolo \
        --prototypes_path /mnt/ddisk/boux/code/ovdsat/run/backbone_prototypes/dior_N5/prototypes_dinov2.pt \
        --ckpt /mnt/ddisk/boux/code/yolov5/runs/train/dior_train_N${N}_frozen_backbone/weights/best.pt \
        --target_size 608 608\
        --batch_size 1 \
        --num_workers 8 \
        --iou_thr 0.2 \
        --conf_thres 0.001 \
        --scale_factor 1 
done
