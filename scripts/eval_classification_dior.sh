#!/bin/bash

# Define the values of N you want to iterate over
#for backbone in clip-14 clip-32 georsclip-32 remoteclip-14 remoteclip-32
backbone=dinov2
dataset=dior
N=10

python eval_classification.py \
    --root_dir /mnt/ddisk/boux/code/data/dior/DIOR/Annotations/subsets/val/images \
    --save_dir "run/train/segmentation/learned_prototype_${backbone}_mean_aggr_weighted_cosim" \
    --val_annotations_file /mnt/ddisk/boux/code/data/dior/DIOR/Annotations/subsets/val/val_coco_annotations.json \
    --prototypes_path /mnt/ddisk/boux/code/ovdsat/run/backbone_prototypes/dior_N10/prototypes_dinov2.pt \
    --backbone_type ${backbone} \
    --target_size 602 602 \
    --batch_size 16 \
    --num_workers 8 \
    --scale_factor 1 


# On the screen on top we used the trainned with bboxes N=10
# On the screen below there is the one with segmentation training.