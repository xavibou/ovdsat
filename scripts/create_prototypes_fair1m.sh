#!/bin/bash

echo "Creating prototypes for ${backbone} with N=${N}"

backbone=dinov2

python build_prototypes.py \
    --data_dir /mnt/ddisk/boux/code/devit/datasets/fair1m_seg_N10 \
    --save_dir /mnt/ddisk/boux/code/ovdsat/run/backbone_prototypes/fair1m/N10 \
    --annotations_file /mnt/ddisk/boux/code/data/fair1m/train/preprocessed/fair1m_train_2.0_subset_N10.json \
    --src_data_dir /mnt/ddisk/boux/code/data/fair1m/train/preprocessed/images \
    --backbone_type ${backbone} \
    --target_size 602 602 \
    --window_size 224 \
    --scale_factor 1 \
    --num_b 5 \
    --k 200 \

