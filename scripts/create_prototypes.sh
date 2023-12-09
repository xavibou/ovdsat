#!/bin/bash

for backbone in dinov2 clip-14 clip-32 georsclip-14 georsclip-32 remoteclip-14 remoteclip-32 openclip-14 openclip-32
do
    echo "Creating prototypes for ${backbone}"

    python build_prototypes.py \
        --data_dir /mnt/ddisk/boux/code/devit/datasets/simd_subset_10 \
        --save_dir /mnt/ddisk/boux/code/ovdsat/run/backbone_prototypes \
        --annotations_file /mnt/ddisk/boux/code/data/simd/train_coco_subset_N10.json \
        --src_data_dir /mnt/ddisk/boux/code/data/simd/training \
        --backbone_type ${backbone} \
        --target_size 602 602 \
        --window_size 224 \
        --scale_factor 1 \
        --num_b 10 \
        --k 200 \
        --store_bg_prototypes
done