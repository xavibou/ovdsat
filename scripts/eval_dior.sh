#!/bin/bash
dataset=dior
for backbone in dinov2
do
    for N in 5 10 30
    do
        echo "Evaluating prototypes for ${backbone}"

        python eval.py \
            --root_dir /mnt/ddisk/boux/code/data/dior/DIOR/Annotations/subsets/val/images \
            --annotations_file /mnt/ddisk/boux/code/data/dior/DIOR/Annotations/subsets/val/val_coco_annotations.json \
            --dataset ${dataset} \
            --save_dir "run/eval/${dataset}/second_round_prototypes_${N}" \
            --prototypes_path run/train/learned_prototypes/dior_${backbone}_no_neg_${N}/prototypes.pth \
            --bg_prototypes_path run/train/learned_prototypes/dior_${backbone}_no_neg_${N}/bg_prototypes.pth \
            --backbone_type ${backbone} \
            --target_size 602 602\
            --batch_size 1 \
            --num_workers 8 \
            --iou_thr 0.2 \
            --conf_thres 0.001 \
            --scale_factor 1 
    done
done

    