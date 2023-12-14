#!/bin/bash
dataset=dior
for backbone in dinov2 clip-14 clip-32 georsclip-14 georsclip-32 remoteclip-14 remoteclip-32 openclip-14 openclip-32
do
    for N in 10 30 50
    do
        echo "Evaluating prototypes for ${backbone}"

        python eval.py \
            --root_dir /mnt/ddisk/boux/code/data/dior/DIOR/Annotations/yolo_annotations/val/images \
            --annotations_file /mnt/ddisk/boux/code/data/dior/DIOR/Annotations/yolo_annotations/val/val_coco_annotations.json \
            --dataset ${dataset} \
            --save_dir "run/eval/${dataset}/prototypes_${N}" \
            --prototypes_path /mnt/ddisk/boux/code/ovdsat/run/backbone_prototypes/dior_N${N}/prototypes_${backbone}.pt \
            --bg_prototypes_path /mnt/ddisk/boux/code/ovdsat/run/backbone_prototypes/dior_N${N}/bg_prototypes_${backbone}.pt \
            --backbone_type ${backbone} \
            --target_size 602 602\
            --batch_size 1 \
            --num_workers 8 \
            --iou_thr 0.2 \
            --conf_thres 0.001 \
            --scale_factor 1 
    done
done

    