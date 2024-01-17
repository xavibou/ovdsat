
backbone=clip-14
python eval_devit.py \
    --root_dir /mnt/ddisk/boux/code/data/simd/validation \
    --annotations_file /mnt/ddisk/boux/code/data/simd/val_coco.json \
    --prototypes_path /mnt/ddisk/boux/code/ovdsat/run/train/simd_experiments/backbone_analysis/clip-14/prototypes.pth \
    --bg_prototypes_path /mnt/ddisk/boux/code/ovdsat/run/train/simd_experiments/backbone_analysis/clip-14/bg_prototypes.pth \
    --backbone_type ${backbone} \
    --target_size 602 602\
    --batch_size 1 \
    --num_workers 8 \
    --iou_thr 0.2 \
    --conf_thres 0.001 \
    --scale_factor 1 \