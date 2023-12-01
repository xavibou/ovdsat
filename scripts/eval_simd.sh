
python eval.py \
    --root_dir /mnt/ddisk/boux/code/data/simd/validation \
    --annotations_file /mnt/ddisk/boux/code/data/simd/val_coco.json \
    --save_dir /mnt/ddisk/boux/code/ovdsat/run/train/plots \
    --prototypes_path /mnt/ddisk/boux/code/ovdsat/run/train/simd/prototypes.pth \
    --bg_prototypes_path prototypes/simd_bg_prototypes.pth \
    --target_size 602 602\
    --batch_size 1 \
    --num_workers 8 \
    --iou_thr 0.2 \
    --conf_thres 0.001 \
    --scale_factor 1 \