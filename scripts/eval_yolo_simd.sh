
python eval.py \
    --root_dir /mnt/ddisk/boux/code/data/simd/validation \
    --annotations_file /mnt/ddisk/boux/code/data/simd/val_coco.json \
    --model_type yolo \
    --prototypes_path prototypes/simd_prototypes.pth \
    --ckpt /mnt/ddisk/boux/code/yolov5/runs/train/simd_train/weights/best.pt \
    --target_size 672 672\
    --batch_size 1 \
    --num_workers 8 \
    --iou_thr 0.1 \
    --conf_thres 0.2 \
    --scale_factor 1 2 \