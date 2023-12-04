
python eval.py \
    --root_dir /mnt/ddisk/boux/code/data/simd/validation \
    --annotations_file /mnt/ddisk/boux/code/data/simd/val_coco.json \
    --save_dir /mnt/ddisk/boux/code/ovdsat/run/train/plots \
    --model_type yolo \
    --prototypes_path prototypes/simd_prototypes.pth \
    --ckpt /mnt/ddisk/boux/code/yolov5/runs/train/simd_train_all_train_no_freeze/weights/best.pt \
    --target_size 608 608\
    --batch_size 1 \
    --num_workers 8 \
    --iou_thr 0.2 \
    --conf_thres 0.001 \
    --scale_factor 1 \
    --save_images \