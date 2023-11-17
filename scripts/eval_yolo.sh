
python eval.py \
    --root_dir /mnt/ddisk/boux/code/data/dota2.0/preprocessed/val/images \
    --annotations_file /mnt/ddisk/boux/code/data/dota2.0/preprocessed/val/DOTA_1.5_val.json \
    --model_type yolo \
    --prototypes_path /mnt/ddisk/boux/code/devit/demo/dota_v2_prototypes.pth \
    --ckpt /mnt/ddisk/boux/code/yolov5/runs/train/dota_N-10-30/weights/best.pt \
    --target_size 602 602\
    --batch_size 1 \
    --num_workers 8 \
    --iou_thr 0.1 \
    --conf_thres 0.2 \
    --scale_factor 1 2\