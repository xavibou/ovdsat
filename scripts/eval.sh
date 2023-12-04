
python eval.py \
    --root_dir /mnt/ddisk/boux/code/data/dota2.0/preprocessed/val/images \
    --annotations_file /mnt/ddisk/boux/code/data/dota2.0/preprocessed/val/DOTA_1.5_val.json \
    --prototypes_path /mnt/ddisk/boux/code/devit/demo/dota_v2_prototypes.pth \
    --bg_prototypes_path /mnt/ddisk/boux/code/devit/weights/initial/background/dota_v2_bg_prototypes.pth \
    --target_size 602 602\
    --batch_size 1 \
    --num_workers 8 \
    --iou_thr 0.1 \
    --conf_thres 0.1 \
    --scale_factor 1 2 \
    --save_images \
    --save_dir /mnt/ddisk/boux/code/ovdsat/run/train/plots \