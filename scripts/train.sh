
python train.py \
    --root_dir /mnt/ddisk/boux/code/data/dota2.0/preprocessed/train/images \
    --annotations_file /mnt/ddisk/boux/code/data/dota2.0/preprocessed/train/DOTA_2.0_subset_train_v2.json \
    --prototypes_path /mnt/ddisk/boux/code/devit/demo/dota_v2_prototypes.pth \
    --bg_prototypes_path /mnt/ddisk/boux/code/devit/weights/initial/background/dota_v2_bg_prototypes.pth \
    --num_epochs 100 \
    --lr 2e-4 \
    --target_size 602 602\
    --batch_size 1 \
    --num_workers 8 \
    --iou_thr 0.1 \
    --conf_thres 0.2 \
    --scale_factor 1 2\
