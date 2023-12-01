
python train.py \
    --root_dir /mnt/ddisk/boux/code/data/dota2.0/preprocessed/train/images \
    --save_dir /mnt/ddisk/boux/code/ovdsat/run/train_with_scheduler_no_bg \
    --annotations_file /mnt/ddisk/boux/code/data/dota2.0/preprocessed/train/DOTA_2.0_subset_train_v2.json \
    --val_annotations_file /mnt/ddisk/boux/code/data/dota2.0/preprocessed/train/DOTA_2.0_subset_train_v2.json \
    --prototypes_path /mnt/ddisk/boux/code/devit/demo/dota_v2_prototypes.pth \
    --num_epochs 200 \
    --lr 2e-4 \
    --target_size 602 602\
    --batch_size 1 \
    --num_workers 8 \
    --iou_thr 0.1 \
    --conf_thres 0.2 \
    --scale_factor 1 \
