
python train.py \
    --root_dir /mnt/ddisk/boux/code/data/simd/training \
    --save_dir "run/train/simd_experiments/backbone_analysis/dinov2" \
    --annotations_file /mnt/ddisk/boux/code/data/simd/train_coco_subset_N10.json \
    --val_annotations_file /mnt/ddisk/boux/code/data/simd/train_coco_subset_2.json \
    --prototypes_path "/mnt/ddisk/boux/code/ovdsat/run/classification_benchmark_exp/prototypes_dinov2.pt" \
    --num_epochs 200 \
    --lr 2e-5 \
    --target_size 602 602 \
    --batch_size 1 \
    --num_workers 8 \
    --iou_thr 0.1 \
    --conf_thres 0.2 \
    --scale_factor 1 \
    --only_train_prototypes