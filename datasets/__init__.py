import torch
import albumentations as A
from torch.utils.data import DataLoader
from datasets.dataset import BoxDataset, OBBDataset


def get_base_new_classes(dataset):
    '''
    Returns the base and new classes for the given dataset.
    '''
    
    if dataset == 'simd':
        base_classes = ['car', 'helicopter', 'boat', 'long-vehicle']
        new_classes = ['trainer-aircraft', 'pushback-truck', 'propeller-aircraft', 'truck',
                        'charted-aircraft', 'figther-aircraft', 'van', 'airliner', 'stair-truck', 'bus']
    elif dataset == 'dior':
        base_classes = ['airplane', 'baseballfield', 'basketballcourt', 'groundtrackfield', 'harbor', 'ship', 'tenniscourt', 'storagetank']
        new_classes = ['airport', 'bridge', 'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station', 'golffield', 'overpass', 'stadium', 'windmill', 'trainstation', 'vehicle']

    return base_classes, new_classes

def init_dataloaders(args):
    train_annotations_file = getattr(args, 'train_annotations_file', None)
    val_annotations_file = getattr(args, 'val_annotations_file', None)
    annotations = getattr(args, 'annotations', None)
    w, h = args.target_size

    if annotations == 'box':
        dataClass = BoxDataset
    elif annotations == 'obb':
        dataClass = OBBDataset
    else:
        raise ValueError(f"Invalid annotations type: {annotations}")

    if train_annotations_file is not None:
        # Define training augmentations
        train_augmentations = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),  # Random 90-degree rotations
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            A.PadIfNeeded(min_height=1024, min_width=1024, border_mode=0, value=[0, 0, 0], p=0.5),
            A.RandomResizedCrop(height=h, width=w, scale=(0.5, 1), p=1),
        ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

        train_dataset = dataClass(
            args.train_root_dir,
            args.train_annotations_file,
            augmentations=train_augmentations,
            target_size=args.target_size
        )
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.num_workers
        )
    else:
        train_dataloader = None

    if val_annotations_file is not None:
        val_augmentations = A.Compose([
            A.Resize(h, w),
        ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

        val_dataset = dataClass(
            args.val_root_dir,
            args.val_annotations_file,
            augmentations=val_augmentations,
            target_size=args.target_size,
        )
        val_dataset.dynamic_size = True if args.batch_size == 1 else False
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers
        )

    else:
        val_dataloader = None

    return train_dataloader, val_dataloader
