import os
import cv2
import json
import torch
import random
import numpy as np
import albumentations as A
import albumentations.pytorch as Apy
from torch.utils.data import Dataset
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from torchvision import transforms as T

class DINODataset(Dataset):
    def __init__(self, root_dir, annotations_file, embedding_classes, augment=True, target_size=(800, 800)):
        self.images_dir = root_dir
        self.augment = augment
        self.target_size = target_size
        self.max_boxes = 500

        with open(annotations_file) as f:
            data = json.load(f)
            n_total = len(data)
        
        self.images = data.get('images', [])
        self.annotations = data.get('annotations', [])
        self.categories = data.get('categories', [])

        # Define a PyTorch image transform
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        
        # Define a list of Albumentations augmentations
        (w, h) = self.target_size
        if self.augment:
            # Define a list of transformations
            self.augs = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),  # Random 90-degree rotations
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                A.PadIfNeeded(min_height=1024, min_width=1024, border_mode=0, value=[0, 0, 0], p=0.5),
                A.RandomResizedCrop(height=h, width=w, scale=(0.5, 1), p=1),
            ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))
        else:
            self.augs = A.Compose([
                A.Resize(h, w),
            ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))
    
    def __len__(self):
        return len(self.images)
    
    def get_categories(self):
        #return {idx: label['name'] for idx, label in enumerate(self.categories)}
        return {
                0: 'ship',
                1: 'harbor',
                2: 'baseballfield',
                3: 'groundtrackfield',
                4: 'chimney',
                5: 'vehicle',
                6: 'airport',
                7: 'golffield',
                8: 'overpass',
                9: 'bridge',
                10: 'Expressway-toll-station',
                11: 'stadium',
                12: 'tenniscourt',
                13: 'storagetank',
                14: 'airplane',
                15: 'trainstation',
                16: 'Expressway-Service-area',
                17: 'windmill',
                18: 'dam',
                19: 'basketballcourt'
            }
    
    def get_category_number(self):
        # return number of categories
        return len(self.categories)

    def load_image(self, idx: int):
        filename = self.images[idx]['file_name']
        path = os.path.join(self.images_dir, filename)
        image = cv2.imread(path)
        return image, path

    def load_target(self, idx: int):
        image_id = self.images[idx]['id']
        annotations = [ann for ann in self.annotations if ann['image_id'] == image_id]

        labels = []
        boxes = []
        masks = []
        for annotation in annotations:
            
            if annotation["bbox"][-1] < 1 or annotation["bbox"][-2] < 1:
                continue
            
            labels.append(annotation["category_id"])
            boxes.append(annotation["bbox"] if "bbox" in annotation else [])    
            if "segmentation" in annotation and len(annotation["segmentation"]) > 0:
                masks.append(np.array(annotation["segmentation"]) * 1)
        return labels, boxes, masks



    def __getitem__(self, idx):
        image, path = self.load_image(idx)
        labels, boxes, masks = self.load_target(idx)
        valid_masks = True if len(masks) > 0 else False

        _, w, h = image.shape
        metadata = {
            "width": w,
            "height": h,
            "impath": path,
        }

        # Apply augmentations        
        transformed = self.augs(image=image, bboxes=boxes, category_ids=labels, masks=masks if valid_masks else None)
        image = transformed['image']
        boxes = transformed['bboxes']
        masks = transformed['masks']
        labels = transformed['category_ids']

        # Pad bounding boxes, masks and labels to a fixed size
        padded_boxes = torch.tensor(boxes + [(0,0,0,0)] * (self.max_boxes - len(boxes))).float()
        padded_labels = torch.tensor(labels + [-1] * (self.max_boxes - len(labels))).float()
        padded_masks = torch.tensor(np.array(masks + [np.zeros(self.target_size) for i in range(self.max_boxes - len(masks))])).float() if valid_masks else []

        return torch.as_tensor(image.astype("float32").transpose(2, 0, 1)), padded_boxes, padded_labels, padded_masks, metadata
