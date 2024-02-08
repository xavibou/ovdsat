import os
import cv2
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from torchvision import transforms as T

class DINODataset(Dataset):
    def __init__(self, root_dir, annotations_file, augmentations=None, target_size=(800, 800)):
        self.images_dir = root_dir
        self.augmentations = augmentations
        self.target_size = target_size
        self.augmentations = augmentations
        self.max_boxes = 500

        # Read the annotations file
        with open(annotations_file) as f:
            data = json.load(f)
            n_total = len(data)
        
        # Extract the images, annotations, and categories
        self.images = data.get('images', [])
        self.annotations = data.get('annotations', [])
        self.categories = data.get('categories', [])
        
    def __len__(self):
        return len(self.images)
    
    def get_categories(self):
        return {
            0: 'ship', 1: 'harbor', 2: 'baseballfield', 3: 'groundtrackfield',
            4: 'chimney', 5: 'vehicle', 6: 'airport', 7: 'golffield',
            8: 'overpass', 9: 'bridge', 10: 'Expressway-toll-station',
            11: 'stadium', 12: 'tenniscourt', 13: 'storagetank', 14: 'airplane',
            15: 'trainstation', 16: 'Expressway-Service-area', 17: 'windmill',
            18: 'dam', 19: 'basketballcourt'
        }
    
    def get_category_number(self):
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
        if self.augmentations:
            transformed = self.augmentations(
                image=image, 
                bboxes=boxes, 
                category_ids=labels, 
                masks=masks if valid_masks else None
            )
            image = torch.as_tensor(transformed['image'].astype("float32").transpose(2, 0, 1))
            boxes = transformed['bboxes']
            masks = transformed['masks']
            labels = transformed['category_ids']
        else:
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        # Pad bounding boxes and labels to a fixed size based on the number of annotations
        num_boxes = len(boxes)
        padded_boxes = torch.tensor(boxes).float()
        padded_labels = torch.tensor(labels).float()

        # Pad masks to a fixed size based on the number of annotations
        if valid_masks:
            padded_masks = torch.tensor(masks).float()
        else:
            padded_masks = []

        # Pad to the maximum number of boxes
        if num_boxes < self.max_boxes:
            pad_boxes = torch.zeros((self.max_boxes - num_boxes, 4))
            pad_labels = torch.full((self.max_boxes - num_boxes,), -1)
            padded_boxes = torch.cat([padded_boxes, pad_boxes], dim=0)
            padded_labels = torch.cat([padded_labels, pad_labels], dim=0)
            if valid_masks:
                pad_masks = torch.zeros((self.max_boxes - num_boxes, self.target_size[0], self.target_size[1]))
                padded_masks = torch.cat([padded_masks, pad_masks], dim=0)

        return image, padded_boxes, padded_labels, padded_masks, metadata
