import os
import cv2
import json
import torch
import random
import albumentations as A
import albumentations.pytorch as Apy
from torch.utils.data import Dataset
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from torchvision import transforms as T

class DINODataset(Dataset):
    def __init__(self, root_dir, annotations_file, embedding_classes, augment=True, target_size=(800, 800), real_indices=False):
        self.images_dir = root_dir
        self.augment = augment
        self.target_size = target_size
        self.embedding_classes = embedding_classes
        self.max_boxes = 100
        self.real_indices = real_indices

        with open(annotations_file) as f:
            data = json.load(f)
            n_total = len(data)

        data = [{k: v} for k, v in data.items() if len(v)]
        
        isinfo = [i for i in data[0].items()]
        if isinfo[0][0] == 'info':
            data = [data[1], data[3], data[2]]

        self.images = data[0]['images']
        self.annotations = data[1]['annotations']
        categories = data[2]['categories']

        self.names = {idx: label for idx, label in enumerate(self.embedding_classes)}
        self.map_id_to_name = {entry['id']: entry['name'] for entry in categories}
        self.counts, self.class_weights = self.get_annotation_counts(self.annotations, categories)

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

    def get_annotation_counts(self, annotations, categories):

        label_counts = {category: 0 for category in self.embedding_classes}


        for annotation in annotations:
            category_id = annotation['category_id']
            category_name = self.map_id_to_name.get(category_id)
            label_counts[category_name] += 1
    
        total_classes = len(self.embedding_classes)
        class_weights = [total_classes / (label_counts[label] * total_classes) for label in self.embedding_classes]
        return label_counts, torch.Tensor(class_weights)

    def map_class_id_to_embedding(self, id):
        target_name = self.map_id_to_name.get(id)
        return self.embedding_classes.index(target_name) 

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
        for annotation in annotations:
            if annotation["bbox"][-1] < 1 or annotation["bbox"][-2] < 1:
                continue
            if self.real_indices:
                labels.append(annotation["category_id"] - 1)
            else:
                labels.append(self.map_class_id_to_embedding(annotation["category_id"]))
            boxes.append(annotation["bbox"])    
        return labels, boxes
    
    def get_by_path(self, path):

        image = cv2.imread(path)
        _, w, h = image.shape
        metadata = {
            "width": w,
            "height": h,
            "impath": path,
        }

        # Apply augmentations
        transformed = self.augs(image=image, bboxes=[], category_ids=[])
        image = transformed['image']

        return torch.as_tensor(image.astype("float32").transpose(2, 0, 1)), metadata
        
    def __getitem__(self, idx):
        image, path = self.load_image(idx)
        labels, boxes = self.load_target(idx)

        _, w, h = image.shape
        metadata = {
            "width": w,
            "height": h,
            "impath": path,
        }

        # Apply augmentations        
        transformed = self.augs(image=image, bboxes=boxes, category_ids=labels)
        image = transformed['image']
        boxes = transformed['bboxes']
        labels = transformed['category_ids']

        # Pad bounding boxes and labels to a fixed size
        padded_boxes = boxes + [(0,0,0,0)] * (self.max_boxes - len(boxes))
        padded_labels = labels + [-1] * (self.max_boxes - len(labels))

        return torch.as_tensor(image.astype("float32").transpose(2, 0, 1)), torch.tensor(padded_boxes), torch.tensor(padded_labels), metadata
        