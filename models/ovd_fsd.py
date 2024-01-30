import cv2
import torch
import torchvision.transforms as T
from detectron2.structures import ImageList
from utils_dir.processing_utils import filter_boxes
from utils_dir.nms import non_max_suppression
from utils_dir.backbones_utils import prepare_image_for_backbone
from models.owlvit import load_OWLViT


class OVDFSDModel(torch.nn.Module):

    def __init__(self,
                model='owlvit',
                target_size=(602,602),
                scale_factor=2,
                min_box_size=5,
                ignore_index=-1,
                ):
        super().__init__()
        self.target_size = target_size
        self.ignore_index = ignore_index
        self.class_names = prototypes['label_names']
        self.num_classes = len(self.class_names)
        self.model = model

        self.detector = load_model(self.model)
    

    def load_model(self, model):
        
        if model == 'owlvit':
            detector = load_OWLViT(self.class_names, device)
        else:
            raise ValueError('Model not supported')

        return detector

    def forward(self, images, iou_thr=0.2, conf_thres=0.001, box_conf_threshold=0.01):
        
        
        
        return processed_predictions
