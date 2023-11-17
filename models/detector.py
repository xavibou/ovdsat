import cv2
import torch
import torchvision.transforms as T
from detectron2.structures import ImageList
from models.classifier import OVDClassifier

from utils_dir.rpn_utils import get_RPN
from utils_dir.processing_utils import filter_boxes
from utils_dir.nms import non_max_suppression


class OVDDetector(torch.nn.Module):

    def __init__(self,
                prototypes,
                bg_prototypes=None,
                dino_version='dinov2_vitl14',
                target_size=(560,560),
                scale_factor=2,
                min_box_size=5,
                ignore_index=-1,
                rpn_config='configs/CenterNet2_R50_1x.yaml',
                rpn_checkpoint='/mnt/ddisk/boux/code/dino_simple_detector/CenterNet2/models/CenterNet2_R50_1x.pth'
                ):
        super().__init__()
        self.scale_factor = scale_factor
        self.target_size = target_size
        self.min_box_size = min_box_size
        self.ignore_index = ignore_index
        self.class_names = prototypes['label_names']
        self.num_classes = len(self.class_names)

        if bg_prototypes is not None:
            all_prototypes = torch.cat([prototypes['prototypes'], bg_prototypes['prototypes']]).float()
        else:
            all_prototypes = prototypes['prototypes']

        self.classifier = OVDClassifier(all_prototypes, dino_version, target_size, scale_factor, min_box_size, ignore_index)
        self.rpn_cfg, self.rpn = get_RPN(rpn_config, rpn_checkpoint)
    
        self.mean = torch.tensor([0.485, 0.456, 0.406]).to(self.rpn_cfg.MODEL.DEVICE)
        self.std = torch.tensor([0.229, 0.224, 0.225]).to(self.rpn_cfg.MODEL.DEVICE)

    def generate_proposals(self, images):
        images = [(x - self.rpn.model.pixel_mean) / self.rpn.model.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.rpn.model.backbone.size_divisibility,
            padding_constraints=self.rpn.model.backbone.padding_constraints,
        )
        features = self.rpn.model.backbone(images.tensor)
        proposals, _ = self.rpn.model.proposal_generator(images, features, None)
        
        boxes = torch.stack([p.proposal_boxes.tensor for p in proposals])
        box_scores = torch.stack([p.scores for p in proposals])
        return boxes, box_scores
    

    def prepare_for_dino(self, input_tensor):
        # Scale the values to range from 0 to 1
        input_tensor /= 255.0
        
        # Normalize the tensor
        normalized_tensor = (input_tensor - self.mean[:, None, None]) / self.std[:, None, None]
        return normalized_tensor
    
    def forward(self, images, iou_thr=0.1, conf_thres=0.1, box_conf_threshold=0.1):

        # Generate box proposals
        proposals, box_scores = self.generate_proposals(images)

        # Classify boxes with classifier
        B, num_proposals, _ = proposals.shape
        # TODO: fix to work with B>2
        preds = self.classifier(self.prepare_for_dino(images), proposals, normalize=True)

        # Extract class scores and predicted classes
        preds = preds.reshape(B, num_proposals, -1)
        scores, _ = torch.max(preds, dim=-1)
        classes = torch.argmax(preds, dim=-1)

        # Filter predictions and prepare for NMS
        processed_predictions = []
        for b in range(B):
            filtered_boxes, filtered_classes, filtered_scores = filter_boxes(proposals[b],
                                                                             preds[b],
                                                                             box_scores[b],
                                                                             self.target_size,
                                                                             self.num_classes,
                                                                             box_conf_threshold)
            pred_boxes_with_scores = torch.cat([filtered_boxes, filtered_scores[:, None], filtered_classes], dim=1)

            # Use the cosine similarity class score as box confidence scores
            max_cls_scores, _ = torch.max(filtered_classes, dim=-1)
            sorted_indices = torch.argsort(filtered_scores, descending=True)
            pred_boxes_with_scores = pred_boxes_with_scores[sorted_indices]

            # Apply non maximum suppression
            nms_results = non_max_suppression(pred_boxes_with_scores.unsqueeze(0), iou_thres=iou_thr, conf_thres=conf_thres)
            processed_predictions.append(nms_results[0])

        del preds, scores, classes, proposals, box_scores
        
        return processed_predictions


def reorder_boxes_by_score(boxes_tensor):
    # Assuming boxes_tensor is of shape [N, K]
    # Extract box scores (fifth element in the second dimension)
    box_scores = boxes_tensor[:, 4]

    # Sort indices of the boxes based on box scores in descending order
    sorted_indices = torch.argsort(box_scores, descending=True)

    # Reorder the boxes tensor using the sorted indices
    sorted_boxes = boxes_tensor[sorted_indices]

    return sorted_boxes