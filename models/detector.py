import cv2
import torch
import torchvision.transforms as T
from detectron2.structures import ImageList
from models.classifier import OVDClassifier

from utils_dir.rpn_utils import get_RPN
from utils_dir.processing_utils import filter_boxes
from utils_dir.nms import non_max_suppression
from utils_dir.backbones_utils import prepare_image_for_backbone

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

class OVDDetector(torch.nn.Module):

    def __init__(self,
                prototypes,
                bg_prototypes=None,
                backbone_type='dinov2',
                target_size=(560,560),
                scale_factor=2,
                min_box_size=5,
                ignore_index=-1,
                rpn_config='/mnt/ddisk/boux/code/DroneDetectron2/configs/Dota-Base-RCNN-FPN.yaml',
                rpn_checkpoint='/mnt/ddisk/boux/code/DroneDetectron2/outputs_FPN_DOTA/model_final.pth'
                ):
        super().__init__()
        self.scale_factor = scale_factor
        self.target_size = target_size
        self.min_box_size = min_box_size
        self.ignore_index = ignore_index
        self.class_names = prototypes['label_names']
        self.num_classes = len(self.class_names)
        self.backbone_type = backbone_type

        if bg_prototypes is not None:
            all_prototypes = torch.cat([prototypes['prototypes'], bg_prototypes['prototypes']]).float()
        else:
            all_prototypes = prototypes['prototypes']

        self.classifier = OVDClassifier(all_prototypes, backbone_type, target_size, scale_factor, min_box_size, ignore_index)
        self.rpn_cfg, self.rpn = get_RPN(rpn_config, rpn_checkpoint)
    

    def generate_proposals(self, images):
        images = [(x - self.rpn.pixel_mean) / self.rpn.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.rpn.backbone.size_divisibility,
            padding_constraints=self.rpn.backbone.padding_constraints,
        )
        features = self.rpn.backbone(images.tensor)
        proposals, _ = self.rpn.proposal_generator(images, features, None)
        
        boxes = torch.stack([p.proposal_boxes.tensor for p in proposals])
        #box_scores = torch.stack([sigmoid(p.objectness_logits) for p in proposals])
        box_scores = torch.stack([p.objectness_logits / 10 for p in proposals])

        return boxes, box_scores

    def forward(self, images, iou_thr=0.2, conf_thres=0.001, box_conf_threshold=0.01):

        # Generate box proposals
        proposals, box_scores = self.generate_proposals(images)

        # Classify boxes with classifier
        B, num_proposals, _ = proposals.shape
        # TODO: fix to work with B>2
        preds = self.classifier(prepare_image_for_backbone(images, self.backbone_type), proposals, normalize=True)

        #breakpoint()

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
