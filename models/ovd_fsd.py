import cv2
import torch
import torchvision.transforms as T
from detectron2.structures import ImageList
from utils_dir.processing_utils import filter_boxes
from utils_dir.nms import non_max_suppression
from utils_dir.backbones_utils import prepare_image_for_backbone
from transformers import AutoProcessor, OwlViTForObjectDetection


class OVDFSDModel(torch.nn.Module):

    def __init__(self,
                prototypes,
                bg_prototypes=None,
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

        if bg_prototypes is not None:
            all_prototypes = torch.cat([prototypes['prototypes'], bg_prototypes['prototypes']]).float()
        else:
            all_prototypes = prototypes['prototypes']

        self.detector = load_model(self.model)
    

    def load_model(self, model):
        
        if model == 'owlvit':
            

        return boxes, box_scores

    def forward(self, images, iou_thr=0.2, conf_thres=0.001, box_conf_threshold=0.01):
        '''
        Args:
            images (torch.Tensor): Input tensor with shape (B, C, H, W)
            iou_thr (float): IoU threshold for NMS
            conf_thres (float): Confidence threshold for NMS
            box_conf_threshold (float): Confidence threshold for box proposals
        '''

        # Generate box proposals
        proposals, box_scores = self.generate_proposals(images)

        # Classify boxes with classifier
        B, num_proposals, _ = proposals.shape
        # TODO: Make it to work with B>1
        preds = self.classifier(prepare_image_for_backbone(images, self.backbone_type), proposals, normalize=True)

        # Extract class scores and predicted classes
        preds = preds.reshape(B, num_proposals, -1)
        scores, _ = torch.max(preds, dim=-1)
        classes = torch.argmax(preds, dim=-1)

        # Filter predictions and prepare for NMS
        processed_predictions = []
        for b in range(B):
            # Filter and prepare boxes for NMS
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
