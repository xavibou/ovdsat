import cv2
import torch
import torchvision.transforms as T
from models.classifier import OVDBoxClassifier, OVDMaskClassifier
from utils_dir.rpn_utils import get_box_RPN
from utils_dir.processing_utils import filter_boxes
from utils_dir.nms import non_max_suppression
from utils_dir.backbones_utils import prepare_image_for_backbone
from models.rpn.obb_rpn import OBBRPN
from models.rpn.box_rpn import BoxRPN

class OVDDetector(torch.nn.Module):

    def __init__(self,
                prototypes,
                bg_prototypes=None,
                backbone_type='dinov2',
                target_size=(602,602),
                scale_factor=2,
                min_box_size=5,
                ignore_index=-1,
                rpn_config='configs/FasterRCNN_FPN_DOTA_config.yaml',
                rpn_checkpoint='weights/FasterRCNN_FPN_DOTA_final_model.pth',
                classification='box'    # Possible values: 'box', 'obb', 'mask'
                ):
        super().__init__()
        self.scale_factor = scale_factor
        self.target_size = target_size
        self.min_box_size = min_box_size
        self.ignore_index = ignore_index
        self.class_names = prototypes['label_names']
        self.num_classes = len(self.class_names)
        self.backbone_type = backbone_type        

        if classification not in ['box', 'mask', 'obb']:
            raise ValueError('Invalid classification type. Must be either "box", "obb" or "mask"')
        self.classification = classification

        if bg_prototypes is not None:
            all_prototypes = torch.cat([prototypes['prototypes'], bg_prototypes['prototypes']]).float()
        else:
            all_prototypes = prototypes['prototypes']

        # Initialize RPN
        if self.classification == 'box':
            self.rpn = BoxRPN(rpn_config, rpn_checkpoint)
        elif self.classification == 'obb':
            self.rpn = OBBRPN(rpn_config, rpn_checkpoint)
        elif self.classification == 'mask':
            raise NotImplementedError('Mask RPN not implemented yet. Should use SAM to generate proposals.')

        # Initialize Classifier
        classifier = OVDBoxClassifier if classification == 'box' else OVDMaskClassifier
        self.classifier = classifier(all_prototypes, prototypes['label_names'], backbone_type, target_size, scale_factor, min_box_size, ignore_index)
    

    def forward(self, images, iou_thr=0.2, conf_thres=0.001, box_conf_threshold=0.01, aggregation='mean', labels=None):
        '''
        Args:
            images (torch.Tensor): Input tensor with shape (B, C, H, W)
            iou_thr (float): IoU threshold for NMS
            conf_thres (float): Confidence threshold for NMS
            box_conf_threshold (float): Confidence threshold for box proposals
        '''

        with torch.no_grad():
            # Generate box proposals
            if self.classification == 'box':
                proposals, proposals_scores = self.rpn(images)
            elif self.classification == 'obb':
                boxes, proposals_scores, proposals = self.rpn(images)
            elif self.classification == 'mask':
                raise NotImplementedError('Mask RPN not implemented yet. Should use SAM to generate proposals.')

            # Classify boxes with classifier
            B, num_proposals = proposals_scores.shape
            preds = self.classifier(prepare_image_for_backbone(images, self.backbone_type), proposals, normalize=True, aggregation=aggregation)

            if num_proposals == 0:
                return [torch.tensor([], device=images.device) for _ in range(B)]

            # Extract class scores and predicted classes
            preds = preds.reshape(B, num_proposals, -1)
            scores, _ = torch.max(preds, dim=-1)
            classes = torch.argmax(preds, dim=-1)

            # Filter predictions and prepare for NMS
            processed_predictions = []
            for b in range(B):
                # Filter and prepare boxes for NMS
                filtered_boxes, filtered_classes, filtered_scores = filter_boxes(proposals[b] if self.classification == 'box' else boxes[b],
                                                                                preds[b],
                                                                                proposals_scores[b],
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

            del preds, scores, classes, proposals, proposals_scores
            if self.classification == 'obb':
                del boxes
            
            return processed_predictions