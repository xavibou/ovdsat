import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
from transformers import CLIPModel
from utils_dir.backbones_utils import extract_backbone_features, load_backbone

class OVDClassifier(torch.nn.Module):

    def __init__(self, prototypes, backbone_type='dinov2', target_size=(602,602), scale_factor=2, min_box_size=5, ignore_index=-1):
        super().__init__()
        self.scale_factor = scale_factor
        self.target_size = target_size
        self.min_box_size = min_box_size
        self.ignore_index = ignore_index
        self.backbone_type = backbone_type
        
        if isinstance(self.scale_factor, int):
            self.scale_factor = [self.scale_factor]

        # Initialize backbone
        self.backbone = load_backbone(backbone_type)  
        
        # Initialize embedding as a learnable parameter
        self.embedding = torch.nn.Parameter(prototypes)
        self.num_classes = self.embedding.shape[0]


    def get_cosim_mini_batch(self, feats, embeddings, batch_size=100, normalize=False):
        '''
        Compute cosine similarity between features and protorype embeddings in mini-batches to avoid memory issues.

        Args:
            feats (torch.Tensor): Features with shape (B, K, D)
            embeddings (torch.Tensor): Embeddings with shape (N, D)
            batch_size (int): mini-batch size for computing the cosine similarity
            normalize (bool): Whether to normalize the cosine similarity maps
        '''

        num_feats = feats.shape[0]
        num_classes = embeddings.shape[0]
        patch_2d_size = int(np.sqrt(feats.shape[1]))

        cosim_list = []
        for start_idx in range(0, num_classes, batch_size):
            end_idx = min(start_idx + batch_size, num_classes)

            embedding_batch = embeddings[start_idx:end_idx]  # Get a batch of embeddings

            # Reshape and broadcast for cosine similarity
            B, K, D = feats.shape
            features_reshaped = feats.view(B, 1, K, D)
            embedding_reshaped = embedding_batch.view(1, end_idx - start_idx, 1, D)

            # Compute dot product (cosine similarity without normalization)
            dot_product = (features_reshaped * embedding_reshaped).sum(dim=3)

            if normalize:
                # Compute norms
                feats_norm = torch.norm(features_reshaped, dim=3, keepdim=True).squeeze(-1)
                embedding_norm = torch.norm(embedding_reshaped, dim=3, keepdim=True).squeeze(-1)
                # Normalize
                dot_product /= (feats_norm * embedding_norm + 1e-8)  # Add epsilon for numerical stability

            # Append the similarity scores for this batch to the list
            cosim_list.append(dot_product)

        # Concatenate the similarity scores from different batches
        cosim = torch.cat(cosim_list, dim=1)

        # Reshape to 2D and return class similarity maps
        cosim = cosim.reshape(-1, num_classes, patch_2d_size, patch_2d_size)

        # Interpolate cosine similarity maps to original resolution
        cosim = F.interpolate(cosim, size=self.target_size, mode='bicubic')

        return cosim
    

    def forward(self, images, boxes, cls=None, normalize=False, return_cosim=False, aggregation='mean', k=10):
        '''
        Args:
            images (torch.Tensor): Input tensor with shape (B, C, H, W)
            boxes (torch.Tensor): Box coordinates with shape (B, max_boxes, 4)
            cls (torch.Tensor): Class labels with shape (B, max_boxes)
            normalize (bool): Whether to normalize the cosine similarity maps
            return_cosim (bool): Whether to return the cosine similarity maps
        '''
        
        scales = []

        for scale in self.scale_factor:
            feats = extract_backbone_features(images, self.backbone, self.backbone_type, scale_factor=scale)
            cosine_sim = self.get_cosim_mini_batch(feats, self.embedding, normalize=normalize)
            scales.append(cosine_sim)

        cosine_sim = torch.stack(scales).mean(dim=0)

         # Gather similarity values inside each box and compute average box similarity
        box_similarities = []
        B = images.shape[0]
        for b in range(B):
            image_boxes = boxes[b][:, :4]
    
            image_similarities = []
            count = 0
            for i, box in enumerate(image_boxes):
                x_min, y_min, x_max, y_max = box.int()
                
                if cls is not None:
                    if (x_min < 0 or
                        y_min < 0 or
                        y_max > self.target_size[0] or
                        x_max > self.target_size[0] or
                        y_max - y_min < self.min_box_size or
                        x_max - x_min < self.min_box_size):
                        count += 1
                        cls[b][i] = self.ignore_index   # If invalid box, assign the label to ignore it while computing the loss
                
                if aggregation == 'mean':
                    box_sim = cosine_sim[b, :, y_min:y_max -1, x_min:x_max -1].mean(dim=[1, 2])
                elif aggregation == 'max':
                    _,n,h,w = cosine_sim.shape
                    box_sim, _ = cosine_sim[b, :, y_min:y_max - 1, x_min:x_max - 1].reshape(n, -1).max(dim=1)
                elif aggregation == 'topk':
                    _,n,h,w = cosine_sim.shape
                    box_sim = cosine_sim[b, :, y_min:y_max - 1, x_min:x_max - 1].reshape(n, -1)
                    topk = k if k <= box_sim.shape[1] else box_sim.shape[1]
                    box_sim, _ = box_sim.topk(topk, dim=1)
                    box_sim = box_sim.mean(dim=1)
                else:
                    raise ValueError('Invalid aggregation method')
                
                has_nan = torch.isnan(box_sim).any().item()
                image_similarities.append(box_sim)
    
            box_similarities.append(torch.stack(image_similarities))
    
        box_similarities = torch.stack(box_similarities)  # Shape: [B, max_boxes, N]
        
        if return_cosim:
            return box_similarities.view(b, -1, self.num_classes), cosine_sim

        # Flatten box_logits and target_labels
        return box_similarities.view(B, -1, self.num_classes)
    
