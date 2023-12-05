import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
from transformers import CLIPModel

class OVDClassifier(torch.nn.Module):

    def __init__(self, prototypes, backbone_type='dinov2', target_size=(560,560), scale_factor=2, min_box_size=5, ignore_index=-1):
        super().__init__()
        self.scale_factor = scale_factor
        self.target_size = target_size
        self.min_box_size = min_box_size
        self.ignore_index = ignore_index
        self.backbone_type = backbone_type
        
        if isinstance(self.scale_factor, int):
            self.scale_factor = [self.scale_factor]

        self.backbone = self.initialize_backbone(backbone_type)  # Initialize backbone
        
        # Initialize embedding as a learnable parameter
        self.embedding = torch.nn.Parameter(prototypes)
        self.num_classes = self.embedding.shape[0]
    
    def initialize_backbone(self, backbone_type):
        if backbone_type == 'dinov2':
            backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        elif backbone_type == 'clip-32':
            backbone = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").vision_model
        elif backbone_type == 'clip-14':
            backbone = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").vision_model
        else:
            raise ValueError(f'Backbone {backbone} not supported')
        
        for name, parameter in backbone.named_parameters():
            parameter.requires_grad = False
        
        return backbone
    
    
    def extract_clip_features(self, images, model, tile_size=224, patch_size=14):
        # Extract size and number of tiles
        B, _, image_size, _ = images.shape
        D = 1024
        num_tiles = (image_size // tile_size)**2 if image_size % tile_size == 0 else (image_size // tile_size + 1)**2
        num_tiles_side = int(num_tiles**0.5)

        # Create full image features tensor and a counter for aggregation
        output_features = torch.zeros((B, image_size // patch_size, image_size // patch_size, D)).to(images.device)
        count_tensor = torch.zeros((B, image_size // patch_size, image_size // patch_size,)).to(images.device)

        # Process tiles through CLIP
        with torch.no_grad():
            for i in range(num_tiles_side):
                for j in range(num_tiles_side):

                    # Update tile coords
                    start_i = i * tile_size
                    start_j = j * tile_size
                    end_i = min(start_i + tile_size, image_size)
                    end_j = min(start_j + tile_size, image_size)

                    # If tile exceeds, make new tile containing more image content
                    if end_i - start_i < tile_size:
                        start_i = end_i - tile_size
                    if end_j - start_j < tile_size:
                        start_j = end_j - tile_size
        
                    # Extract the tile from the original image
                    tile = images[:, :, start_i:end_i, start_j:end_j]
        
                    # Extract CLIP's features before token pooling
                    image_features = model(tile).last_hidden_state[:, 1:]
                    _, K, D = image_features.shape
                    p_w = p_h = int(K**0.5)
                    image_features = image_features.reshape(B, p_h, p_w, -1)  # Reshape to 2D

                    # Add features to their location in the original image and track counts per location
                    output_features[:, start_i // patch_size:end_i // patch_size, start_j // patch_size:end_j // patch_size] += image_features
                    count_tensor[:, start_i // patch_size:end_i // patch_size, start_j // patch_size:end_j // patch_size] += 1
        
        # Average the overlapping patches
        output_features /= count_tensor.unsqueeze(-1)
    
        return output_features, count_tensor
    
    def extract_features(self, images, scale_factor=1):
        images = F.interpolate(images, scale_factor=scale_factor, mode='bicubic')

        if self.backbone_type == 'dinov2':
            with torch.no_grad():
                feats = self.backbone.forward_features(images[:2])['x_prenorm'][:, 1:]
        elif self.backbone_type.startswith('clip'):
            patch_size = 14 if self.backbone_type == 'clip-14' else 32
            feats, _ = self.extract_clip_features(images, self.backbone, tile_size=224, patch_size=patch_size)
            feats = feats.view(feats.shape[0], -1, feats.shape[-1])

        return feats

    def get_cosim(self, feats, embedding, normalize=False):
        # Reshape and broadcast for cosine similarity
        B, K, D = feats.shape
        features_reshaped = feats.view(B, 1, K, D)
        embedding_reshaped = embedding.view(1, self.num_classes, 1, D)
    
        # Compute dot product (cosine similarity without normalization)
        dot_product = (features_reshaped * embedding_reshaped).sum(dim=3)

        if normalize:
            # Compute norms
            feats_norm = torch.norm(features_reshaped, dim=3, keepdim=True).squeeze(-1)
            embedding_norm = torch.norm(embedding_reshaped, dim=3, keepdim=True).squeeze(-1)
            # Normalize
            dot_product /= (feats_norm * embedding_norm + 1e-8)  # Add epsilon for numerical stability
    
        # Reshape to 2D and return class similarity maps
        patch_2d_size = int(np.sqrt(feats.shape[1]))
        return dot_product.reshape(-1, self.num_classes, patch_2d_size, patch_2d_size)

    def get_cosim_mini_batch(self, feats, embeddings, batch_size=100, normalize=False):
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
    

    def forward(self, images, boxes, cls=None, normalize=False, return_cosim=False):
        
        scales = []
        for scale in self.scale_factor:
        
            # Get images DINOv2 features
            feats = self.extract_features(images, scale)

            # Compute cosine similarity with all classes in the embedding
            cosine_sim = self.get_cosim_mini_batch(feats, self.embedding, normalize=normalize)
            scales.append(cosine_sim)
        
        cosine_sim = torch.stack(scales).mean(dim=0)

         # Gather similarity values inside each box and compute average box similarity
        box_similarities = []
        for b in range(images.shape[0]):
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
                
                box_sim = cosine_sim[b, :, y_min:y_max -1, x_min:x_max -1].mean(dim=[1, 2])
                has_nan = torch.isnan(box_sim).any().item()
                image_similarities.append(box_sim)
    
            box_similarities.append(torch.stack(image_similarities))
    
        box_similarities = torch.stack(box_similarities)  # Shape: [B, max_boxes, N]
        
        if return_cosim:
            return box_similarities.view(-1, self.num_classes), cosine_sim

        # Flatten box_logits and target_labels
        return box_similarities.view(-1, self.num_classes)
    
