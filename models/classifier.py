import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T

class OVDClassifier(torch.nn.Module):

    def __init__(self, prototypes, dino_version='dinov2_vitl14', target_size=(560,560), scale_factor=2, min_box_size=5, ignore_index=-1):
        super().__init__()
        self.scale_factor = scale_factor
        self.target_size = target_size
        self.min_box_size = min_box_size
        self.ignore_index = ignore_index
        
        if isinstance(self.scale_factor, int):
            self.scale_factor = [self.scale_factor]

        # Initialize DINOv2 frozen backbone
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        for name, parameter in self.dino.named_parameters():
            parameter.requires_grad = False
        
        # Initialize embedding as a learnable parameter
        self.embedding = torch.nn.Parameter(prototypes)
        self.num_classes = self.embedding.shape[0]
    
    
    def extract_dinov2_features(self, images, scale_factor=1):
        images = F.interpolate(images, scale_factor=scale_factor, mode='bicubic')
        
        with torch.no_grad():
            feats = self.dino.forward_features(images[:2])['x_prenorm'][:, 1:]
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

    def extract_image_cosim(self, images, normalize=False):

        with torch.no_grad():
            # Get images DINOv2 features
            feats = self.extract_dinov2_features(images, self.scale_factor)
    
            # Compute cosine similarity with all classes in the embedding
            cosine_sim = self.get_cosim(feats, self.embedding, normalize=normalize)
            
        return cosine_sim
    

    def forward(self, images, boxes, cls=None, normalize=False, return_cosim=False):
        
        scales = []
        for scale in self.scale_factor:
        
            # Get images DINOv2 features
            feats = self.extract_dinov2_features(images, scale)

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
                x_min, y_min, w, h = box.int()
                y_max = y_min + h
                x_max = x_min + w
                # TODO: either choose on format or adapt to both!!!!!!!!!!!!!
                #x_min, y_min, x_max, y_max = box.int()
                
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

        # Flatten box_logits and target_labels for CrossEntropyLoss
        return box_similarities.view(-1, self.num_classes)
    
