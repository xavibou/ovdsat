


'''
1 - Provide directory with object examples and their segmentation masks
2 - Create model (CLIP / DINO) to extract features
3 - For each class, extract the name and features of the image masked by the segmentation mask
4 - Average the features over the patches that are not masked, which cointain the object
5 - Repeat until done for each object in each class separately
6 - For each class, stack, normalize and average the features of all objects in the class
7 - Save the features as a tensor and the class names as a list in a dictionary and save the prototype
'''

import os
import cv2
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from glob import glob
import os.path as osp
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torchvision import transforms
from transformers import CLIPModel
from torchvision import transforms
from argparse import ArgumentParser
from utils_dir.backbones_utils import load_backbone, extract_backbone_features, get_backbone_params

def preprocess(image, mask=None, backbone_type='dinov2', target_size=(602, 602), patch_size=14):
    '''
    Preprocess an image and its mask to fed the image to the backbone and mask the extracted patches.

    Args:
        image (PIL.Image): Input image
        mask (PIL.Image): Input mask
        backbone_type (str): Backbone type
        target_size (tuple): Target size of the image
        patch_size (int): Patch size of the backbone
    '''

    if 'clip' in backbone_type:
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    # Transform images to tensors and normalize
    transform = transforms.Compose([
        transforms.Resize(target_size),  # Resize the images to a size larger than the window size
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)  # Normalize the images
    ])

    if mask is not None:
        m_w, m_h = target_size
        mask = transforms.Resize((m_w//patch_size, m_h//patch_size), interpolation=Image.NEAREST)(mask)

    image = transform(image).unsqueeze(0)

    return image, mask

# Step 1: Randomly generate boxes that do not intersect with annotations
def generate_mask(image, annotations, num_b, min_size=30, max_size=200, max_iter = 100, patch_size=14):
    '''
    Randomly generate num_b masks that do not intersect with any annotation to generate negative samples.

    Args:
        image (PIL.Image): Input image
        annotations (list): List of annotations
        num_b (int): Number of background samples to extract per image
        min_size (int): Minimum size of the boxes
        max_size (int): Maximum size of the boxes
        max_iter (int): Maximum number of iterations to generate a valid box
        patch_size (int): Patch size of the backbone
    '''
    _, _, h, w = image.shape
    mask = np.zeros((1, h, w), dtype=np.uint8)

    for _ in range(num_b):
        valid_box = False
        count = 0
        while not valid_box:
            count += 1
            if count > max_iter:
                break
            # Generate random coordinates for the top-left corner (x, y), width, and height
            x = random.randint(0, w - max_size)
            y = random.randint(0, h - max_size)
            width = random.randint(min_size, max_size)
            height = random.randint(min_size, max_size)

            # Calculate the coordinates of the bottom-right corner
            x2 = x + width
            y2 = y + height

            # Check if the box intersects with any annotation
            intersects = any(
                x < (ann['bbox'][0] + ann['bbox'][2]) and (ann['bbox'][0] < x2) and
                y < (ann['bbox'][1] + ann['bbox'][3]) and (ann['bbox'][1] < y2)
                for ann in annotations
            )

            if not intersects:
                mask[:, y:y2, x:x2] = 1
                valid_box = True
    
    mask = torch.as_tensor(mask)
    mask = F.interpolate(mask.unsqueeze(0), size=(w//patch_size, h//patch_size), mode='nearest').squeeze(0)
    mask = mask.reshape(-1, (w//patch_size)*(h//patch_size))

    return mask

def cluster_features(data, K=10):
    '''
    Cluster high-dimensional patch features using K-Means and return the cluster averages.

    Args:
        data (numpy.ndarray): Input data with shape (D, N)
        K (int): Number of clusters
    '''
    
    # Create and fit the K-Means model
    kmeans = KMeans(n_clusters=K, random_state=0)
    kmeans.fit(data.T)  # Transpose the data for sklearn's format
    
    # Get cluster labels for each data point
    cluster_labels = kmeans.labels_
    
    # Initialize an array to store the cluster averages
    cluster_averages = np.zeros((K, data.shape[0]))
    
    # Calculate the average vector for each cluster
    for cluster_id in range(K):
        cluster_data = data[:, cluster_labels == cluster_id]
        cluster_average = np.mean(cluster_data, axis=1)
        cluster_averages[cluster_id, :] = cluster_average
    return cluster_averages

def build_background_prototypes(args, model, device, patch_size):
    '''
    Build background prototypes by extracting negative samples and clustering their features into K clusters.

    Args:
        args (argparse.Namespace): Input arguments. Parameters for the clustering are stored here (K, num_b, etc.)
        model (torch.nn.Module): Backbone model
        device (str): Device to run the model on
        patch_size (int): Patch size of the backbone
    '''

    with open(args.annotations_file, "r") as f:
        dataset = json.load(f)

    background_features = []

    print('Extracting background image samples...')
    for i, image_info in tqdm(enumerate(dataset['images'])):
        
        # Read image and annotations
        filename = image_info['file_name']
        annotations = [ann for ann in dataset['annotations'] if ann['image_id'] == image_info['id']]
        image = Image.open(os.path.join(args.src_data_dir, filename))
        image, _ = preprocess(image, backbone_type=args.backbone_type, target_size=args.target_size, patch_size=patch_size)

        # Generate mask
        mask = generate_mask(image, annotations, args.num_b, patch_size=patch_size).to(device)
            
        # Extract features and reshape to 2D image
        features = extract_backbone_features(image.to(device), model, args.backbone_type, scale_factor=args.scale_factor)
        features = features.squeeze().permute(1,0)

        # Mask tokens and keep only the ones that are not masked
        masked_patch_tokens = features * mask
        filtered_patch_tokens = masked_patch_tokens[:, mask.squeeze(0) == 1]
        background_features.append(filtered_patch_tokens.cpu())
    
    background_features = torch.cat(background_features, dim=-1).numpy()
    d, n = background_features.shape
    print('\nDone. Extracted background {} vectors of dimensionality = {}'.format(n, d)) 

    print('Clustering features into {} clusters...'.format(args.k))
    kmeans_features = cluster_features(background_features, args.k)
    
    # Normalize the tensor along dim=1 using F.normalize
    prototypes = F.normalize(torch.from_numpy(kmeans_features), dim=1)      # Normalize the feature vectors
    classes = ['bg_class_{}'.format(i+1) for i in range(prototypes.shape[0])]

    category_dict = {
        'prototypes': prototypes,
        'label_names': classes
    }

    return category_dict

def build_object_prototypes(args, model, device, patch_size):
    '''
    Build object prototypes by extracting the features containing the objects and averaging them for each class, separately.

    Args:
        args (argparse.Namespace): Input arguments
        model (torch.nn.Module): Backbone model
        device (str): Device to run the model on
        patch_size (int): Patch size of the backbone
    '''

    # Retrieve masked images and their classes
    class2images = {}
    classes = []
    masked_imgs = []
    for f in glob(osp.join(args.data_dir, '**/*'), recursive=True):
        if osp.isfile(f) and 'mask' not in f:
            image_file = f
            class_name = osp.basename(osp.dirname(f))
            
            classes = classes + [class_name] if class_name not in classes else classes
            mask_file = osp.splitext(f)[0] + '.mask.jpg'
            if class_name not in class2images:
                class2images[class_name] = []
            class2images[class_name.strip().lower()].append((image_file, mask_file)) 
            
            masked_imgs.append(Image.fromarray(cv2.imread(image_file) * (cv2.imread(mask_file) != 0)))
    
    class2tokens = {}
    for cls, images in tqdm(class2images.items()):
        class2tokens[cls] = []
        for image_file, mask_file in images:
            # Read image and mask
            image = Image.open(image_file)
            mask = Image.open(mask_file)
            image, mask = preprocess(image, mask, backbone_type=args.backbone_type, target_size=args.target_size, patch_size=patch_size)
            mask = torch.as_tensor(np.array(mask) / 255).to(device)
            
            # Extract features and reshape to 2D image
            features = extract_backbone_features(image.to(device), model, args.backbone_type, scale_factor=args.scale_factor)
            _, K, D = features.shape
            p_w = p_h = int(K**0.5)
            features = features.reshape(p_h, p_w, -1).permute(2, 0, 1)

            # If the mask is empty, skip the image
            if mask.sum() <= 0.5:
                continue

            # Average the features of the masked patches
            avg_patch_token = (mask.unsqueeze(0) * features).flatten(1).sum(1) / mask.sum()
            class2tokens[cls].append(avg_patch_token)
    
    # Average the features of all objects in each class
    for cls in class2tokens:
        class2tokens[cls] = torch.stack(class2tokens[cls]).mean(dim=0)
    prototypes = F.normalize(torch.stack([class2tokens[c] for c in classes]), dim=1)    # Normalize the feature vectors

    # Create dictionary and save
    category_dict = {
        'prototypes': prototypes.cpu(),
        'label_names': classes
    }
    return category_dict

def main(args):
    '''
    Main function to build object and background prototypes.

    Args:
        args (argparse.Namespace): Input arguments
    '''

    print('Building prototypes...')
    print(f'Loading model: {args.backbone_type}...')

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_backbone(args.backbone_type)
    model = model.to(device)
    model.eval()
    patch_size, _ = get_backbone_params(args.backbone_type)

    # Build object prototypes
    obj_category_dict = build_object_prototypes(args, model, device, patch_size)
    
    # Create save directory if it does not exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Build background prototypes if specified
    if args.store_bg_prototypes:

        bg_category_dict = build_background_prototypes(args, model, device, patch_size)
        save_name = f'bg_prototypes_{args.backbone_type}.pt'
        torch.save(bg_category_dict, os.path.join(args.save_dir, save_name))

    save_name = f'prototypes_{args.backbone_type}.pt'
    torch.save(obj_category_dict, os.path.join(args.save_dir, save_name))

    print(f'Saved prototypes to {os.path.join(args.save_dir, save_name)}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/mnt/ddisk/boux/code/devit/datasets/simd_subset_10')
    parser.add_argument('--save_dir', type=str, default='/mnt/ddisk/boux/code/ovdsat/run/classification_benchmark_exp')
    parser.add_argument('--annotations_file', type=str, default='/mnt/ddisk/boux/code/data/simd/train_coco_subset_N10.json')
    parser.add_argument('--src_data_dir', type=str, default='/mnt/ddisk/boux/code/data/simd/training')
    parser.add_argument('--backbone_type', type=str, default='dinov2')
    parser.add_argument('--target_size', nargs=2, type=int, metavar=('width', 'height'), default=(602, 602))
    parser.add_argument('--window_size', type=int, default=224)
    parser.add_argument('--scale_factor', type=int, default=1)
    parser.add_argument('--num_b', type=int, default=10, help='Number of background samples to extract per image')
    parser.add_argument('--k', type=int, default=200, help='Number of background prototypes (clusters for k-means)')
    parser.add_argument('--store_bg_prototypes', action='store_true', default=False)
    args = parser.parse_args()

    main(args)