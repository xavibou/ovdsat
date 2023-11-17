import torch
import random
from torch.utils.data import DataLoader
from datasets.dataset import DINODataset
from models.detector import OVDClassifier
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm  # Import tqdm
from torch.optim.lr_scheduler import StepLR
from argparse import ArgumentParser

def prepare_model(args):
    # Use GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load prototypes from checkpoint
    prototypes = torch.load(args.prototypes_path)
    bg_prototypes = torch.load(args.bg_prototypes_path) if args.bg_prototypes_path is not None else None
    if args.bg_prototypes_path is not None:
        all_prototypes = torch.cat([prototypes['prototypes'], bg_prototypes['prototypes']]).float()
    else:
        all_prototypes = prototypes['prototypes']

    # Initialize model and move it to device
    model = OVDClassifier(all_prototypes, target_size=args.target_size, scale_factor=args.scale_factor).to(device)
    model.train()
    
    return model, device


def generate_additional_boxes(images_batch, bounding_boxes_batch, iou_threshold, min_size, max_size, num_additional_boxes):
    B, N, _ = bounding_boxes_batch.shape  # B: Batch size, N: Number of boxes per image

    additional_boxes = []
    for b in range(B):
        boxes_per_image = bounding_boxes_batch[b]  # Extract bounding boxes for each image

        for _ in range(num_additional_boxes):
            generated_box = []
            max_attempts = 100  # Max attempts to generate a box with IoU < iou_threshold

            for _ in range(max_attempts):
                # Generate random box dimensions
                width = random.randint(min_size, max_size)
                height = random.randint(min_size, max_size)
                x = random.randint(0, images_batch.size(3) - width)
                y = random.randint(0, images_batch.size(2) - height)

                # Calculate IoU with existing boxes
                iou = torch.tensor([0.0])
                for gt_box in boxes_per_image:
                    box_area = (width * height)
                    intersection_x1 = max(x, gt_box[0])
                    intersection_y1 = max(y, gt_box[1])
                    intersection_x2 = min(x + width, gt_box[0] + gt_box[2])
                    intersection_y2 = min(y + height, gt_box[1] + gt_box[3])

                    intersection_area = max(intersection_x2 - intersection_x1, 0) * max(intersection_y2 - intersection_y1, 0)
                    gt_box_area = (gt_box[2] * gt_box[3]).float()
                    union_area = box_area + gt_box_area - intersection_area

                    iou = max(iou, intersection_area / union_area)

                if iou < iou_threshold:
                    # If generated box has IoU < iou_threshold with all existing boxes, add to the list
                    generated_box = [x, y, width, height]
                    break

            if generated_box:
                additional_boxes.append(generated_box)

    # Reshape additional boxes to [B, M, 4]
    additional_boxes_tensor = torch.tensor(additional_boxes).reshape(B, -1, 4)
    return additional_boxes_tensor

def train(args, model, dataloader, device):

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Define the loss function (already defined in your model)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    num_cls = len(dataloader.dataset.embedding_classes)

    # Training loop
    for epoch in range(args.num_epochs):
        total_loss = 0.0

        # Use tqdm to create a progress bar for the dataloader
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch + 1}/{args.num_epochs}', leave=False):
            images, boxes, labels, _ = batch
            if len(labels[labels != -1]) == 0:
                continue   # skip if no boxes in the image

            # Create negative boxes and add them to the batch
            neg_boxes = generate_additional_boxes(images, boxes, args.iou_threshold, args.min_neg_size, args.max_neg_size, args.num_neg)
            neg_labels = torch.full((neg_boxes.shape[0], neg_boxes.shape[1]), -2, dtype=torch.long)
            boxes = torch.cat([boxes, neg_boxes], dim=1)
            labels = torch.cat([labels, neg_labels], dim=1)

            images = images.float().to(device)
            boxes = boxes.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(images, boxes, labels)

            # Assign bg_labels to background examples
            bg_logits = torch.argmax(logits[:, num_cls:], dim=1) + num_cls
            labels[labels == -2] = bg_logits[None, :][labels == -2]

            # Compute loss
            loss = criterion(logits, labels.view(-1))

            # If no valid box, continue
            has_nan = torch.isnan(loss).any().item()
            if has_nan:
                continue

            # Backward and optimizer steps
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        print(f"Epoch [{epoch + 1}/{args.num_epochs}] Loss: {total_loss / len(dataloader)}")

    return model

def save_results(learned_embedding, class_names, save_dir):
    learned_prototypes = model.embedding[:dataset.num_classes]
    learned_bg_prototypes = model.embedding[dataset.num_classes:]

    prototypes_dict = {
        'prototypes': learned_prototypes,
        'label_names': class_names
    }

    bg_class_names = ['bg_class_{}'.format(i+1) for i in range(learned_bg_prototypes.shape[0])]
    bg_prototypes_dict = {
        'prototypes': learned_bg_prototypes,
        'label_names': bg_class_names
    }

    # Save learned prototypes and bg_prototypes
    torch.save(learned_prototypes, os.path.join(save_dir, 'prototypes.pth'))
    torch.save(learned_bg_prototypes, os.path.join(save_dir, 'bg_prototypes.pth'))

            
def main(args):
    print('Setting up training...')

    # Initialize dataloader
    embedding_labels = torch.load(args.prototypes_path)['label_names']
    dataset = DINODataset(args.root_dir, args.annotations_file, embedding_labels, augment=True, target_size=args.target_size)
    dataloader = test_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Load model
    model, device = prepare_model(args)

    # Perform training
    model = train(args, model, dataloader, device)

    # Save model
    if args.save_dir is not None:
        print("Training finished. Saving model...")
        save_results(model.embedding.detach().cpu(), embedding_labels, args.save_dir)

    print('Done!')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str)
    parser.add_argument('--annotations_file', type=str)
    parser.add_argument('--prototypes_path', type=str, default=None)
    parser.add_argument('--bg_prototypes_path', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--target_size', nargs=2, type=int, metavar=('width', 'height'), default=(560, 560))
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--iou_thr', type=float, default=0.1)
    parser.add_argument('--conf_thres', type=float, default=0.2)
    parser.add_argument('--scale_factor', nargs='+', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--num_neg', type=int, default=20)
    parser.add_argument('--min_neg_size', type=int, default=5)
    parser.add_argument('--max_neg_size', type=int, default=150)
    parser.add_argument('--iou_threshold', type=float, default=0.3)
    args = parser.parse_args()
    main(args)

