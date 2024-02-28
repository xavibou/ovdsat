import os
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from models.detector import OVDBoxClassifier, OVDMaskClassifier
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm  # Import tqdm
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from argparse import ArgumentParser
from utils_dir.backbones_utils import prepare_image_for_backbone
from utils_dir.processing_utils import map_labels_to_prototypes
from sklearn.metrics import classification_report
from datasets import init_dataloaders

def prepare_model(args):
    # TODO: move to utils or to models __init__.py
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
    modelClass = OVDBoxClassifier if args.annotations == 'box' else OVDMaskClassifier
    model = modelClass(all_prototypes, prototypes['label_names'], backbone_type=args.backbone_type, target_size=args.target_size, scale_factor=args.scale_factor).to(device)
    model.train()
    
    return model, device

def custom_xywh2xyxy(x):
    # TODO: put it in utils as we use it in eval too!!!
    # Convert nx4 boxes from [xmin, ymin, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 2] = x[..., 0] + x[..., 2]  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3]  # bottom right y
    return y

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

def train(args, model, dataloader, val_dataloader, device):

    # Define the optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    torch.autograd.set_detect_anomaly(True)
    scheduler = MultiStepLR(optimizer, milestones=[10, 100], gamma=args.lr_decay)

    # Define the loss function (already defined in your model)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    num_cls = dataloader.dataset.get_category_number()

    use_masks = False if args.annotations == 'box' else True

    # Training loop
    for epoch in range(args.num_epochs):
        total_loss = 0.0
        val_loss = 0.0
        total_correct = 0
        total_samples = 0

        # Use tqdm to create a progress bar for the dataloader
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch + 1}/{args.num_epochs}', leave=False):
            images, boxes, labels, _ = batch


            # Convert map dataset labels classes to the model prototype indices
            labels = map_labels_to_prototypes(dataloader.dataset.get_categories(), model.get_categories(), labels)

            if len(labels[labels != -1]) == 0:
                continue   # skip if no boxes in the image

            if not use_masks:
                if args.only_train_prototypes == False:
                    # Create negative boxes and add them to the batch
                    neg_boxes = generate_additional_boxes(images, boxes, args.iou_threshold, args.min_neg_size, args.max_neg_size, args.num_neg)
                    neg_labels = torch.full((neg_boxes.shape[0], neg_boxes.shape[1]), -2, dtype=torch.long)
                    boxes = torch.cat([boxes, neg_boxes], dim=1)
                    labels = torch.cat([labels, neg_labels], dim=1)

                boxes = custom_xywh2xyxy(boxes)
                images = images.float().to(device)
                boxes = boxes.to(device)
                labels = labels.to(device)

                # Forward pass
                logits = model(prepare_image_for_backbone(images, args.backbone_type), boxes, labels, aggregation=args.aggregation)

                if args.only_train_prototypes == False:
                    # Assign bg_labels to background examples
                    bg_logits = torch.argmax(logits[:, :, num_cls:], dim=-1) + num_cls
                    labels[labels == -2] = bg_logits[labels == -2]
            
            else:
                images = images.float().to(device)
                masks = boxes.float().to(device)
                labels = labels.to(device)
                # Forward pass
                logits = model(prepare_image_for_backbone(images, args.backbone_type), masks, labels, aggregation=args.aggregation)
            
            # Compute loss
            B, N, C = logits.shape
            loss = criterion(logits.view(-1, C), labels.view(-1))

            # If no valid box, continue
            has_nan = torch.isnan(loss).any().item()
            if has_nan:
                continue

            # Backward and optimizer steps
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validation loop
        with torch.no_grad():
            total_correct = 0
            total_samples = 0
            true_labels = []
            total_predicted_labels = []
            for i, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc=f'Val Epoch {epoch + 1}/{args.num_epochs}', leave=False):
                images, boxes, labels, _ = batch

                # Convert map dataset labels classes to the model prototype indices
                labels = map_labels_to_prototypes(val_dataloader.dataset.get_categories(), model.get_categories(), labels)

                if not use_masks:
                    if args.only_train_prototypes == False:
                        # Create negative boxes and add them to the batch
                        neg_boxes = generate_additional_boxes(images, boxes, args.iou_threshold, args.min_neg_size, args.max_neg_size, args.num_neg)
                        neg_labels = torch.full((neg_boxes.shape[0], neg_boxes.shape[1]), -2, dtype=torch.long)
                        boxes = torch.cat([boxes, neg_boxes], dim=1)
                        labels = torch.cat([labels, neg_labels], dim=1)

                    boxes = custom_xywh2xyxy(boxes)
                    images = images.float().to(device)
                    boxes = boxes.to(device)
                    labels = labels.to(device)

                    # Forward pass
                    logits = model(prepare_image_for_backbone(images, args.backbone_type), boxes, labels, aggregation=args.aggregation)

                    if args.only_train_prototypes == False:
                        # Assign bg_labels to background examples
                        bg_logits = torch.argmax(logits[:, :, num_cls:], dim=-1) + num_cls
                        labels[labels == -2] = bg_logits[labels == -2]
                
                else:
                    images = images.float().to(device)
                    masks = boxes.float().to(device)
                    labels = labels.to(device)
                    # Forward pass
                    logits = model(prepare_image_for_backbone(images, args.backbone_type), masks, aggregation=args.aggregation)
                
                # Compute loss
                B, N, C = logits.shape
                loss = criterion(logits.view(-1, C), labels.view(-1))

                # if loss contains nans continue
                has_nan = torch.isnan(loss).any().item()
                if has_nan:
                    continue

                # Calculate predicted labelsnex
                predicted_labels = torch.argmax(logits, dim=-1).view(-1)[labels.view(-1)>=0]

                # Count correct predictions
                total_correct += torch.sum(predicted_labels == labels[labels != -1]).item()
                total_samples += labels[labels != -1].numel()
                total_predicted_labels += predicted_labels.cpu().tolist()
                true_labels += labels[labels != -1].cpu().tolist()

                val_loss += loss.item()
        accuracy = total_correct / total_samples

        # Update the learning rate
        scheduler.step()
        print(f"Epoch [{epoch + 1}/{args.num_epochs}] Train Loss: {total_loss / len(dataloader)}  |  Val Loss: {val_loss / len(val_dataloader)} \nVal Accuracy: {accuracy} --> ({total_correct}/{total_samples})")

        # Print precision, recall, and accuracy for each class every 10 epochs
        if (epoch + 1) % 10 == 0:
            # Convert the predicted labels and true labels to numpy arrays
            predicted_labels_np = np.array(total_predicted_labels)
            true_labels_np = np.array(true_labels)

            # Get the classification report
            report = classification_report(true_labels_np, predicted_labels_np, output_dict=True, zero_division=1)

            # Print precision, recall, and accuracy for each class
            for cls in range(num_cls):
                cls_report = report.get(str(cls), {})  # Use .get() to handle KeyError
                precision = cls_report.get('precision', -1)  # Default to -1.0 if precision is not available
                recall = cls_report.get('recall', -1)  # Default to -1.0 if recall is not available
                f1_score = cls_report.get('f1-score', -1)  # Default to -1.0 if F1-score is not available
                support = cls_report.get('support', 0)  # Default to 0 if support is not available
                
                # Calculate accuracy for the current class
                correct_indices = (true_labels_np == cls) & (predicted_labels_np == cls)
                accuracy = correct_indices.sum() / max(1, support)  # Avoid division by zero

                print(f'{model.get_categories()[cls]}: Pr={precision:.4f}, Re={recall:.4f}, F1={f1_score:.4f}, Acc={accuracy:.4f}')


    return model

def save_results(learned_embedding, class_names, save_dir):
    learned_prototypes = learned_embedding[:len(class_names)]

    prototypes_dict = {
        'prototypes': learned_prototypes,
        'label_names': class_names
    }

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Save learned prototypes and bg_prototypes
    torch.save(prototypes_dict, os.path.join(save_dir, 'prototypes.pth'))
    print('Saved {} prototypes to {}'.format(learned_prototypes.shape[0], os.path.join(save_dir, 'prototypes.pth')))

    if args.bg_prototypes_path is not None:
        learned_bg_prototypes = learned_embedding[len(class_names):]
        bg_class_names = ['bg_class_{}'.format(i+1) for i in range(learned_bg_prototypes.shape[0])]
        bg_prototypes_dict = {
            'prototypes': learned_bg_prototypes,
            'label_names': bg_class_names
        }
        torch.save(bg_prototypes_dict, os.path.join(save_dir, 'bg_prototypes.pth'))

        print('Saved {} bg prototypes to {}'.format(learned_bg_prototypes.shape[0], os.path.join(save_dir, 'bg_prototypes.pth')))

            
def main(args):
    print('Setting up training...')

    # Initialize dataloaders
    train_dataloader, val_dataloader = init_dataloaders(args)
    
    # Load model
    model, device = prepare_model(args)

    # Perform training
    model = train(args, model, train_dataloader, val_dataloader, device)

    # Save model
    if args.save_dir is not None:
        print("Training finished. Saving model...")
        save_results(model.embedding.detach().cpu(), model.class_names, args.save_dir)

    print('Done!')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train_root_dir', type=str)
    parser.add_argument('--val_root_dir', type=str)
    parser.add_argument('--train_annotations_file', type=str)
    parser.add_argument('--val_annotations_file', type=str)
    parser.add_argument('--prototypes_path', type=str, default=None)
    parser.add_argument('--bg_prototypes_path', type=str, default=None)
    parser.add_argument('--aggregation', type=str, default='mean')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--annotations', type=str, default='box')
    parser.add_argument('--backbone_type', type=str, default='dinov2')
    parser.add_argument('--target_size', nargs=2, type=int, metavar=('width', 'height'), default=(560, 560))
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--iou_thr', type=float, default=0.1)
    parser.add_argument('--conf_thres', type=float, default=0.2)
    parser.add_argument('--scale_factor', nargs='+', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_step_size', type=int, default=30)
    parser.add_argument('--lr_decay', type=int, default=0.1)
    parser.add_argument('--num_neg', type=int, default=0)
    parser.add_argument('--min_neg_size', type=int, default=5)
    parser.add_argument('--max_neg_size', type=int, default=150)
    parser.add_argument('--iou_threshold', type=float, default=0.05)
    parser.add_argument('--only_train_prototypes', action='store_true', default=False)
    args = parser.parse_args()
    main(args)