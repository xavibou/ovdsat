import os
import torch
import random
from torch.utils.data import DataLoader
from datasets.dataset import DINODataset
from models.detector import OVDClassifier
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm  # Import tqdm
from torch.optim.lr_scheduler import StepLR
import torch.optim.lr_scheduler as lr_scheduler
from argparse import ArgumentParser
from models.owlvit import load_OWLViT

from transformers import AutoProcessor
from losses import PushPullLoss

def custom_xywh2xyxy(x):
    # Convert nx4 boxes from [xmin, ymin, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 2] = x[..., 0] + x[..., 2]  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3]  # bottom right y
    return y

def prepare_model(args):
    # Use GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Init model
    queries = torch.load(args.prototypes_path)['label_names']
    model = load_OWLViT(queries, device)
    model.train()
    
    return model, device

def train(args, model, dataloader, val_dataloader, device):

    num_cls = len(dataloader.dataset.embedding_classes)

    # Define the optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.lr_decay,
    )
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[40, 90, 150], gamma=0.1)

    # Define the loss function (already defined in your model)
    scales = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    criterion = PushPullLoss(
        num_cls,
        scales=None,
    )
    num_cls = len(dataloader.dataset.embedding_classes)
    processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")

    # Training loop
    for epoch in range(args.num_epochs):
        total_loss = 0.0
        cls_loss = 0.0
        box_loss = 0.0
        bg_loss = 0.0

        model.train()
        # Use tqdm to create a progress bar for the dataloader
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch + 1}/{args.num_epochs}', leave=False):
            
            images, boxes, labels, metadata = batch
            if len(labels[labels != -1]) == 0:
                continue   # skip if no boxes in the image

            keep = labels > -1
            boxes = boxes[None, keep, :]
            labels = labels[None, keep]

            # Preprocess images
            images = processor(images=images, return_tensors="pt", device=device)["pixel_values"]

            # Create negative boxes and add them to the batch
            images = images.float().to(device)
            boxes = boxes.to(device)
            labels = labels.to(device)

            # Forward pass
            all_pred_boxes, pred_classes, pred_sims, _ = model(images)

            boxes = custom_xywh2xyxy(boxes) / args.target_size[0]
            losses = criterion(pred_sims, labels, all_pred_boxes, boxes.float())

            loss = 0.0  # Initialize the total loss to zero
            loss = (
                losses["loss_ce"]
                + losses["loss_bg"]
                #+ losses["loss_bbox"]
                #+ losses["loss_giou"]
            )

            # Backward and optimizer steps
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            cls_loss += losses["loss_ce"].item()
            box_loss += losses["loss_bbox"].item() + losses["loss_giou"].item()
            bg_loss += losses["loss_bg"].item()
        
        # Update the learning rate
        #scheduler.step()

        model.eval()
        val_total_loss = 0.0
        val_cls_loss = 0.0
        val_box_loss = 0.0
        val_bg_loss = 0.0
        # Use tqdm to create a progress bar for the dataloader
        for i, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc=f'Epoch {epoch + 1}/{args.num_epochs}', leave=False):
            with torch.no_grad():
                images, boxes, labels, metadata = batch
                if len(labels[labels != -1]) == 0:
                    continue   # skip if no boxes in the image

                keep = labels > -1
                boxes = boxes[None, keep, :]
                labels = labels[None, keep]

                # Preprocess images
                images = processor(images=images, return_tensors="pt", device=device)["pixel_values"]

                # Create negative boxes and add them to the batch
                images = images.float().to(device)
                boxes = boxes.to(device)
                labels = labels.to(device)

                # Forward pass
                all_pred_boxes, pred_classes, pred_sims, _ = model(images)

                boxes = custom_xywh2xyxy(boxes) / args.target_size[0]
                losses = criterion(pred_sims, labels, all_pred_boxes, boxes.float())

                loss = 0.0  # Initialize the total loss to zero
                loss = (
                    losses["loss_ce"]
                    + losses["loss_bg"]
                    + losses["loss_bbox"]
                    + losses["loss_giou"]
                )

            val_total_loss += loss.item()
            val_cls_loss += losses["loss_ce"].item()
            val_box_loss += losses["loss_bbox"].item() + losses["loss_giou"].item()
            val_bg_loss += losses["loss_bg"].item()
        
        print(f"Epoch [{epoch + 1}/{args.num_epochs}] Losses: cls:{cls_loss / len(dataloader)} | bg: {bg_loss / len(dataloader)} | box: {box_loss / len(dataloader)}")
        print(f"Val loss: cls:{val_cls_loss / len(val_dataloader)} | bg: {val_bg_loss / len(val_dataloader)} | box: {val_box_loss / len(val_dataloader)}")
        print()
        

    return model

def save_results(learned_embedding, class_names, save_dir):
    learned_prototypes = learned_embedding[:len(class_names)]
    learned_bg_prototypes = learned_embedding[len(class_names):]

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
    torch.save(prototypes_dict, os.path.join(save_dir, 'prototypes.pth'))
    torch.save(bg_prototypes_dict, os.path.join(save_dir, 'bg_prototypes.pth'))

    print('Saved {} prototypes to {}'.format(learned_prototypes.shape[0], os.path.join(save_dir, 'prototypes.pth')))
    print('Saved {} bg prototypes to {}'.format(learned_bg_prototypes.shape[0], os.path.join(save_dir, 'bg_prototypes.pth')))

            
def main(args):
    print('Setting up training...')

    # Initialize dataloader
    embedding_labels = torch.load(args.prototypes_path)['label_names']
    dataset = DINODataset(args.root_dir, args.annotations_file, embedding_labels, augment=True, target_size=args.target_size)
    val_dataset = DINODataset(args.root_dir, args.val_annotations_file, embedding_labels, augment=False, target_size=args.target_size)
    dataloader = test_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = test_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Load model
    model, device = prepare_model(args)

    # Perform training
    model = train(args, model, dataloader, val_dataloader, device)

    # Save model
    if args.save_dir is not None:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        torch.save(model.state_dict(), os.path.join(args.save_dir,'owlvit_model_weights.pth'))

    print('Done!')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str)
    parser.add_argument('--annotations_file', type=str)
    parser.add_argument('--val_annotations_file', type=str)
    parser.add_argument('--prototypes_path', type=str, default=None)
    parser.add_argument('--bg_prototypes_path', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--target_size', nargs=2, type=int, metavar=('width', 'height'), default=(560, 560))
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--iou_thr', type=float, default=0.1)
    parser.add_argument('--conf_thres', type=float, default=0.2)
    parser.add_argument('--scale_factor', nargs='+', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--lr_step_size', type=int, default=30)
    parser.add_argument('--lr_decay', type=int, default=0.1)
    parser.add_argument('--num_neg', type=int, default=20)
    parser.add_argument('--min_neg_size', type=int, default=5)
    parser.add_argument('--max_neg_size', type=int, default=150)
    parser.add_argument('--iou_threshold', type=float, default=0.3)
    args = parser.parse_args()
    main(args)
