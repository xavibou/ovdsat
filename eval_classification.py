import os
import torch
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
from argparse import ArgumentParser
from sklearn.metrics import classification_report
from models.detector import OVDBoxClassifier, OVDMaskClassifier
from utils_dir.backbones_utils import prepare_image_for_backbone
from utils_dir.processing_utils import map_labels_to_prototypes
from utils_dir.nms import custom_xywh2xyxy
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

    # Initialize model and move it to device
    modelClass = OVDMaskClassifier if args.use_segmentation else OVDBoxClassifier
    model = modelClass(prototypes['prototypes'], prototypes['label_names'], backbone_type=args.backbone_type, target_size=args.target_size, scale_factor=args.scale_factor).to(device)
    model.train()
    
    return model, device

def eval_classification(args, model, val_dataloader, device):

    true_labels = []
    total_predicted_labels = []

    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader), leave=False):

            # Location signal is either the boxes or the masks
            if not args.use_segmentation:
                images, boxes, labels, _, metadata = batch
                loc = custom_xywh2xyxy(boxes).to(device)
            else:
                images, _, labels, masks, metadata = batch
                loc = masks.float().to(device)
            
            # Convert map dataset labels classes to the model prototype indices
            labels = map_labels_to_prototypes(val_dataloader.dataset.get_categories(), model.get_categories(), labels)
            images = images.float().to(device)
            labels = labels.to(device)

            # Forward pass
            with torch.no_grad():
                logits = model(prepare_image_for_backbone(images, args.backbone_type), loc, labels, normalize=True, aggregation=args.aggregation)

            # Calculate predicted labelsnex
            predicted_labels = torch.argmax(logits, dim=-1).view(-1)[labels.view(-1)>=0]

            # Track the true and predicted labels
            total_predicted_labels += predicted_labels.cpu().tolist()
            true_labels += labels[labels != -1].cpu().tolist()

        # Convert the predicted labels and true labels to numpy arrays
        total_predicted_labels = np.array(total_predicted_labels)
        true_labels = np.array(true_labels)


        # Get the classification report
        report = classification_report(true_labels, total_predicted_labels, output_dict=True, zero_division=1)

        return report, total_predicted_labels, true_labels
            
def main(args):
    print('Setting up training...')

    # Initialize dataloader
    _, val_dataloader = init_dataloaders(args)

    # Load model
    model, device = prepare_model(args)

    # Perform training
    report, pred_labels, true_labels = eval_classification(
        args, 
        model, 
        val_dataloader, 
        device
    )

    # Initialize an empty list to store the results
    results_table = []
    num_cls = val_dataloader.dataset.get_category_number()

    # Print precision, recall, and accuracy for each class
    for cls in range(num_cls):
        cls_report = report.get(str(cls), {})  # Use .get() to handle KeyError
        precision = cls_report.get('precision', -1)  # Default to -1.0 if precision is not available
        recall = cls_report.get('recall', -1)  # Default to -1.0 if recall is not available
        f1_score = cls_report.get('f1-score', -1)  # Default to -1.0 if F1-score is not available
        support = cls_report.get('support', 0)  # Default to 0 if support is not available
        
        # Calculate accuracy for the current class
        correct_indices = (true_labels == cls) & (pred_labels == cls)
        accuracy = correct_indices.sum() / max(1, support)  # Avoid division by zero
        
        # Append the results to the table
        results_table.append([model.get_categories()[cls], precision, recall, f1_score, accuracy])

    # Print the results in tabular format
    print(tabulate(results_table, headers=["Class", "Precision", "Recall", "F1-score", "Accuracy"], tablefmt="grid"))

    # Print the mean results across all classes
    mean_precision = report.get('macro avg', {}).get('precision', -1)
    mean_recall = report.get('macro avg', {}).get('recall', -1)
    mean_f1_score = report.get('macro avg', {}).get('f1-score', -1)
    mean_accuracy = np.mean(true_labels == pred_labels)

    # Print the mean results in tabular format
    print(tabulate([["Mean", mean_precision, mean_recall, mean_f1_score, mean_accuracy]], headers=["Class", "Precision", "Recall", "F1-score", "Accuracy"], tablefmt="grid"))
    print('Done!')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--val_root_dir', type=str)
    parser.add_argument('--val_annotations_file', type=str)
    parser.add_argument('--prototypes_path', type=str, default=None)
    parser.add_argument('--aggregation', type=str, default='mean')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--backbone_type', type=str, default='dinov2')
    parser.add_argument('--target_size', nargs=2, type=int, metavar=('width', 'height'), default=(560, 560))
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--scale_factor', nargs='+', type=int, default=2)
    parser.add_argument('--use_segmentation', action='store_true', default=False)
    args = parser.parse_args()
    main(args)