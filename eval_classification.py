import os
import torch
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
from argparse import ArgumentParser
from sklearn.metrics import classification_report, confusion_matrix
from utils_dir.backbones_utils import prepare_image_for_backbone
from models.detector import OVDBoxClassifier, OVDMaskClassifier
from utils_dir.processing_utils import map_labels_to_prototypes
from utils_dir.nms import custom_xywh2xyxy
from datasets import init_dataloaders
from utils_dir.visualization_utils import plot_classification_confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score


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
    modelClass = OVDBoxClassifier if args.annotations == 'box' else OVDMaskClassifier
    model = modelClass(prototypes['prototypes'], prototypes['label_names'], backbone_type=args.backbone_type, target_size=args.target_size, scale_factor=args.scale_factor).to(device)
    model.train()
    
    return model, device

def eval_classification(args, model, val_dataloader, device):

    true_labels = []
    total_predicted_labels = []
    use_masks = False if args.annotations == 'box' else True

    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader), leave=False):
            images, boxes, labels, _ = batch

            # Location signal is either the boxes or the masks
            if not use_masks:
                loc = custom_xywh2xyxy(boxes).to(device)
            else:
                loc = boxes.float().to(device)
            
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

        # Compute the confusion matrix
        conf_matrix = confusion_matrix(true_labels, total_predicted_labels)

        return conf_matrix, total_predicted_labels, true_labels
            
def main(args):
    print('Setting up training...')

    # Initialize dataloader
    _, val_dataloader = init_dataloaders(args)

    # Load model
    model, device = prepare_model(args)

    # Perform training
    conf_matrix, pred_labels, true_labels = eval_classification(
        args, 
        model, 
        val_dataloader, 
        device
    )

    # Initialize an empty list to store the results
    results_table = []
    num_cls = val_dataloader.dataset.get_category_number()

    class_precision = precision_score(true_labels, pred_labels, average=None)
    class_recall = recall_score(true_labels, pred_labels, average=None)
    class_f1_score = 2 * (class_precision * class_recall) / (class_precision + class_recall + 1e-6)
    class_accuracy = []
    categories = model.get_categories()
    # Print precision, recall, and accuracy for each class

    for cls in range(num_cls):
        
        # Calculate accuracy for the current class
        accuracy = np.sum((true_labels == cls) & (pred_labels == cls)) / np.sum(true_labels == cls)
        class_accuracy.append(accuracy)

        # Append the results to the table
        results_table.append([categories[cls], class_precision[cls], class_recall[cls], class_f1_score[cls], accuracy])

    # Print the results in tabular format
    result_str = tabulate(results_table, headers=["Class", "Precision", "Recall", "F1-score", "Accuracy"], tablefmt="grid")
    print(result_str)

    # Print the mean results across all classes
    mean_precision = np.mean(class_precision)
    mean_recall = np.mean(class_recall)
    mean_f1_score = np.mean(class_f1_score)
    mean_accuracy = np.mean(class_accuracy)

    # Print the mean results in tabular format
    mean_result_str = tabulate([["Mean", mean_precision, mean_recall, mean_f1_score, mean_accuracy]], headers=["Class", "Precision", "Recall", "F1-score", "Accuracy"], tablefmt="grid")
    print(mean_result_str)


    # Write the results to a text file
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        filepath = os.path.join(args.save_dir, "classification_results.txt")
        with open(filepath, "w") as file:
            file.write("Individual Class Results:\n")
            file.write(result_str)
            file.write("\n\n")
            file.write("Mean Results:\n")
            file.write(mean_result_str)

        # Save the confusion matrix
        names = [categories[key] for key in sorted(categories.keys())]
        plot_classification_confusion_matrix(conf_matrix, num_cls, save_dir=args.save_dir, names=names)

        print('Results written to "classification_results.txt"')
    print('Done!')
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--val_root_dir', type=str)
    parser.add_argument('--val_annotations_file', type=str)
    parser.add_argument('--prototypes_path', type=str, default=None)
    parser.add_argument('--aggregation', type=str, default='mean')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--annotations', type=str, default='box')
    parser.add_argument('--backbone_type', type=str, default='dinov2')
    parser.add_argument('--target_size', nargs=2, type=int, metavar=('width', 'height'), default=(560, 560))
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--scale_factor', nargs='+', type=int, default=2)
    args = parser.parse_args()
    main(args)