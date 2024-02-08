import torch

def filter_boxes(boxes, classes, scores, target_size, num_labels, box_conf_threshold=0):
    target_height, target_width = target_size
    keep = ((boxes[:, 0] >= 0) & (boxes[:, 1] >= 0) &
            (boxes[:, 2] <= target_width) & (boxes[:, 3] <= target_height))

    filtered_boxes = boxes[keep]
    filtered_classes = classes[keep]
    filtered_scores = scores[keep]

    # Filter out boxes classified as background
    predictions = torch.argmax(filtered_classes, dim=-1)    
    filtered_boxes = filtered_boxes[predictions < num_labels, ...]
    filtered_classes = filtered_classes[predictions < num_labels, ...]
    filtered_scores = filtered_scores[predictions < num_labels]
    
    # Filter out boxes with low confidence
    keep = filtered_scores > box_conf_threshold
    filtered_boxes = filtered_boxes[keep, ...]
    filtered_classes = filtered_classes[keep, ...]
    filtered_scores = filtered_scores[keep]


    return filtered_boxes, filtered_classes[:, :num_labels], filtered_scores



def map_labels_to_prototypes(dataset_categories, model_prototypes, labels):
    mapped_labels = []
    # Create a reverse mapping from class names to indices for the dataset categories
    dataset_categories_reverse = {v: k for k, v in model_prototypes.items()}
    # Map dataset category indices to model prototype indices
    for batch_labels in labels:
        mapped_batch_labels = []
        for label in batch_labels:
            if label == -1:
                mapped_batch_labels.append(-1)
            elif label.item() in dataset_categories and dataset_categories[label.item()] in dataset_categories_reverse:
                class_name = dataset_categories[label.item()]
                if class_name in dataset_categories_reverse:
                    mapped_batch_labels.append(dataset_categories_reverse[class_name])
                else:
                    mapped_batch_labels.append(-1)
            else:
                mapped_batch_labels.append(-1)
        mapped_labels.append(mapped_batch_labels)
    
    return torch.tensor(mapped_labels)