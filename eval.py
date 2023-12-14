# Imports
import torch
import numpy as np
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from utils_dir.nms import non_max_suppression
from datasets.dataset import DINODataset
from utils_dir.metrics import ConfusionMatrix, ap_per_class, box_iou
from datasets import get_base_new_classes

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

def plot_image_with_boxes(image_path, detections, label_names, filepath, target_size):
    '''
    Plot image with bounding boxes
    
    Args:
        image_path (str): Path to the image
        detections (array[N, 6]): x1, y1, x2, y2, conf, class
        label_names (list): List of label names
        filepath (str): Path to save the image
        target_size (tuple): Target size of the image
    '''
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)

    # Create figure and axes
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Plot bounding boxes
    for detection in detections:
        xmin, ymin, xmax, ymax, conf, class_id = detection

        # Get label name based on class_id
        label_name = label_names[int(class_id)]

        # Calculate box width and height
        width = xmax - xmin
        height = ymax - ymin

        # Create a Rectangle patch
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

        # Add label name above the bounding box
        ax.text(xmin, ymin - 5, f'{label_name}: {conf:.2f}', color='red', fontsize=8, ha='left', va='bottom')

    # Save the plot to the specified directory
    save_path = f"{filepath}"
    plt.savefig(save_path)
    plt.close()


def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

def custom_xywh2xyxy(x):
    '''
    Convert nx4 boxes from [xmin, ymin, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right

    Args:
        x (torch.Tensor or np.array): Input tensor with shape (N, 4)
    '''
    # TODO: put it in utils as we use it in train too!!!
    # Convert nx4 boxes from [xmin, ymin, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 2] = x[..., 0] + x[..., 2]  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3]  # bottom right y
    return y


def evaluate(args, model, dataloader, device):
    '''
    Args:
        args (argparse.Namespace): Input arguments
        model (torch.nn.Module): Model to evaluate
        dataloader (torch.utils.data.DataLoader): Dataloader
        device (torch.device): Device to use
    '''
    seen = 0
    jdict, stats, ap, ap_class = [], [], [], []
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    nc = len(dataloader.dataset.embedding_classes)
    plots = False
    save_dir = 'test/'
    names = dataloader.dataset.names
    count = 0

    for i, batch in enumerate(dataloader):
        images, boxes, targets, metadata = batch

        if i % 100 == 0:
            print('Evaluating batch {}/{}'.format(i+1, len(dataloader)))

        images = images.float().to(device)
        boxes = boxes.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            if args.model_type == 'DINOv2RPN':
                preds = model(images, iou_thr=args.iou_thr, conf_thres=args.conf_thres)
            else:
                images /= 255
                preds = model(images, augment=False)
                preds = non_max_suppression(preds,
                                        args.conf_thres,
                                        args.iou_thr,
                                        multi_label=True,
                                        isdino=False)
        
        # Save images
        if args.save_dir is not None and args.save_images:
            filepath = os.path.join(args.save_dir, '{}.png'.format(count))
            plot_image_with_boxes(metadata['impath'][0], preds[0].cpu(), dataloader.dataset.embedding_classes, filepath, args.target_size)
            count += 1

        
        # Metrics
        for si, pred in enumerate(preds):
            keep = targets[si] > -1
            labels = targets[si, keep]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:]))
                continue
                
            predn = pred.clone()
            # Evaluate
            if nl:
                tbox = custom_xywh2xyxy(boxes[si, keep, :])  # target boxes
                labelsn = torch.cat((labels[..., None], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)

            stats.append((correct, pred[:, 4], pred[:, 5], labels[:]))  # (correct, conf, pcls, tcls)

    # Compute metrics
    nc = len(dataloader.dataset.embedding_classes)
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class

    pf = '%22s' + '%11i' * 2 + '%11.3g' * 4  # print format
    s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
    print(s)
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    # Print results per class
    if nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # File writing if save_dir is provided
    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        filename = 'results_{}.txt'.format(args.backbone_type)
        save_file_path = os.path.join(args.save_dir, filename)
        base_classes, new_classes = get_base_new_classes(args.dataset)
        

        with open(save_file_path, 'w') as file:
            file.write('Class Images Instances P R mAP50 mAP50-95\n')
            file.write('%22s%11i%11i%11.3g%11.3g%11.3g%11.3g\n' % ('all', seen, nt.sum(), mp, mr, map50, map))

            # Results per class
            if nc > 1 and len(stats):
                map50_base = map_base = mr_base = mp_base = 0
                map50_new = map_new = mr_new = mp_new = 0
                for i, c in enumerate(ap_class):
                    file.write('%22s%11i%11i%11.3g%11.3g%11.3g%11.3g\n' % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

                    # TODO: fix this --> fix this to use a numpy array and not so repetitive code
                    if names[c] in base_classes:
                        map50_base += ap50[i]
                        map_base += ap[i]
                        mr_base += r[i]
                        mp_base += p[i]
                    elif names[c] in new_classes:
                        map50_new += ap50[i]
                        map_new += ap[i]
                        mr_new += r[i]
                        mp_new += p[i]
                map50_base /= len(base_classes)
                map_base /= len(base_classes)
                mr_base /= len(base_classes)
                mp_base /= len(base_classes)
                map50_new /= len(new_classes)
                map_new /= len(new_classes)
                mr_new /= len(new_classes)
                mp_new /= len(new_classes)
                file.write('%22s%11i%11i%11.3g%11.3g%11.3g%11.3g\n' % ('total base', seen, nt.sum(), mp_base, mr_base, map50_base, map_base))
                file.write('%22s%11i%11i%11.3g%11.3g%11.3g%11.3g\n' % ('total new', seen, nt.sum(), mp_new, mr_new, map50_new, map_new))
                


def get_model(args):
    '''
    Loads the model to evaluate given the input arguments and returns it.
    
    Args:
        args (argparse.Namespace): Input arguments
    '''

    # Use GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.model_type == 'DINOv2RPN':
        from models.detector import OVDDetector
        # Load prototypes and background prototypes
        prototypes = torch.load(args.prototypes_path)
        bg_prototypes = torch.load(args.bg_prototypes_path) if args.bg_prototypes_path is not None else None
        model = OVDDetector(prototypes, bg_prototypes, scale_factor=args.scale_factor, backbone_type=args.backbone_type, target_size=args.target_size).to(device)
    elif args.model_type == 'yolo':
        from ultralytics import YOLO
        if args.ckpt is None:
            raise ValueError('You must provide a checkpoint path for YOLOv5')
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=args.ckpt).to(device)

    model.eval()
    return model, device

def main(args):
    print('Evaluating model: {}: {} on dataset: {}'.format(args.model_type, args.backbone_type,args.annotations_file))

    # Initialize dataloader
    real_indices = False if args.model_type == 'DINOv2RPN' else True 
    dataset = DINODataset(args.root_dir, args.annotations_file, torch.load(args.prototypes_path)['label_names'], augment=False, target_size=args.target_size, real_indices=real_indices)
    dataloader = test_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Load model
    model, device = get_model(args)

    # Perform evaluation
    evaluate(args, model, dataloader, device)

    print('Done!')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--annotations_file', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model_type', type=str, default='DINOv2RPN')
    parser.add_argument('--backbone_type', type=str, default='dinov2')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--prototypes_path', type=str, default=None)
    parser.add_argument('--bg_prototypes_path', type=str, default=None)
    parser.add_argument('--target_size', nargs=2, type=int, metavar=('width', 'height'), default=(560, 560))
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--iou_thr', type=float, default=0.1)
    parser.add_argument('--conf_thres', type=float, default=0.2)
    parser.add_argument('--scale_factor', nargs='+', type=int, default=2)
    parser.add_argument('--save_images', action='store_true', default=False)
    args = parser.parse_args()

    main(args)
