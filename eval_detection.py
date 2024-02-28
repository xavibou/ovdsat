import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from skimage import color


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
from models.detector import OVDDetector


from utils_dir.metrics import ConfusionMatrix, ap_per_class, box_iou
from datasets import get_base_new_classes

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
    original_image_size = (1024, 1024)
    image = cv2.resize(image, original_image_size)
    # Create figure and axes
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Define a list of colors for each class
    colors = ['lightgreen', 'lightgreen', 'darkmagenta', 'cyan', 'magenta', 'cornflowerblue', 'white', 'white', 'cornflowerblue', 'gold', 
              'brown', 'cyan', 'gray', 'olive', 'teal', 'darkblue', 'darkgreen', 'darkblue', 'darkcyan', 'darkmagenta']

    # Plot bounding boxes with different colors for each class
    for detection in detections:
        xmin, ymin, xmax, ymax, conf, class_id = detection

        # Scale the bounding box coordinates to the original image size
        xmin = int(xmin * original_image_size[0] / target_size[0])
        ymin = int(ymin * original_image_size[1] / target_size[1])
        xmax = int(xmax * original_image_size[0] / target_size[0])
        ymax = int(ymax * original_image_size[1] / target_size[1])

        # Get label name based on class_id
        label_name = label_names[int(class_id)]

        # Calculate box width and height
        width = xmax - xmin
        height = ymax - ymin

        # Create a Rectangle patch with a color corresponding to the class
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor=colors[int(class_id)], facecolor='none')
        ax.add_patch(rect)

        # Add label name above the bounding box with the same color and background
        bbox_props = dict(boxstyle="square", fc=colors[int(class_id)], ec="black", lw=0)

        # Adjust text position to start at the same x-coordinate as the bounding box
        text_x = xmin+9.5

        if colors[int(class_id)] in ['blue', 'darkmagenta']:
            text_color = 'white'
        else:
            text_color = 'black'
        
        if label_name == 'figther-aircraft':
            label_name = 'fighter-aircraft'
        
        ax.text(text_x, ymin - 10, label_name, color=text_color, fontsize=10, ha='left', va='bottom', bbox=bbox_props)

    # Save the plot to the specified directory
    plt.axis('off')
    save_path = f"{filepath}"
    plt.savefig(save_path[:-4] + '.pdf', format='pdf', transparent=True)
    plt.savefig(save_path[:-4] + '.png', transparent=True)
    plt.close()



def prepare_model(args):
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

    # Load prototypes and background prototypes
    prototypes = torch.load(args.prototypes_path)
    bg_prototypes = torch.load(args.bg_prototypes_path) if args.bg_prototypes_path is not None else None
    model = OVDDetector(prototypes, bg_prototypes, scale_factor=args.scale_factor, backbone_type=args.backbone_type, target_size=args.target_size, classification=args.classification).to(device)
    #model.eval() 
    return model, device

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

def eval_detection(args, model, val_dataloader, device):
    seen = 0
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    nc = val_dataloader.dataset.get_category_number()
    names = model.classifier.get_categories()
    count = 0

    stats = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader), leave=False):

            #if i not in [141, 34, 514, 61]:
            if i not in [514]:
                count += 1
                continue

            if args.classification != 'mask':
                images, boxes, labels, _, metadata = batch
                boxes = boxes.to(device)
            else:
                images, _, labels, masks, _ = batch
                loc = masks.float().to(device)
            
            labels = map_labels_to_prototypes(val_dataloader.dataset.get_categories(), model.classifier.get_categories(), labels)
            images = images.float().to(device)
            labels = labels.to(device)

            preds = model(images, iou_thr=args.iou_thr, conf_thres=args.conf_thres, aggregation=args.aggregation)
            #preds = model(images, iou_thr=args.iou_thr, conf_thres=args.conf_thres, aggregation=args.aggregation, labels=custom_xywh2xyxy(boxes))

            filepath = os.path.join('/mnt/ddisk/boux/code/ovdsat/run/plots/detections_chosen', '{}.png'.format(count))
            plot_image_with_boxes(metadata['impath'][0], preds[0].cpu(), names, filepath, args.target_size)
            count += 1

            for si, pred in enumerate(preds):
                keep = labels[si] > -1
                targets = labels[si, keep]
                nl, npr = targets.shape[0], pred.shape[0]  # number of labels, predictions
                correct = torch.zeros(npr, len(iouv), dtype=torch.bool, device=device)  # init
                seen += 1

                if npr == 0:
                    if nl:
                        stats.append((correct, *torch.zeros((2, 0), device=device), targets[:]))
                    continue
                    
                predn = pred.clone()
                if nl:
                    tbox = custom_xywh2xyxy(boxes[si, keep, :])  # target boxes
                    labelsn = torch.cat((targets[..., None], tbox), 1)  # native-space labels
                    correct = process_batch(predn, labelsn, iouv)

                stats.append((correct, pred[:, 4], pred[:, 5], targets[:]))

    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class

    pf = '%22s' + '%11i' * 2 + '%11.3g' * 4  # print format
    s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
    print(s)
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    if nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        filename = 'results_{}.txt'.format(args.backbone_type)
        save_file_path = os.path.join(args.save_dir, filename)
        base_classes, new_classes = get_base_new_classes(args.dataset)
        
        with open(save_file_path, 'w') as file:
            file.write('Class Images Instances P R mAP50 mAP50-95\n')
            file.write('%22s%11i%11i%11.4g%11.4g%11.4g%11.4g\n' % ('all', seen, nt.sum(), mp, mr, map50, map))

            if nc > 1 and len(stats):
                map50_base = map_base = mr_base = mp_base = 0
                map50_new = map_new = mr_new = mp_new = 0
                for i, c in enumerate(ap_class):
                    file.write('%22s%11i%11i%11.4g%11.4g%11.4g%11.4g\n' % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

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
                file.write('%22s%11i%11i%11.4g%11.4g%11.4g%11.4g\n' % ('total base', seen, nt.sum(), mp_base, mr_base, map50_base, map_base))
                file.write('%22s%11i%11i%11.4g%11.4g%11.4g%11.4g\n' % ('total new', seen, nt.sum(), mp_new, mr_new, map50_new, map_new))

def main(args):
    print('Setting up training...')

    # Initialize dataloader
    _, val_dataloader = init_dataloaders(args)


    # Load model
    model, device = prepare_model(args)

    # Perform training
    eval_detection(
        args, 
        model, 
        val_dataloader, 
        device
    )

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--val_root_dir', type=str)
    parser.add_argument('--val_annotations_file', type=str)
    parser.add_argument('--prototypes_path', type=str)
    parser.add_argument('--bg_prototypes_path', type=str, default=None)
    parser.add_argument('--aggregation', type=str, default='mean')
    parser.add_argument('--classification', type=str, default='box')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--backbone_type', type=str, default='dinov2')
    parser.add_argument('--target_size', nargs=2, type=int, metavar=('width', 'height'), default=(560, 560))
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--scale_factor', nargs='+', type=int, default=2)
    parser.add_argument('--iou_thr', type=float, default=0.2)
    parser.add_argument('--conf_thres', type=float, default=0.001)
    args = parser.parse_args()
    main(args)