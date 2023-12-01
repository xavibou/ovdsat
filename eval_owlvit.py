# Imports
import torch
import numpy as np
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from utils_dir.nms import non_max_suppression
from datasets.dataset import DINODataset
from utils_dir.metrics import ConfusionMatrix, ap_per_class, box_iou


## ----------------- FOR OWL-ViT -----------------##
from transformers import AutoProcessor
from torchvision.ops import batched_nms

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
    # Convert nx4 boxes from [xmin, ymin, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 2] = x[..., 0] + x[..., 2]  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3]  # bottom right y
    return y

def postprocess(all_pred_boxes, pred_classes, confidence_threshold=0.5, iou_threshold=0.1, target_size=(560, 560)):
    pred_boxes = all_pred_boxes.squeeze(0)
    pred_classes = pred_classes.squeeze(0)

    top = torch.max(pred_classes, dim=1)
    scores = top.values
    classes = top.indices

    idx = scores > confidence_threshold
    scores = scores[idx]
    classes = classes[idx]
    pred_boxes = pred_boxes[idx]

    idx = batched_nms(pred_boxes, scores, classes, iou_threshold=iou_threshold)
    classes = classes[idx]
    pred_boxes = pred_boxes[idx]
    scores = scores[idx]

    height, width = target_size
    pred_boxes = pred_boxes * height
    pred_boxes_with_scores = torch.cat([pred_boxes, scores[:, None], classes[:, None]], dim=1)

    return pred_boxes_with_scores

def evaluate(args, model, dataloader, device):
    seen = 0
    jdict, stats, ap, ap_class = [], [], [], []
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    nc = len(dataloader.dataset.embedding_classes)
    plots = False
    save_dir = 'test/'
    names = dataloader.dataset.names

    if args.model_type == 'owlvit':
        processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")

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
            elif args.model_type == 'yolo':
                images /= 255
                preds = model(images, augment=False)
                preds = non_max_suppression(preds,
                                        args.conf_thres,
                                        args.iou_thr,
                                        multi_label=True,
                                        isdino=False)
            elif args.model_type == 'owlvit':
                images = processor(images=images, return_tensors="pt", device=device)["pixel_values"]
                #raise ValueError('done')
                pred_boxes, _, pred_classes, _ = model(images.to(device))
                preds = postprocess(pred_boxes, pred_classes, args.conf_thres, args.iou_thr, args.target_size).unsqueeze(0)

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
                #tbox = boxes[si, keep, :]  # target boxes
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


def get_model(args):

    # Load model and dataset
    # Use GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.model_type == 'DINOv2RPN':
        from models.detector import OVDDetector
        # Load prototypes and background prototypes
        prototypes = torch.load(args.prototypes_path)
        bg_prototypes = torch.load(args.bg_prototypes_path)
        model = OVDDetector(prototypes, bg_prototypes, scale_factor=args.scale_factor, target_size=args.target_size).to(device)
    elif args.model_type == 'yolo':
        from ultralytics import YOLO
        if args.ckpt is None:
            raise ValueError('You must provide a checkpoint path for YOLOv5')
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=args.ckpt).to(device)
    elif args.model_type == 'owlvit':
        from models.owlvit import load_OWLViT
        queries = torch.load(args.prototypes_path)['label_names']
        model = load_OWLViT(queries, device)
        ckpt = '/mnt/ddisk/boux/code/ovdsat/run/train/test_owlvit/owlvit_model_weights.pth'
        model.load_state_dict(torch.load(ckpt))
        print('model loaded')

    model.eval()
    return model, device

def main(args):
    print('Evaluating model: {} on dataset: {}'.format(args.model_type, args.annotations_file))

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
    parser.add_argument('--annotations_file', type=str)
    parser.add_argument('--model_type', type=str, default='DINOv2RPN')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--prototypes_path', type=str, default=None)
    parser.add_argument('--bg_prototypes_path', type=str, default=None)
    parser.add_argument('--target_size', nargs=2, type=int, metavar=('width', 'height'), default=(560, 560))
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--iou_thr', type=float, default=0.1)
    parser.add_argument('--conf_thres', type=float, default=0.2)
    parser.add_argument('--scale_factor', nargs='+', type=int, default=2)
    args = parser.parse_args()

    main(args)