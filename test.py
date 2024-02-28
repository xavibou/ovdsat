
from mmdet.apis import init_detector#, inference_detector
import mmrotate
import cv2
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import math

from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose

def convert_to_xyxyxyxy(bboxes):
        N = bboxes.shape[0]
        xyxyxyxy = torch.zeros(N, 4, 2)
        #breakpoint()
        for i, bbox in enumerate(bboxes):
            xc, yc, w, h, ag = bbox.cpu().numpy()
            wx, wy = w / 2 * np.cos(ag), w / 2 * np.sin(ag)
            hx, hy = -h / 2 * np.sin(ag), h / 2 * np.cos(ag)
            x1, y1 = xc - wx - hx, yc - wy - hy
            x2, y2 = xc + wx - hx, yc + wy - hy
            x3, y3 = xc + wx + hx, yc + wy + hy
            x4, y4 = xc - wx + hx, yc - wy + hy
            xyxyxyxy[i] = torch.tensor([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        return xyxyxyxy

def obb_to_bbox(obb_tensor):
    """
    Convert oriented bounding boxes (OBB) to regular bounding boxes (bbox).
    
    Args:
        obb_tensor (torch.Tensor): Tensor of shape [N, 4, 2] containing N oriented bounding boxes.
                                    Each bounding box is represented by the coordinates of its four corners
                                    in the format xyxyxyxy.
    
    Returns:
        bbox_tensor (torch.Tensor): Tensor of shape [N, 4] containing N regular bounding boxes.
                                    Each bounding box is represented by the coordinates of its top-left and
                                    bottom-right corners in the format x1y1x2y2.
    """
    # Extract coordinates of the four corners
    x_min = torch.min(obb_tensor[:, :, 0], dim=1)[0]
    y_min = torch.min(obb_tensor[:, :, 1], dim=1)[0]
    x_max = torch.max(obb_tensor[:, :, 0], dim=1)[0]
    y_max = torch.max(obb_tensor[:, :, 1], dim=1)[0]
    
    # Stack to form regular bounding boxes
    bbox_tensor = torch.stack([x_min, y_min, x_max, y_max], dim=1)
    
    return bbox_tensor
        
class OBBRPN(torch.nn.Module):

    def __init__(self,
                config_file='/mnt/ddisk/boux/code/mmrotate/oriented_rcnn_r50_fpn_1x_dota_le90.py',
                checkpoint_file='/mnt/ddisk/boux/code/mmrotate/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth'
                ):
        super().__init__()
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        #self.box_norm_factor = 10   # Used to normalize the positive bounding box scores to be in the same range as the class scores
        
        self.model = init_detector(config_file, checkpoint_file)
        self.test_pipeline = Compose(self.model.cfg.data.test.pipeline)
    
    def generate_masks_from_boxes(self, boxes_tensor, image_size):
        masks = []
        for box in boxes_tensor:
            # Convert box coordinates to numpy array
            box = box.cpu().numpy()

            # Create a blank image and draw the box on it
            img = Image.new('L', image_size, 0)
            draw = ImageDraw.Draw(img)
            draw.polygon(box, fill=255)

            # Convert the PIL image to a PyTorch tensor
            mask = F.to_tensor(img)
            masks.append(mask[0])
        
        # Stack if any, else return an empty tensor
        if len(masks) == 0:
            return torch.tensor([])
        masks_tensor = torch.stack(masks)
        return masks_tensor
    
    def inference_detector(self, imgs):
        """Inference image(s) with the detector.

        Args:
            model (nn.Module): The loaded detector.
            imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
            Either image files or loaded images.

        Returns:
            If imgs is a list or tuple, the same length list type results
            will be returned, otherwise return the detection results directly.
        """

        if isinstance(imgs, (list, tuple)):
            is_batch = True
        else:
            imgs = [imgs]
            is_batch = False

        cfg = self.model.cfg
        device = next(self.model.parameters()).device  # model device

        if isinstance(imgs[0], np.ndarray):
            cfg = cfg.copy()
            # set loading pipeline type
            cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
        cfg.data.test.pipeline[1]['img_scale'] = imgs[0].shape[:2]
        test_pipeline = Compose(cfg.data.test.pipeline)
        

        datas = []
        for img in imgs:
            # prepare data
            if isinstance(img, np.ndarray):
                # directly add img
                data = dict(img=img)
            else:
                # add information into dict
                data = dict(img_info=dict(filename=img), img_prefix=None)
            # build the data pipeline
            data = test_pipeline(data)
            datas.append(data)

        data = collate(datas, samples_per_gpu=len(imgs))
        # just get the actual data from DataContainer
        data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
        data['img'] = [img.data[0] for img in data['img']]

        if next(self.model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [device])[0]
        else:
            for m in self.model.modules():
                assert not isinstance(
                    m, RoIPool
                ), 'CPU inference with RoIPool is not supported currently.'
        
        # forward the model
        with torch.no_grad():
            #results = model(return_loss=False, rescale=True, **data)
            x = self.model.extract_feat(data['img'][0])
            proposals = self.model.rpn_head.simple_test_rpn(x, data['img_metas'][0])
            
        return proposals

    def forward(self, images):
        '''
        Generate oriented bounding box proposals using the model's RPN.

        Args:
            images (torch.Tensor): Input tensor with shape (B, C, H, W)
        '''

        # extract oriented bounding boxes the model
        proposals = []
        masks = []
        scores = []
        for image in images:
            image = image.cpu().numpy().transpose(1, 2, 0)
            result = self.inference_detector(image)
            p = convert_to_xyxyxyxy(result[0][:,:5])
            m = self.generate_masks_from_boxes(p, image.shape[:2])
            proposals.append(obb_to_bbox(p))
            scores.append(result[0][:,5])
            masks.append(m)

        proposals = torch.stack(proposals)
        masks = torch.stack(masks)
        scores = torch.stack(scores)

        return proposals, masks, scores


# test script

def make_batch(image_src, image_filenames, size=(602, 602)):
    images = []
    for filename in image_filenames:
        img = cv2.imread(image_src + filename)
        img = cv2.resize(img, size).transpose(2, 0, 1)
        images.append(torch.tensor(img))
    return torch.stack(images)


config_file = '/mnt/ddisk/boux/code/mmrotate/oriented_rcnn_r50_fpn_1x_dota_le90.py'
checkpoint_file = '/mnt/ddisk/boux/code/mmrotate/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth'
image_src = '/mnt/ddisk/boux/code/data/simd/val/'
image_filenames = [
    '0371.jpg',
    '0745.jpg',
    '4065.jpg'
]
target_size = (602, 602)

batch = make_batch(image_src, image_filenames, size=target_size).cuda()

model = OBBRPN(config_file, checkpoint_file)
model = model.cuda()

proposals, masks, scores = model(batch)

breakpoint()