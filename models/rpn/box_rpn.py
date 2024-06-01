import torch
from detectron2.structures import ImageList
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from models.crop_rcnn import CropRCNN
from detectron2.config import CfgNode as CN
from detectron2.engine import DefaultTrainer
from detectron2.checkpoint import DetectionCheckpointer

def add_croptrainer_config(cfg):
    _C = cfg
    _C.CROPTRAIN = CN()
    _C.CROPTRAIN.USE_CROPS = False
    _C.CROPTRAIN.CLUSTER_THRESHOLD = 0.1
    _C.CROPTRAIN.CROPSIZE = (320, 476, 512, 640)
    _C.CROPTRAIN.MAX_CROPSIZE = 800
    _C.CROPTEST = CN()
    _C.CROPTEST.PREDICT_ONLY = False
    _C.CROPTEST.CLUS_THRESH = 0.3
    _C.CROPTEST.MAX_CLUSTER = 5
    _C.CROPTEST.CROPSIZE = 800
    _C.CROPTEST.DETECTIONS_PER_IMAGE = 800
    _C.MODEL.CUSTOM = CN()
    _C.MODEL.CUSTOM.FOCAL_LOSS_GAMMAS = []
    _C.MODEL.CUSTOM.FOCAL_LOSS_ALPHAS = []

    _C.MODEL.CUSTOM.CLS_WEIGHTS = []
    _C.MODEL.CUSTOM.REG_WEIGHTS = []
     
    # dataloader
    # supervision level
    _C.DATALOADER.SUP_PERCENT = 100.0  # 5 = 5% dataset as labeled set
    _C.DATALOADER.RANDOM_DATA_SEED = 42  # random seed to read data
    _C.DATALOADER.RANDOM_DATA_SEED_PATH = "dataseed/COCO_supervision.txt"

def add_fcos_config(cfg):
    _C = cfg
    _C.MODEL.FCOS = CN()
    _C.MODEL.FCOS.NORM = "GN"
    _C.MODEL.FCOS.NUM_CLASSES = 80
    _C.MODEL.FCOS.NUM_CONVS = 4
    _C.MODEL.FCOS.SCORE_THRESH_TEST = 0.01
    _C.MODEL.FCOS.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]

def setup(config_file):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_croptrainer_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg

def get_box_RPN(config_file, checkpoint_file):
    Trainer = DefaultTrainer
    cfg = setup(config_file)
    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model).resume_or_load(checkpoint_file, resume=False)
    model.eval()
    return cfg, model

class BoxRPN(torch.nn.Module):

    def __init__(self,
                config_file='configs/FasterRCNN_FPN_DOTA_config.yaml',
                checkpoint_file='weights/FasterRCNN_FPN_DOTA_final_model.pth'
                ):
        super().__init__()
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.box_norm_factor = 10   # Used to normalize the positive bounding box scores to be in the same range as the class scores
        
        self.cfg, self.model = get_box_RPN(self.config_file, self.checkpoint_file)

    def forward(self, images):
        '''
        Generate box proposals using the model's RPN.

        Args:
            images (torch.Tensor): Input tensor with shape (B, C, H, W)
        '''
        images = [(x - self.model.pixel_mean) / self.model.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.model.backbone.size_divisibility,
            padding_constraints=self.model.backbone.padding_constraints,
        )
        features = self.model.backbone(images.tensor)

        with torch.no_grad():
            proposals, _ = self.model.proposal_generator(images, features, None)
        
        boxes = torch.stack([p.proposal_boxes.tensor for p in proposals])
        box_scores = torch.stack([p.objectness_logits / self.box_norm_factor for p in proposals])

        return boxes, box_scores
