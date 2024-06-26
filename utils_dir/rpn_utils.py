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

def get_box_RPN(yaml_path, checkpoint_path):
    Trainer = DefaultTrainer
    cfg = setup(yaml_path)
    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model).resume_or_load(checkpoint_path, resume=False)
    model.eval()
    return cfg, model