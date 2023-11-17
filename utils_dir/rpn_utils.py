
from detectron2.config import get_cfg
from models.centernet.config import add_centernet_config
from detectron2.engine.defaults import DefaultPredictor

def setup_cfg(yaml_path, checkpoint_path):
    # load config from file and command-line arguments
    weights = ['MODEL.WEIGHTS', '/mnt/ddisk/boux/code/dino_simple_detector/CenterNet2/models/CenterNet2_R50_1x.pth']
    cfg = get_cfg()
    add_centernet_config(cfg)
    cfg.merge_from_file(yaml_path)
    cfg.merge_from_list(weights)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0
    if cfg.MODEL.META_ARCHITECTURE in ['ProposalNetwork', 'CenterNetDetector']:
        cfg.MODEL.CENTERNET.INFERENCE_TH = 0
        cfg.MODEL.CENTERNET.NMS_TH = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0
    cfg.freeze()
    return cfg

def get_RPN(yaml_path, checkpoint_path):
    cfg = setup_cfg(yaml_path, checkpoint_path)
    predictor = DefaultPredictor(cfg)

    # Initialize the Faster R-CNN model
    return cfg, predictor