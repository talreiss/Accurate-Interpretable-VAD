import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

class Predictor:
    def __init__(self, confidence_threshold=0.5):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold  # threshold for the predictor
        self.cfg.MODEL.WEIGHTS = 'pre_processing/checkpoints/model_final_f10217.pkl'
        self.model = DefaultPredictor(self.cfg)

    def __call__(self, img):
        """
        This function gets an image and returns bounding boxes.
        """
        with torch.no_grad():
            result = self.model(img)
            instances = result['instances']
            bboxes = instances.get('pred_boxes')
            classes = instances.get('pred_classes')
        return bboxes.tensor, classes
