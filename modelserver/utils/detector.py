import atexit
import bisect
import multiprocessing as mp
from collections import deque

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from adet.config import get_cfg
from adet.utils.visualizer import TextVisualizer
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer


class Detector:
    def __init__(self, cfg_path, model_weight_path):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(cfg_path)
        # self.cfg.merge_from_list(model_weight_path)
        self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TEST[0] if len(self.cfg.DATASETS.TEST) else "__unused")
        self.cfg.MODEL.WEIGHTS = model_weight_path
        self.cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.3
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
        self.cfg.MODEL.FCOS.INFERENCE_TH_TEST = 0.3
        self.cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.3
        self.cfg.freeze()
        self.cpu_device = torch.device("cpu")
        self.vis_text = "TextHead" in self.cfg.MODEL.ROI_HEADS.NAME 
        self.predictor = DefaultPredictor(self.cfg)

    def predict(self, image):
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        if self.vis_text:
            visualizer = TextVisualizer(image, self.metadata, instance_mode=ColorMode.IMAGE,cfg=self.cfg)
        else:
            visualizer = Visualizer(image, self.metadata, instance_mode=ColorMode.IMAGE)

        if "bases" in predictions:
            self.vis_bases(predictions["bases"])
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(panoptic_seg.to(self.cpu_device), segments_info)
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(predictions["sem_seg"].argmax(dim=0).to(self.cpu_device))
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)
        # vis_output.save('result.jpg')
        return predictions, vis_output
        
