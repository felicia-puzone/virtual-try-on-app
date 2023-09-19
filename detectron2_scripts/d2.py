import detectron2
import torch
from detectron2.utils.logger import setup_logger
import numpy as np
import os, json, cv2, random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import json


def get_person_mask_from_output(outputs: dict):
    instances = outputs["instances"]
    people_inst = instances[instances.pred_classes == 0]
    inseg_fields = people_inst.get_fields()
    person_mask = torch.where(inseg_fields["pred_masks"].to("cpu"), 255,
                              torch.zeros(inseg_fields["pred_masks"].shape, dtype=torch.uint8)).numpy()
    print(person_mask.shape)
    print(person_mask)
    return person_mask

def extract_keypoints_op_from_d(outputs):
    keypoints = outputs["instances"].pred_keypoints
    keypoints.to("cpu")
    keypoints=torch.squeeze(keypoints)
    results=torch.zeros((18,3))
    #adapting detectron set to op order
    map=[
        0,15,14,17,16,5,2,6,3,7,4,11,8,12,9,13,10
    ]
    for i in range(0,17):
        results[map[i],:]=keypoints[i,:]

    results[1,:]=(results[5,:]+results[2,:])/2
    tens_im = torch.zeros((512, 384))
    tens_im[int(results[2, 1]), int(results[2, 0])] = 1
    tens_im[int(results[5, 1]), int(results[5, 0])] = 1
    tens_im[int(results[1, 1]), int(results[1, 0])] = 1
    print(results)
    cv2.imshow("tentativo",tens_im.numpy())
    cv2.waitKey()
    return results