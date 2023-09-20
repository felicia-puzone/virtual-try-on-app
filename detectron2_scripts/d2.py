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

def detectron2_predictor():
    cfg_k = get_cfg()  # get a fresh new config
    cfg_k.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    cfg_k.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
    cfg_k.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    predictor_k = DefaultPredictor(cfg_k)
    return predictor_k

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
    return results

def get_keypoints(input_im):
    pred=detectron2_predictor()
    out=pred(input_im)
    return extract_keypoints_op_from_d(out)

if __name__=="__main__":
    json_dir=r"F:\pycharm_projects_F\CV_P_13_2\detectron2_luna_scripts"
    op_dict=json.load(open(os.path.join(json_dir,"000000_2.json"),"r"))
    tens=op_dict["people"]["pose_keypoints"]
    tens=torch.Tensor(tens)
    tens=torch.reshape(tens,(-1,3))
    print(tens.shape)
    imdir = r"F:\pycharm_projects_F\CV_P_13_2\cp-vton-plus\data\train\images"
    im = cv2.imread(os.path.join(imdir, "000000_0.jpg"))
    out=get_keypoints(im)
    for i in range(0,18):
        out_im=torch.zeros((512, 384))
        print(out.shape)
        out_im[int(out[i,1]),int(out[i,0])]=1
        tens_im = torch.zeros((512, 384))
        tens_im[int(tens[i,1]),int(tens[i,0])]=1
        cv2.imshow("prova1", out_im.numpy())
        cv2.imshow("prova",tens_im.numpy())
        cv2.waitKey(600)