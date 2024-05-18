import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import os

# radiate sdk
import sys
sys.path.insert(0, '..')
import radiate

# path to the sequence
root_path = '../data/radiate/'
sequence_name = 'tiny_foggy'

network = 'faster_rrcnn_R_101_FPN_3x'
setting = 'good_and_bad_weather_radar'

# time (s) to retrieve next frame
dt = 0.25

# load sequence
seq = radiate.Sequence(os.path.join(root_path, sequence_name), config_file="/home/2023120218/new_project/radiate_sdk/config/config.yaml")

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
# cfg.merge_from_file(os.path.join('test','config' , network + '.yaml'))
cfg.merge_from_file('/home/2023120218/new_project/radiate_sdk/vehicle_detection/test/config/faster_rcnn_R_50_FPN_3x.yaml')
cfg.MODEL.DEVICE = 'cpu'
# cfg.MODEL.WEIGHTS = os.path.join('weights',  network +'_' + setting + '.pth')
cfg.MODEL.WEIGHTS = "/home/2023120218/train_results/faster_rcnn_R_50_FPN_3x_good_weather/model_final.pth"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (vehicle)
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128]]
predictor = DefaultPredictor(cfg)

for t in np.arange(seq.init_timestamp, seq.end_timestamp, dt):
    output = seq.get_from_timestamp(t)
    if output != {}:
        radar = output['sensors']['radar_cartesian']
        camera = output['sensors']['camera_left_rect']
        predictions = predictor(radar)


        predictions_class = predictions["instances"].pred_classes
        predictions = predictions["instances"].to("cpu")
        print(predictions_class)



        boxes = predictions.pred_boxes

        objects = []

        for box in boxes:
            if cfg.MODEL.PROPOSAL_GENERATOR.NAME == 'RRPN':
                bb, angle = box.numpy()[:4], box.numpy()[4]
            else:
                bb, angle = box.numpy(), 0
                bb[2] = bb[2] - bb[0]
                bb[3] = bb[3] - bb[1]
            objects.append({'bbox': {'position': bb, 'rotation': angle}, 'class_name': 'vehicle'})

        radar = seq.vis(radar, objects, color=(255,0,0))
        bboxes_cam = seq.project_bboxes_to_camera(objects,
                                                seq.calib.left_cam_mat,
                                                seq.calib.RadarToLeft)
        # camera = seq.vis_3d_bbox_cam(camera, bboxes_cam)
        camera = seq.vis_bbox_cam(camera, bboxes_cam)

        cv2.imshow('radar', radar)
        cv2.imshow('camera_left_rect', camera)

        cv2.waitKey(1)


# a = {'instances': Instances(num_instances=2, image_height=1152, image_width=1152, fields=[pred_boxes: Boxes(tensor([[581.5248            , 364.1570, 613.4710, 437.6550],
#         [587.1679, 234.9748, 612.1450, 261.9734]])), scores: tensor([1.0000, 1.0000]), pred_classes: tensor([0, 0])])}


