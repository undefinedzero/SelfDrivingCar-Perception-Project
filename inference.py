# Some basic setup
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from glob import glob
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def rot(n):
    n = np.asarray(n).flatten()
    assert(n.size == 3)

    theta = np.linalg.norm(n)
    if theta:
        n /= theta
        K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])

        return np.identity(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
    else:
        return np.identity(3)


def get_bbox(p0, p1):
    """
    Input:
    *   p0, p1
        (3)
        Corners of a bounding box represented in the body frame.

    Output:
    *   v
        (3, 8)
        Vertices of the bounding box represented in the body frame.
    *   e
        (2, 14)
        Edges of the bounding box. The first 2 edges indicate the `front` side
        of the box.
    """
    v = np.array([
        [p0[0], p0[0], p0[0], p0[0], p1[0], p1[0], p1[0], p1[0]],
        [p0[1], p0[1], p1[1], p1[1], p0[1], p0[1], p1[1], p1[1]],
        [p0[2], p1[2], p0[2], p1[2], p0[2], p1[2], p0[2], p1[2]]
    ])
    e = np.array([
        [2, 3, 0, 0, 3, 3, 0, 1, 2, 3, 4, 4, 7, 7],
        [7, 6, 1, 2, 1, 2, 4, 5, 6, 7, 5, 6, 5, 6]
    ], dtype=np.uint8)

    return v, e


classes = (
    'Unknown', 'Compacts', 'Sedans', 'SUVs', 'Coupes',
    'Muscle', 'SportsClassics', 'Sports', 'Super', 'Motorcycles',
    'OffRoad', 'Industrial', 'Utility', 'Vans', 'Cycles',
    'Boats', 'Helicopters', 'Planes', 'Service', 'Emergency',
    'Military', 'Commercial', 'Trains'
)


from glob import glob
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import os

files = glob('./data-2019/test/*/*.jpg')
guid_list=[]
label_list=[]
cfg = get_cfg()
cfg.merge_from_file("./detectron2_repo/configs/Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml")
# cfg.merge_from_file("./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
# cfg.merge_from_file("./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
# cfg.MODEL.WEIGHTS = "./model_final_f10217.pkl"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor = DefaultPredictor(cfg)

for snapshot in files:
  print(snapshot)
  im = cv2.imread(snapshot)

  xyz = np.fromfile(snapshot.replace('_image.jpg', '_cloud.bin'), dtype=np.float32)
  xyz = xyz.reshape([3, -1])

  proj = np.fromfile(snapshot.replace('_image.jpg', '_proj.bin'), dtype=np.float32)
  proj.resize([3, 4])

  uv = proj @ np.vstack([xyz, np.ones_like(xyz[0, :])])
  uv = uv / uv[2, :]

  dist = np.linalg.norm(xyz, axis=0)

  outputs=predictor(im)

  uv_map = np.vstack([uv[0:2,:], dist])

  dist_idx_show=[]
  labels=[1,2,3]
  for idx, label in enumerate(outputs["instances"].pred_classes.to("cpu").numpy()):
    if label in labels:
      bbx_arr = outputs["instances"].pred_boxes[idx].to("cpu").tensor.numpy()
      dist_idx=np.where(np.bitwise_and(
        np.bitwise_and(uv_map[0,:]>=bbx_arr[0,0], uv_map[1,:]>=bbx_arr[0,1]),
        np.bitwise_and(uv_map[0,:]<=bbx_arr[0,2], uv_map[1,:]<=bbx_arr[0,3]))==1)
      if np.sum(uv_map[2,dist_idx]):
        dist_temp = np.median(uv_map[2,dist_idx])
        if dist_temp < 50:
          dist_idx_show.append(dist_idx)
          label_test = label
          print(label_test)

    if dist_idx_show:
      break

  if not dist_idx_show:
    label_test = 0
    print(label_test)

  a = snapshot.replace('_image.jpg', '')
  b = a.replace('./data-2019/test/', '')
  guid_list.append(b)
  label_list.append(label_test)


import csv
path = "./test4.csv"
with open(path,'w') as f:
    csv_write = csv.writer(f)
    csv_head = ["guid/image","label"]
    csv_write.writerow(csv_head)
    for i in range(1799):
      csv_data = [guid_list[i],str(label_list[i])]
      csv_write.writerow(csv_data)