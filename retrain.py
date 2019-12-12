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
classes_new = (0,1,1,1,1,1,1,1,1,3,3,2,2,2,3,0,0,0,2,2,2,2,0)
class2coco = (0,1,2,3)

import csv
path = "./labels.csv"
image_names=[]
image_labels=[]
total_annos={}
with open(path,'r') as f:
  csv_read = csv.reader(f)
  for row in csv_read:
    image_names.append('./data-2019/trainval/'+row[0]+'_image.jpg')
    image_labels.append(row[1])
for i in range(1,len(image_names)):
  snapshot = image_names[i]

  proj = np.fromfile(snapshot.replace('_image.jpg', '_proj.bin'), dtype=np.float32)
  proj.resize([3, 4])

  try:
      bbox = np.fromfile(snapshot.replace('_image.jpg', '_bbox.bin'), dtype=np.float32)
  except FileNotFoundError:
      print('[*] bbox not found.')
      bbox = np.array([], dtype=np.float32)

  bbox = bbox.reshape([-1, 11])

  bbxs = []
  for k, b in enumerate(bbox):
      R = rot(b[0:3])
      t = b[3:6]

      sz = b[6:9]
      vert_3D, edges = get_bbox(-sz / 2, sz / 2)
      vert_3D = R @ vert_3D + t[:, np.newaxis]
      vert_2D = proj @ np.vstack([vert_3D, np.ones(vert_3D.shape[1])])
      vert_2D = vert_2D / vert_2D[2, :]
      c = classes_new[int(b[9])]
      bbx = [min(vert_2D[0,:]),min(vert_2D[1,:]),max(vert_2D[0,:]),max(vert_2D[1,:]),class2coco[c]]
      bbxs.append(bbx)
  total_annos[snapshot]=bbxs

import os
import json
import numpy as np
import glob
import cv2
import os
classname_to_id = {"unknown": 0, "car":1, "truck":2, "motorcycle":3}

class Csv2CoCo:

    def __init__(self,image_dir,total_annos):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
        self.image_dir = image_dir
        self.total_annos = total_annos

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w'), ensure_ascii=False, indent=2)  # indent=2 更加美观显示

    def to_coco(self, keys):
        self._init_categories()
        for key in keys:
            self.images.append(self._image(key))
            self.img_id += 1
        instance = {}
        instance['info'] = 'Jianing Created'
        instance['license'] = ['license']
        instance['images'] = self.images
        return instance

    def _init_categories(self):
        for k, v in classname_to_id.items():
            category = {}
            category['id'] = v
            category['name'] = k
            self.categories.append(category)

    def _image(self, path):
        image = {}
        img = cv2.imread(path)
        image['height'] = img.shape[0]
        image['width'] = img.shape[1]
        image['id'] = self.img_id
        image['file_name'] = path
        shapes = self.total_annos[path]
        self.annotations=[]
        for shape in shapes:
            bboxi = []
            for cor in shape[:-1]:
                bboxi.append(int(cor))
            label = shape[-1]
            bboxi[0]=max(0,bboxi[0])
            bboxi[1]=max(0,bboxi[1])
            bboxi[2]=min(1914,bboxi[2])
            bboxi[3]=min(1052,bboxi[3])
            if(label==0):
              annotation = self._annotation(bboxi,label)
              self.annotations.append(annotation)
              self.ann_id += 1
            if(label==1):
              annotation = self._annotation(bboxi,label)
              self.annotations.append(annotation)
              self.ann_id += 1
            if(label==2):
              annotation = self._annotation(bboxi,label)
              self.annotations.append(annotation)
              self.ann_id += 1
            if(label==3):
              annotation = self._annotation(bboxi,label)
              self.annotations.append(annotation)
              self.ann_id += 1
        image['annotations'] = self.annotations
        return image

    def _annotation(self, shape,label):
        # label = shape[-1]
        points = shape[:4]
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = int(label)
        annotation['segmentation'] = self._get_seg(points)
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
        annotation['area'] = self._get_area(points)
        return annotation

    def _get_box(self, points):
        min_x = points[0]
        min_y = points[1]
        max_x = points[2]
        max_y = points[3]
        return [min_x, min_y, max_x, max_y]

    def _get_area(self, points):
        min_x = points[0]
        min_y = points[1]
        max_x = points[2]
        max_y = points[3]
        return (max_x - min_x+1) * (max_y - min_y+1)
    # segmentation
    def _get_seg(self, points):
        min_x = points[0]
        min_y = points[1]
        max_x = points[2]
        max_y = points[3]
        h = max_y - min_y
        w = max_x - min_x
        a = []
        a.append([min_x,min_y, min_x,min_y+0.5*h, min_x,max_y, min_x+0.5*w,max_y, max_x,max_y, max_x,max_y-0.5*h, max_x,min_y, max_x-0.5*w,min_y])
        return a

if __name__ == '__main__':
    l2c_train = Csv2CoCo(image_dir="./",total_annos=total_annos)
    train_instance = l2c_train.to_coco(image_names[1:])
    l2c_train.save_coco_json(train_instance, './data-2019/trainval/via_region_data.json')

import os
import numpy as np
import json
from detectron2.structures import BoxMode

# write a function that loads the dataset into detectron2's standard format
def get_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    imgs_anns=imgs_anns["images"]
    for v in imgs_anns:
        record = {}
        
        filename = v["file_name"]
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = v["id"]
        record["height"] = height
        record["width"] = width
      
        annos = v["annotations"]
        objs = []
        for anno in annos:
            obj = {
                "bbox": anno["bbox"],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": anno["segmentation"],
                "category_id": anno["category_id"],
                "iscrowd": 0, 
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


from detectron2.data import DatasetCatalog, MetadataCatalog
for d in ["train"]:
    DatasetCatalog.register("car_train", lambda d=d: get_dicts("./data-2019/trainval/"))
    MetadataCatalog.get("car_train").set(thing_classes=["unknown", "car", "truck", "motorcycle"])
car_metadata = MetadataCatalog.get("car_train")
print(car_metadata)

dataset_dicts = get_dicts("./data-2019/trainval/")

for d in random.sample(dataset_dicts, 1):
    img = cv2.imread(d["file_name"])
    print(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=car_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    print(vis.get_image()[:, :, ::-1])
    cv2.imwrite('./image.png', vis.get_image()[:, :, ::-1])

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.merge_from_file("./detectron2_repo/configs/Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml")
cfg.MODEL.WEIGHTS = "./model_0039999_e76410.pkl"
# cfg.merge_from_file("./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
# cfg.MODEL.WEIGHTS = "./model_final_f10217.pkl"
# cfg.merge_from_file("./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
# cfg.MODEL.WEIGHTS = "./model_final_2d9806.pkl"
cfg.DATASETS.TRAIN = ("car_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 1
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 3000    # 3000 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80  # four classes, unknown, car, truck, motocycle

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
cfg.DATASETS.TEST = ("car_train", )
predictor = DefaultPredictor(cfg)

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("car_train", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "car_train")
inference_on_dataset(trainer.model, val_loader, evaluator)