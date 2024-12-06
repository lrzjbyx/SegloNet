import copy
import os
import json
import time

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from core.align.align import Align
from core.converter.convert import convert_to_arc, convert_to_line, centerline_to_poly, line_to_poly, arc_to_poly, \
    circle_to_poly, convert_polygon
from core.dataset import transforms
from core.dataset.seal import create_predict_dict
from core.models.sealnet import SealNet, u2_hr_backbone
from core.tools.draw_utils import draw_keypoints,point_color
from core.train_utils.calculate import *
from core.train_utils.draw import plot_region

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def readJsonFile(path):
    with open(path, 'r',encoding='utf-8') as f:
        data = json.load(f)
    return data





def predict_all_seal(src_folder,output_folder,weights_path,eval_json_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resize_hw = (256, 256)
    data_transform = transforms.Compose([
        transforms.AffineTransform(scale=(1.25, 1.25), fixed_size=resize_hw),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])



    # create model
    model = u2_hr_backbone(key_joints=10)
    weights = torch.load(weights_path, map_location=device)
    weights = weights if "model" not in weights else weights["model"]
    model.load_state_dict(weights)
    model.to(device)
    model.eval()

    image_paths = [p for p in os.listdir(src_folder) if p.endswith(".png")]

    predict_list =[]

    for file_index,file_name in enumerate(image_paths):
        print(file_name)
        # if file_index ==0:
        #     continue
        # if file_index >1:
        #     break

        img_path = os.path.join(src_folder,file_name)
        out_path = os.path.join(output_folder,file_name)
        instance, target = create_predict_dict(img_path)
        instance, target = data_transform(instance, target)
        image = cv2.imread(img_path)
        polys = []

        print("=========================================")
        with torch.inference_mode():
            torch.unsqueeze(instance['resized_image'], dim=0)
            instance['resized_image'] = torch.unsqueeze(instance['resized_image'], dim=0).to(device)
            pred_images, (keypoints, scores, sn, type, polygons, shrink_polygons) = model(instance, [target])

            try:
                for i, keypoint in enumerate(keypoints):
                    # cv2.fillPoly(image, [polygons[i].astype(np.int32)], (100, 100, 0), 1)
                    # for j, k in enumerate(keypoint):
                    #     cv2.circle(image, (int(k[0]), int(k[1])), 2, point_color[j], 2)
                    # cv2.imwrite("sdawasd.png", image)
                    # cv2.polylines(image, [polygons[i].reshape(-1, 2).astype(np.int32)], True, (197, 0, 113), 2)
                    if type[i] == 5:
                        item = convert_to_arc(keypoint, image, polygons[i])
                        poly = centerline_to_poly(item)
                        # plot_region([poly.reshape(-1, 2).astype(np.int32)],image)
                        # cv2.polylines(image, [poly.reshape(-1, 2).astype(np.int32)], True, (192, 255, 0), 2)
                        # cv2.imwrite("error/{0}-result-arc{1}.png".format(os.path.basename(file_name).split(".")[0], i),
                        #             Align(300, 60).run(image, item))

                    else:
                        item = convert_to_line(keypoint, image, polygons[i])
                        poly = centerline_to_poly(item)
                        # cv2.polylines(image, [polygons[i].reshape(-1, 2).astype(np.int32)], True, (255, 255, 0), 2)
                        # cv2.polylines(image, [poly.reshape(-1, 2).astype(np.int32)], True, (255, 0, 192), 2)
                        # cv2.imwrite("error/{0}-result-line{1}.png".format(os.path.basename(file_name).split(".")[0], i),
                        #             Align(300, 60).run(image, item))

                    polys.append(poly.reshape(-1, 2).astype(np.int32))

                    ann_t = {}
                    ann_t["image_id"] = file_index + 1
                    ann_t["category_id"] = 1
                    ann_t["bbox"] = list(cv2.boundingRect(poly.reshape(-1, 2).astype(np.int32)))
                    ann_t["score"] = round(float(np.mean(scores[i])),5)
                    ann_t["segmentation"] = [np.round(poly.reshape(-1),1).tolist()]

                    # cv2.rectangle(image, (ann_t["bbox"][0], ann_t["bbox"][1]), (ann_t["bbox"][0] + ann_t["bbox"][2], ann_t["bbox"][1] + ann_t["bbox"][3]), (255, 123, 5), 2)

                    predict_list.append(ann_t)

                plot_region(polys,keypoints, image,out_path)

            except Exception as e:
                print(e)
                continue


    print("===================predict json file======================")
    json_str = json.dumps(predict_list, indent=4)
    with open(eval_json_path, 'w') as json_file:
        json_file.write(json_str)
    print("===================predict json file======================")


def load_annotation(json_folder,ann_json_path):
    predict_dict = {
        "categories": [
            {
                "supercategory": "seal",
                "id": 1,
                "name": "seal"
            }
        ],
        "images": [],
        "annotations": []
    }

    image_template = {
        "file_name": "",
        "height": 0,
        "width": 0,
        "id": 1
    }

    annotation_template = {
        "segmentation": [
            []
        ],
        "area": 0,
        "iscrowd": 0,
        "image_id": 1,
        "bbox": [0],
        "category_id": 1,
        "id": 1
    }

    json_paths = [p for p in os.listdir(json_folder) if p.endswith(".json")]
    instance_id = 1
    for json_index,json_path in enumerate(json_paths):
        print(json_path)

        # if json_index ==0:
        #     continue
        #
        # if json_index >1:
        #     break
        annotation_file = os.path.join(json_folder, json_path)
        annotation_data = readJsonFile(annotation_file)

        annotation_image = annotation_data["image"]

        image = cv2.imdecode(np.fromfile(os.path.join(json_folder, annotation_image["filename"]), dtype=np.uint8), -1)

        it = copy.deepcopy(image_template)
        it["file_name"] = annotation_image["filename"]
        it["height"] = annotation_image["height"]
        it["width"] = annotation_image["width"]
        it["id"] = json_index + 1

        predict_dict["images"].append(it)

        annotation_labels = annotation_data["label"]
        for annotation_label in annotation_labels:
            for group_index, annotation_group in enumerate(annotation_label["groups"]):
                for link_index, link in enumerate(annotation_group["links"]):

                    pts = convert_polygon(link["entity"])

                    cv2.polylines(image, [pts.reshape(-1, 2).astype(np.int32)], True, (255, 123, 15), 2)


                    at = copy.deepcopy(annotation_template)
                    at["segmentation"] = [np.round(pts.reshape(-1),1).tolist()]
                    at["area"] = cv2.contourArea(pts.reshape(-1, 2).astype(np.int32))
                    at["iscrowd"] = 0
                    at["image_id"] = json_index + 1
                    at["bbox"] = list(cv2.boundingRect(pts.reshape(-1, 2).astype(np.int32)))
                    at["category_id"] = 1
                    at["id"] = instance_id

                    instance_id+=1

                    predict_dict["annotations"].append(at)

        # cv2.imwrite("ak/{0}".format(annotation_image["filename"]), image)

    print("===================annotation json file======================")
    json_str = json.dumps(predict_dict, indent=4)
    with open(ann_json_path, 'w') as json_file:
        json_file.write(json_str)
    print("===================annotation json file======================")



if __name__ == '__main__':
    weights_path = "save_weights/model_best.pth"
    # weights_path = "save_weights/model_best1.pth"
    # weights_path = "save_weights/model_best2.pth"
    # weights_path = "save_weights/model_165.pth"

    load_annotation(r"\\192.168.1.225\share\test_seal","ground_truth.json")
    predict_all_seal(r"\\192.168.1.225\share\test_seal", "sourcell", weights_path, "predictions-kernel-2-key-10.json")

    annType = ['segm', 'bbox', 'keypoints']
    annType = annType[0]  # specify type here

    annFile = "ground_truth.json"
    cocoGt = COCO(annFile)

    resFile = 'predictions.json'
    # resFile = 'predictions-kernel-4-key-20.json'
    cocoDt = cocoGt.loadRes(resFile)

    imgIds = sorted(cocoGt.getImgIds())

    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = imgIds
    # 设置要计算的IoU阈值
    # cocoEval.params.iouThrs = [0.5, 0.7]
    # 设置最大检测数量
    # cocoEval.params.maxDets = [1, 5, 10]
    # cocoEval.params.iouThrs = [0.5]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    # print('AP@.5:.05:.95:', cocoEval.stats[0])
    # print('AP@.5:', cocoEval.stats[1])
    # print('AP@.75:', cocoEval.stats[2])
    # print('AR@.5:.05:.95:', cocoEval.stats[8])
    # print('AR@.5:', cocoEval.stats[9])
    # print('AR@.75:', cocoEval.stats[10])






