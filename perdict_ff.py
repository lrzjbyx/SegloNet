import os
import json

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from core.align.align import Align
from core.converter.convert import convert_to_arc, convert_to_line
from core.dataset import transforms
from core.dataset.seal import create_predict_dict
from core.models.sealnet import SealNet, u2_hr_backbone
from core.tools.draw_utils import draw_keypoints,point_color
from core.train_utils.calculate import *


def predict_all_seal(src_folder,output_folder,weights_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resize_hw = (256, 256)
    data_transform = transforms.Compose([
        transforms.AffineTransform(scale=(1.25, 1.25), fixed_size=resize_hw),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    # read single-person image
    # instance, target = create_predict_dict(img_path)
    # instance, target = data_transform(instance, target)

    # create model
    model = u2_hr_backbone()
    weights = torch.load(weights_path, map_location=device)
    weights = weights if "model" not in weights else weights["model"]
    model.load_state_dict(weights)
    model.to(device)
    model.eval()

    image_paths = [p for p in os.listdir(src_folder) if p.endswith(".png")]

    for file_name in image_paths:
        print(file_name)
        img_path = os.path.join(src_folder,file_name)
        out_path = os.path.join(output_folder,file_name)
        instance, target = create_predict_dict(img_path)
        instance, target = data_transform(instance, target)
        image = cv2.imread(img_path)
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
                    cv2.polylines(image, [polygons[i].reshape(-1, 2).astype(np.int32)], True, (197, 0, 113), 2)
                    if type[i] == 5:
                        item = convert_to_arc(keypoint, image, polygons[i])
                        cv2.imwrite("error/{0}-result-arc{1}.png".format(os.path.basename(file_name).split(".")[0], i),
                                    Align(300, 60).run(image, item))

                    else:
                        item = convert_to_line(keypoint, image, polygons[i])
                        cv2.polylines(image, [polygons[i].reshape(-1, 2).astype(np.int32)], True, (255, 255, 0), 2)
                        cv2.imwrite("error/{0}-result-line{1}.png".format(os.path.basename(file_name).split(".")[0], i),
                                    Align(300, 60).run(image, item))

                    item.pop("la")
                    item.pop("mu")
                    print(item)

            except Exception as e:
                continue

        print("=========================================")
        cv2.imwrite(out_path, image)
        print("=====")





def predict_single_seal(src_path,output_folder,weights_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    resize_hw = (256, 256)
    # # img_path = "images-44_142.png"
    # # img_path = "2222.png"
    # # img_path = "myplot.png"
    # img_path = "myplot1.png"
    # # img_path = "Snipaste_2023-06-09_10-26-39.png"
    # # img_path = "Snipaste_2023-06-09_10-29-04.png"
    # # img_path = "pred_result-2.png"
    # img_path = "zzdx.png"
    # img_path = "images-34.jpg"
    # img_path = "Snipaste_2023-06-26_16-28-32.png"
    # img_path = "xuexiao.png"
    # img_path = r"\\192.168.1.225\share\test_images\115.jpg"
    # img_path = r"\\192.168.1.225\share\test_images\1377.jpg"
    # img_path = r"\\192.168.1.225\share\test_images\1065.jpg"
    # img_path = r"experiment/1-1PS00Z105L5.jpg"
    # img_path = r"experiment/images-26.jpg"
    # img_path = r"experiment/images-28.jpg"
    # img_path = r"experiment/images-1.jpg"
    # img_path = "xuexiao.png"
    # img_path = r"\\192.168.1.225\share\test_images\1080.jpg"
    # img_path = "pred_result-2.png"

    keypoint_json_path = "seal_keypoints2.json"
    assert os.path.exists(src_path), f"file: {src_path} does not exist."
    assert os.path.exists(weights_path), f"file: {weights_path} does not exist."
    assert os.path.exists(keypoint_json_path), f"file: {keypoint_json_path} does not exist."

    data_transform = transforms.Compose([
        transforms.AffineTransform(scale=(1.25, 1.25), fixed_size=resize_hw),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # read json file
    with open(keypoint_json_path, "r") as f:
        person_info = json.load(f)

    # read single-person image
    instance, target = create_predict_dict(src_path)
    instance, target = data_transform(instance,target)

    # create model
    model = u2_hr_backbone()
    weights = torch.load(weights_path, map_location=device)
    weights = weights if "model" not in weights else weights["model"]
    model.load_state_dict(weights)
    model.to(device)
    model.eval()
    image = cv2.imread(src_path)
    with torch.inference_mode():
        torch.unsqueeze(instance['resized_image'], dim=0)
        instance['resized_image'] = torch.unsqueeze(instance['resized_image'], dim=0).to(device)
        pred_images, (keypoints, scores, sn, type, polygons, shrink_polygons) = model(instance, [target])


        for i, keypoint in enumerate(keypoints):
            # cv2.fillPoly(image, [polygons[i].astype(np.int32)], (100, 100, 0), 1)
            # cv2.polylines(image, [polygons[i].reshape(-1, 2).astype(np.int32)], True, (255, 255, 0), 2)
            for j, k in enumerate(keypoint):
                cv2.circle(image, (int(k[0]), int(k[1])), 2, point_color[j], 2)
            cv2.imwrite("sdawasd.png", image)

            if type[i] == 5:
                item = convert_to_arc(keypoint, image, polygons[i])
                cv2.imwrite("error/{0}-result-arc{1}.png".format(os.path.basename(src_path).split(".")[0], i),
                            Align(300, 60).run(image, item))
                # params1 = cv2.fitEllipseAMS(keypoint.astype(np.int32))
                # cv2.ellipse(image, params1, (0, 0, 255), 2)
            else:
                item = convert_to_line(keypoint, image, polygons[i])
                cv2.imwrite("error/{0}-result-line{1}.png".format(os.path.basename(src_path).split(".")[0], i),
                            Align(300, 60).run(image, item))


    cv2.imwrite(os.path.join(output_folder,os.path.basename(src_path)),image)









if __name__ == '__main__':

    # weights_path = "save_weights/model_50.pth"
    weights_path = "save_weights/model_best.pth"
    weights_path = "save_weights/model_best1.pth"
    weights_path = "save_weights/model_best2.pth"
    weights_path = "save_weights/model_22.pth"
    # src_path = r"\\192.168.1.225\share\test_images\1399.jpg"
    # src_path = r"\\192.168.1.225\share\test_images\1001.jpg"
    # src_path = r"Snipaste_2022-09-24_11-33-02.png"
    # src_path = r"experiment\images-11.jpg"
    # src_path = r"experiment\0524.png"
    # src_path = r"experiment\images-30.jpg"
    # src_path = r"experiment\images-32.jpg"
    # src_path = r"experiment\test.jpg"
    # src_path = r"experiment\Snipaste_2022-09-24_11-33-02.png"
    # src_path = r"seal_dataset/val/image/2016.png"
    # src_path = r"seal_dataset/val/image/0435.png"
    # src_path = r"experiment\6666.png"
    # src_path = r"experiment\9999.png"
    # src_path = r"experiment\6665.png"
    # src_path = r"experiment\images-35.jpg"
    # src_path = r"experiment\0435.png"

    ##### 错误 #####
    # 2363
    # src_path = r"\\192.168.1.225\share\test_images\1350.jpg"
    # src_path = r"\\192.168.1.225\share\test_images\1367.jpg"
    # src_path = r"\\192.168.1.225\share\test_images\1337.jpg"
    # src_path = r"\\192.168.1.225\share\test_images\1314.jpg"
    # src_path = r"\\192.168.1.225\share\test_images\1294.jpg"
    # src_path = r"\\192.168.1.225\share\test_images\1251.jpg"
    # src_path = r"\\192.168.1.225\share\test_images\1247.jpg"
    # src_path = r"\\192.168.1.225\share\test_images\1221.jpg"
    # src_path = r"\\192.168.1.225\share\test_images\1368.jpg"
    # src_path = r"\\192.168.1.225\share\test_images\1670.jpg"
    # src_path = r"\\192.168.1.225\share\test_images\1619.jpg"
    # src_path = r"\\192.168.1.225\share\test_images\141.jpg"
    # src_path = r"\\192.168.1.225\share\test_seal\download-2.png"

    src_path = r"\\192.168.1.225\share\test_images\1703.jpg"
    src_path = r"\\192.168.1.225\share\test_seal\t0131.png"
    src_path = r"\\192.168.1.225\share\test_seal\t0170.png"
    src_path = r"\\192.168.1.225\share\test_seal\t0048.png"
    src_path = r"\\192.168.1.225\share\test_seal\t0122.png"
    src_path = r"\\192.168.1.225\share\test_seal\t0086.png"




    # predict_single_seal(src_path,".",weights_path)

    # weights_path = "save_weights/model-220.pth"
    # weights_path = "save_weights/model-100.pth"
    # weights_path = "save_weights/model_42.pth"
    # weights_path = "save_weights/model_50.pth"
    # weights_path = "save_weights/model_best1.pth"
    # weights_path = "save_weights/model_best.pth"
    # predict_all_seal(r"\\192.168.1.225\share\test_images","source",weights_path)
    # predict_all_seal(r"\\192.168.1.225\share\test_seal","sourcell",weights_path)
    predict_all_seal(r"C:\Users\jackm\Desktop\png", "sourcell", weights_path)

    # predict_all_seal(r"seal_dataset/val/image","sourceval",weights_path)
    # #
    # predict_all_seal(r"experiment","sourceex",weights_path)