import os
import json

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

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

    for file_name in os.listdir(src_folder):
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
            # try:
            #     pred_images, (keypoints, scores, sn, type, polygons, shrink_polygons) = model(instance, [target])
            # except Exception as e:
            #     print(e)
            #     continue

            # if torch.is_tensor(pred_images):
            #     continue

            line_shrink_mask = pred_images[0][0].astype(np.uint8)
            oval_shrink_mask = pred_images[0][2].astype(np.uint8)

            cv2.imwrite("oval_shrink_maskwd.png",oval_shrink_mask)
            cv2.imwrite("line_shrink_maskwd.png",line_shrink_mask)
            for i, keypoint in enumerate(keypoints):
                if type[i] == 5:
                    dists = []
                    shrink_side_points = []
                    for j, k in enumerate(keypoint.astype(np.int32)):
                        # print(0< k[0]< oval_shrink_mask.shape[0] and 0<= k[1]< oval_shrink_mask.shape[1])
                        # if 0< k[0]< oval_shrink_mask.shape[0] and 0<= k[1]< oval_shrink_mask.shape[1] and oval_shrink_mask[k[0]][k[1]] != 255:
                        if 0< k[0]< oval_shrink_mask.shape[0] and 0<= k[1]< oval_shrink_mask.shape[1]:

                            # if cv2.isContourConvex(polygons[i].astype(np.int32)):
                            #     hull = cv2.convexHull(polygons[i].astype(np.int32))
                            #     # Simplify the convex hull to reduce the number of points
                            #     simplified_hull = cv2.approxPolyDP(hull, epsilon=0.01, closed=True)
                            #
                            #     # Reshape the simplified hull to have shape (200, 1, 2)
                            #     simplified_hull = simplified_hull.reshape((200, 1, 2))
                            #
                            # # canvas = target
                            #
                            cv2.fillPoly(image,[polygons[i].astype(np.int32)],(100,100,0),1)
                            # cv2.fillPoly(canvas, [np.array(mask_poly.exterior.coords, dtype=np.int32)],
                            #              color=(0, 255, 0))

                            # cv2.imwrite("dwadsda.png",image)

                            dist = cv2.pointPolygonTest(polygons[i].astype(np.float32), k.astype(np.float32), True)
                            dists.append(dist)
                            shrink_side_points.append(k)
                    # print(np.array(dists).max())
                    if len(shrink_side_points) >= 5:
                        params1 = cv2.fitEllipseAMS(np.array(shrink_side_points))
                    else:
                        params1 = cv2.fitEllipseAMS(keypoint.astype(np.int32))
                    # params1 = cv2.fitEllipseAMS(keypoint.astype(np.int32))
                    # params1 = cv2.fitEllipse(keypoint.astype(np.int32))
                    params1 = cv2.fitEllipse(keypoint.astype(np.int32))
                    cv2.ellipse(image, params1, (0, 0, 255), 2)
                    for j, k in enumerate(keypoint):
                        cv2.circle(image, (int(k[0]), int(k[1])), 2, point_color[j], 2)

                    print(params1)
                else:

                    rows, cols = image.shape[:2]
                    vx, vy, x, y = cv2.fitLine(keypoint.astype(np.int32), cv2.DIST_L2, 0, 0.01, 0.01)
                    lefty = int((-x * vy / vx) + y)
                    righty = int(((cols - x) * vy / vx) + y)
                    # cv2.line(image, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)
                    cv2.fillPoly(image, [polygons[i].astype(np.int32)], (100, 0, 192), 1)

                    for j, k in enumerate(keypoint):
                        cv2.circle(image, (int(k[0]), int(k[1])), 2, point_color[j], 2)




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

    # 圆弧行高计算
    rhc = RowHeightCalculator()

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
        # try:
        #     pred_images, (keypoints, scores, sn, type, polygons, shrink_polygons) = model(instance, [target])
        # except Exception as e:
        #     print(e)
        #     continue

        # if torch.is_tensor(pred_images):
        #     continue

        line_shrink_mask = pred_images[0][0].astype(np.uint8)
        oval_shrink_mask = pred_images[0][2].astype(np.uint8)

        cv2.imwrite("oval_shrink_maskwd.png", oval_shrink_mask)
        cv2.imwrite("line_shrink_maskwd.png", line_shrink_mask)
        for i, keypoint in enumerate(keypoints):
            if type[i] == 5:
                dists = []
                shrink_side_points = []
                for j, k in enumerate(keypoint.astype(np.int32)):
                    # print(0< k[0]< oval_shrink_mask.shape[0] and 0<= k[1]< oval_shrink_mask.shape[1])
                    # if 0< k[0]< oval_shrink_mask.shape[0] and 0<= k[1]< oval_shrink_mask.shape[1] and oval_shrink_mask[k[0]][k[1]] != 255:
                    if 0 < k[0] < oval_shrink_mask.shape[0] and 0 <= k[1] < oval_shrink_mask.shape[1]:
                        # if cv2.isContourConvex(polygons[i].astype(np.int32)):
                        #     hull = cv2.convexHull(polygons[i].astype(np.int32))
                        #     # Simplify the convex hull to reduce the number of points
                        #     simplified_hull = cv2.approxPolyDP(hull, epsilon=0.01, closed=True)
                        #
                        #     # Reshape the simplified hull to have shape (200, 1, 2)
                        #     simplified_hull = simplified_hull.reshape((200, 1, 2))
                        #
                        # # canvas = target
                        #
                        # cv2.fillPoly(image, [polygons[i].astype(np.int32)], (100, 100, 0), 1)
                        # cv2.fillPoly(canvas, [np.array(mask_poly.exterior.coords, dtype=np.int32)],
                        #              color=(0, 255, 0))

                        # cv2.imwrite("dwadsda.png",image)

                        dist = cv2.pointPolygonTest(polygons[i].astype(np.float32), k.astype(np.float32), True)
                        dists.append(dist)
                        shrink_side_points.append(k)
                hh = np.array(dists).max()
                print("hh:{0}".format(hh))
                if len(shrink_side_points) >= 5:
                    params1 = cv2.fitEllipseAMS(np.array(shrink_side_points))
                else:
                    params1 = cv2.fitEllipseAMS(keypoint.astype(np.int32))
                # params1 = cv2.fitEllipseAMS(keypoint.astype(np.int32))
                # params1 = cv2.fitEllipse(keypoint.astype(np.int32))
                params1 = cv2.fitEllipse(keypoint.astype(np.int32))
                cv2.ellipse(image, params1, (0, 0, 255), 2)
                # for j, k in enumerate(keypoint):
                #     cv2.circle(image, (int(k[0]), int(k[1])), 2, point_color[j], 2)

                print(params1)

                x2 = int(params1[0][0] + 200 * np.cos(np.deg2rad(params1[2]-90)))
                y2 = int(params1[0][1] + 200 * np.sin(np.deg2rad(params1[2]-90)))


                # 绘制直线
                cv2.line(image, (int(params1[0][0]), int(params1[0][1])), (x2, y2), (255, 255, 0), 5)

                print("======")
                # 计算开始角度
                # print((-params1[2]+360+90)%360)
                rotated_angle = -params1[2]+90
                # print((params1[2]-90+360)%360)
                start_angle = calculate_angle_ann([params1[0][0],params1[0][1]],keypoint[-1])
                end_angle = calculate_angle_ann([params1[0][0], params1[0][1]], keypoint[0])
                # print(start_angle)
                # print(end_angle)


                span_angle = counter_clockwise_subtract(end_angle,start_angle)
                print("start_angle:{0}".format(start_angle))
                print("end_angle:{0}".format(end_angle))
                print("旋转角度:{0}".format(rotated_angle))
                print("过度角度:{0}".format(span_angle))
                # print("开始角度:{0}".format(start_angle))
                diff = counter_clockwise_difference(rotated_angle,start_angle)
                print("开始角度:{0}".format((diff+360)%360))
                # print(counter_clockwise_difference(rotated_angle,start_angle))
                # counter_clockwise_difference(rotated_angle,start_angle)
                # print((rotated_angle + (rotated_angle - start_angle-360)%360+360)%360)
                # print(counter_clockwise_subtract(start_angle,rotated_angle))
                # print((rotated_angle - start_angle + 360) % 360)
                hh = rhc.calculate(image,params1[0],keypoint[4],2 * max(params1[1][0] / 2,params1[1][1] / 2),polygons[i].astype(np.int32))
                print(hh)
                item = {
                    "la": 1,
                    "mu": 0,
                    "x": 0,
                    "y": 0,
                    # "rect": [
                    #     params1[0][0] - params1[1][0] / 2, params1[0][0] - params1[1][1] / 2, params1[1][0],
                    #     params1[1][1]
                    # ],
                    "rect": [
                        params1[0][0] - params1[1][0] / 2, params1[0][1] - params1[1][1] / 2, params1[1][0],
                        params1[1][1]
                    ],
                    "rotation": -rotated_angle * 16,
                    "text": "",
                    "type": 6,
                    "sequence": "从左到右",
                    "startAngle": ((diff+360)%360) * 16,
                    "spanAngle": span_angle * 16,
                    "b": params1[1][0] / 2,
                    "a": params1[1][1] / 2,
                    "h": hh

                }

                print(item)


            else:

                rows, cols = image.shape[:2]
                vx, vy, x, y = cv2.fitLine(keypoint.astype(np.int32), cv2.DIST_L2, 0, 0.01, 0.01)
                lefty = int((-x * vy / vx) + y)
                righty = int(((cols - x) * vy / vx) + y)
                # cv2.line(image, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)
                cv2.fillPoly(image, [polygons[i].astype(np.int32)], (100, 0, 192), 1)

                for j, k in enumerate(keypoint):
                    cv2.circle(image, (int(k[0]), int(k[1])), 2, point_color[j], 2)

    cv2.imwrite(os.path.join(output_folder,os.path.basename(src_path)),image)









if __name__ == '__main__':

    # weights_path = "save_weights/model_50.pth"
    weights_path = "save_weights/model_best.pth"
    # src_path = r"\\192.168.1.225\share\test_images\1399.jpg"
    # src_path = r"Snipaste_2022-09-24_11-33-02.png"
    # src_path = r"experiment\images-11.jpg"
    # src_path = r"experiment\Snipaste_2022-09-24_11-33-02.png"
    # src_path = r"seal_dataset/val/image/2016.png"
    # src_path = r"seal_dataset/val/image/0435.png"
    # src_path = r"experiment\6666.png"
    # src_path = r"experiment\9999.png"
    src_path = r"experiment\6665.png"
    # src_path = r"experiment\images-35.jpg"
    # src_path = r"experiment\0435.png"

    # predict_single_seal(src_path,".",weights_path)

    # weights_path = "save_weights/model-220.pth"
    # weights_path = "save_weights/model-100.pth"
    # weights_path = "save_weights/model_42.pth"
    # weights_path = "save_weights/model_50.pth"
    # weights_path = "save_weights/model_best1.pth"
    # weights_path = "save_weights/model_best.pth"
    # predict_all_seal(r"\\192.168.1.225\share\test_images","source",weights_path)
    # predict_all_seal(r"seal_dataset/val/image","sourceval",weights_path)

    predict_all_seal(r"experiment","sourceex",weights_path)