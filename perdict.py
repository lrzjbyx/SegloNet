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

def predict_all_seal(src_folder,output_folder):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resize_hw = (256, 256)
    data_transform = transforms.Compose([
        transforms.AffineTransform(scale=(1.25, 1.25), fixed_size=resize_hw),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # weights_path = "save_weights/model-209.pth"
    weights_path = "save_weights/model-137.pth"
    weights_path = "save_weights/model-136.pth"
    weights_path = "save_weights/model-182.pth"
    weights_path = "save_weights/model-32.pth"
    weights_path = "save_weights/model-227.pth"
    weights_path = "save_weights/model-220.pth"
    # weights_path = "save_weights/model-100.pth"


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

        with torch.inference_mode():
            torch.unsqueeze(instance['resized_image'], dim=0)
            instance['resized_image'] = torch.unsqueeze(instance['resized_image'], dim=0).to(device)
            try:
                pred_images, (keypoints, scores, sn, type, polygons, shrink_polygons) = model(instance, [target])
            except Exception as e:
                print(e)
                continue

            if torch.is_tensor(pred_images):
                continue

            line_shrink_mask = pred_images[0][0].astype(np.uint8)
            oval_shrink_mask = pred_images[0][2].astype(np.uint8)

            cv2.imwrite("oval_shrink_maskwd.png",oval_shrink_mask)
            for i, keypoint in enumerate(keypoints):
                if type[i] == 5:

                    shrink_side_points = []
                    for j, k in enumerate(keypoint.astype(np.int32)):
                        # print(0< k[0]< oval_shrink_mask.shape[0] and 0<= k[1]< oval_shrink_mask.shape[1])
                        # if 0< k[0]< oval_shrink_mask.shape[0] and 0<= k[1]< oval_shrink_mask.shape[1] and oval_shrink_mask[k[0]][k[1]] != 255:
                        if 0< k[0]< oval_shrink_mask.shape[0] and 0<= k[1]< oval_shrink_mask.shape[1]:
                            shrink_side_points.append(k)

                    if len(shrink_side_points) >= 5:
                        params1 = cv2.fitEllipseAMS(np.array(shrink_side_points))
                    else:
                        params1 = cv2.fitEllipseAMS(keypoint.astype(np.int32))
                    # params1 = cv2.fitEllipseAMS(keypoint.astype(np.int32))
                    # params1 = cv2.fitEllipse(keypoint.astype(np.int32))
                    params1 = cv2.fitEllipseDirect(keypoint.astype(np.int32))
                    cv2.ellipse(image, params1, (0, 0, 255), 2)
                    for j, k in enumerate(keypoint):
                        cv2.circle(image, (int(k[0]), int(k[1])), 2, point_color[j], 2)

                    print(params1)
                else:
                    try:
                        rows, cols = image.shape[:2]
                        vx, vy, x, y = cv2.fitLine(keypoint.astype(np.int32), cv2.DIST_L2, 0, 0.01, 0.01)
                        lefty = int((-x * vy / vx) + y)
                        righty = int(((cols - x) * vy / vx) + y)
                        cv2.line(image, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)
                    except  Exception as e:
                        continue

        cv2.imwrite(out_path, image)




def predict_single_seal():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    flip_test = False
    resize_hw = (256, 256)
    # img_path = "images-44_142.png"
    # img_path = "2222.png"
    # img_path = "myplot.png"
    img_path = "myplot1.png"
    # img_path = "Snipaste_2023-06-09_10-26-39.png"
    # img_path = "Snipaste_2023-06-09_10-29-04.png"
    # img_path = "pred_result-2.png"
    img_path = "zzdx.png"
    img_path = "images-34.jpg"
    img_path = "Snipaste_2023-06-26_16-28-32.png"
    img_path = "xuexiao.png"
    img_path = r"\\192.168.1.225\share\test_images\115.jpg"
    img_path = r"\\192.168.1.225\share\test_images\1377.jpg"
    img_path = r"\\192.168.1.225\share\test_images\1065.jpg"
    img_path = r"experiment/1-1PS00Z105L5.jpg"
    img_path = r"experiment/images-26.jpg"
    img_path = r"experiment/images-28.jpg"
    img_path = r"experiment/images-1.jpg"
    # img_path = "xuexiao.png"
    # img_path = r"\\192.168.1.225\share\test_images\1080.jpg"
    # img_path = "pred_result-2.png"
    # weights_path = "save_weights/model-359.pth"
    weights_path = "save_weights/model-209.pth"
    weights_path = "save_weights/model-95.pth"
    weights_path = "save_weights/model-112.pth"
    weights_path = "save_weights/model-137.pth"
    weights_path = "save_weights/model-136.pth"
    weights_path = "save_weights/model-32.pth"
    weights_path = "save_weights/model-32.pth"
    keypoint_json_path = "seal_keypoints2.json"
    assert os.path.exists(img_path), f"file: {img_path} does not exist."
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
    instance, target = create_predict_dict(img_path)
    instance, target = data_transform(instance,target)

    # create model
    model = u2_hr_backbone()
    weights = torch.load(weights_path, map_location=device)
    weights = weights if "model" not in weights else weights["model"]
    model.load_state_dict(weights)
    model.to(device)
    model.eval()
    image = cv2.imread(img_path)
    with torch.inference_mode():
        torch.unsqueeze(instance['resized_image'], dim=0)
        instance['resized_image'] = torch.unsqueeze(instance['resized_image'], dim=0).to(device)
        pred_images,(keypoints,scores,sn,type,polygons,shrink_polygons) = model(instance, [target])

        for i, keypoint in enumerate(keypoints):
            if type[i] == 5:
                # params1 = cv2.fitEllipseAMS(keypoint.astype(np.int32))
                # params1 = cv2.fitEllipse(keypoint.astype(np.int32))
                # params1 = cv2.fitEllipseDirect(keypoint[1:9].astype(np.int32))
                params1 = cv2.fitEllipseDirect(keypoint.astype(np.int32))
                cv2.ellipse(image, params1, (0, 0, 255), 2)
                for j,k in enumerate(keypoint):
                    cv2.circle(image, (int(k[0]), int(k[1])), 2 , point_color[j], 2)

                # params2 =  cv2.minEnclosingCircle(keypoint.astype(np.int32))
                # cv2.circle(image, (int(params2[0][0]), int(params2[0][1])), int(params2[1]), 255, 2)

                print(params1)
            else:
                rows, cols = image.shape[:2]
                vx, vy, x, y = cv2.fitLine(keypoint.astype(np.int32), cv2.DIST_L2, 0, 0.01, 0.01)
                lefty = int((-x * vy / vx) + y)
                righty = int(((cols - x) * vy / vx) + y)
                cv2.line(image, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)


    cv2.imwrite("123123123.png",image)







if __name__ == '__main__':
    # predict_single_seal()

    predict_all_seal(r"\\192.168.1.225\share\test_images","source")
    # predict_all_seal(r"seal_dataset/val/image","source")
    # predict_all_seal(r"experiment","source")