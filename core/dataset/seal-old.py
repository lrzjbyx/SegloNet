from torch.utils.data import Dataset
import os
import cv2
import numpy as np
from core.dataset.utils import readJsonFile, convertMask, convertPoints
import torch
import copy

class SealDataset(Dataset):

    def __init__(self, root: str, train: bool = True, transforms=None):
        assert os.path.exists(root), f"path '{root}' does not exist."
        if train:
            self.image_root = os.path.join(root, "./train/", "image")
            self.ann_root = os.path.join(root, "./train/", "ann")
        else:
            self.image_root = os.path.join(root, "./val/", "image")
            self.ann_root = os.path.join(root, "./val/", "ann")

        assert os.path.exists(self.image_root), f"path '{self.image_root}' does not exist."
        assert os.path.exists(self.ann_root), f"path '{self.ann_root}' does not exist."

        image_names = [p for p in os.listdir(self.image_root) if p.endswith(".png")]
        ann_names = [p for p in os.listdir(self.ann_root) if p.endswith(".json")]
        assert len(image_names) > 0, f"not find any images in {self.image_root}."

        # check images and mask
        re_ann_names = []
        for p in image_names:
            ann_name = "{0}.json".format(p.split(".")[0])
            assert ann_name in ann_names, f"{p} has no corresponding mask."
            re_ann_names.append(ann_name)
        ann_names = re_ann_names

        self.images_path = [os.path.join(self.image_root, n) for n in image_names]
        self.anns_path = [os.path.join(self.ann_root, n) for n in ann_names]

        self.transforms = transforms

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        image_path = self.images_path[idx]
        ann_path = self.anns_path[idx]
        image = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)
        assert image is not None, f"failed to read image: {image_path}"

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        h, w, _ = image.shape
        annotation_data = readJsonFile(ann_path)
        # 缩水多变形、原始多边形、序号编号、外接矩形、最小外接矩形
        shrink_polygon_points, polygon_points, polygon_ids, bboxs, rbboxs, polygon_types = convertMask(annotation_data)

        for shrink_polygon_point in shrink_polygon_points:
            cv2.polylines(image, [shrink_polygon_point.reshape(-1, 2).astype(np.int32)], True, (255, 123, 15), 2)

        cv2.imwrite("ak/{0}.png".format(idx),image)

        # 关键点
        key_points = convertPoints(annotation_data)



        # # 原始标签
        # original_params = convertParams(annotation_data)

        target = {
            "image_id":idx,
            # 原始图片
            "raw_image": image,
            # 加噪后图片
            "noisy_image":image,
            # 设置大小后的 图片
            "resized_image": None,
            # 设置图片大小的缩放尺寸
            "resized_hw": None,
            #  设置mask 后的图片
            #  直线图、直线缩水图、曲线图、曲线缩水图
            "masked_image": None,
            #  图片的宽高
            "raw_image_size": [0, 0, w, h],
            #  标注文件名称
            "ann_path": ann_path,
            #  原始图片路径
            "raw_image_path": image_path,
            #  实例的序号
            "instance_nm": None,
            #  多边形的点集
            "polygon_points": polygon_points,
            "resized_polygon_points": [],
            #  缩水多边形的点集
            "shrink_polygon_points": shrink_polygon_points,
            "resized_shrink_polygon_points": [],
            #  多边形类型
            "polygon_types": polygon_types,
            #  多边形的序号
            "polygon_ids": polygon_ids,
            #  关键点是否可见
            "visible": np.array([[2 for j in range(10)] for i in range(len(key_points))], dtype=np.float32),
            #  关键点
            "keypoints": key_points,
            "resized_keypoints": [],
            #  关键点的权重
            "kps_weights": [],
            #  生成的mask 后的实体
            "entity": [],
            # 实体的 中心点
            "entity_center": [],
            #  热力图
            "heatmap": [],
            #  外接矩形
            "bboxs": bboxs,
            #  最小外接矩形
            "rbboxs": rbboxs,
            #  文字标注
            "recs": [],
            # resize 的 M
            "trans": [],
            #  anti resize 的 M
            "reverse_trans": []
        }

        instance = {
            "resized_image": None,
            "masked_image": None,
            "entity": [],
            "heatmap": [],
            "entity_center": [],
            "kps_weights": [],
            "entity_sn": [],
            "visible": np.array([[2 for j in range(10)] for i in range(len(key_points))], dtype=np.float32),
            "entity_type": []
        }

        if self.transforms is not None:
            instance, target = self.transforms(instance, target)

        return instance, target

    def coco_index(self, idx):
        """
        该方法是专门为pycocotools统计标签信息准备，不对图像和标签作任何处理
        由于不用去读取图片，可大幅缩减统计时间

        Args:
            idx: 输入需要获取图像的索引
        """
        ann_path = self.anns_path[idx]
        annotation_data = readJsonFile(ann_path)
        image_info = {
             "height": annotation_data["image"]["height"],
             "width": annotation_data["image"]["width"],
             "id": idx
        }

        annotation_info = {
            "num_keypoints": 10,
            "area": 0,
            "iscrowd": 0,
            "keypoints": [],
            "image_id": idx,
            "bbox": [],
            "category_id": 1,
            "id": 0
        }
        annotation_infos = []
        key_points = convertPoints(annotation_data)
        _, _, _, bboxs, _, _ = convertMask(annotation_data)
        for i,key_point in enumerate(key_points):
            ann = copy.deepcopy(annotation_info)
            # bbox = [0,0,annotation_data["image"]["width"],annotation_data["image"]["height"]]
            bbox = list(bboxs[i])
            area = bbox[2] * bbox[3]
            keypoints = key_point.reshape(-1,2)
            arr_with_2 = np.hstack((keypoints, np.ones((keypoints.shape[0], 1)) * 2))

            keypoints = np.array(arr_with_2.reshape(-1), np.int32).tolist()

            ann["bbox"] =bbox
            ann["keypoints"] =keypoints
            ann["area"] =area
            annotation_infos.append(ann)


        return image_info, annotation_infos

    @staticmethod
    def collate_fn(batch):
        instance_tuple, targets_tuple = tuple(zip(*batch))

        # "resized_image": None,
        # "masked_image": None,
        # "entity": [],
        # "heatmap": [],
        # "entity_center": [],
        # "kps_weights": [],
        # "entity_sn": [],
        # "visible":

        resized_image = torch.stack([i['resized_image'] for i in instance_tuple])
        masked_image = torch.stack([i['masked_image'] for i in instance_tuple])
        entity = torch.concat([i['entity'] for i in instance_tuple])
        heatmap = torch.concat([i['heatmap'] for i in instance_tuple])
        entity_center = torch.concat([i['entity_center'] for i in instance_tuple])
        kps_weights = torch.concat([i['kps_weights'] for i in instance_tuple])
        visible = torch.concat([i['visible'] for i in instance_tuple])
        # 序号
        sn = [np.array([s for j in range(ii['entity'].shape[0])]).flatten() for s, ii in enumerate(instance_tuple)]
        flattened = np.array([item for array in sn for item in array.flatten()])
        entity_sn = torch.as_tensor(flattened, dtype=torch.int64)
        # 类型
        entity_type = torch.concat([i['entity_type'] for i in instance_tuple])
        # 每个标注的id
        entity_id = [np.array([j for j in range(ii['entity'].shape[0])]).flatten() for s, ii in enumerate(instance_tuple)]
        flattened_id = np.array([item for array in entity_id for item in array.flatten()])
        entity_id = torch.as_tensor(flattened_id, dtype=torch.int64)

        # resized_polygon_points = torch.concat([i['resized_polygon_points'] for i in instance_tuple])



        instance_tuple = {
            "resized_image": resized_image,
            "masked_image": masked_image,
            "entity": entity,
            "heatmap": heatmap,
            "entity_center": entity_center,
            "kps_weights": kps_weights,
            "visible": visible,
            # 区分batch sn
            "entity_sn": entity_sn,
            # 区域的类型
            "entity_type": entity_type,
            #  区域所在印章的序号
            "entity_id":entity_id
            # "resized_polygon_points":resized_polygon_points
        }

        return instance_tuple, targets_tuple



def create_predict_dict(path):

    image = cv2.imread(path, flags=cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape


    target = {
        "image_id": 0,
        # 原始图片
        "raw_image": image,
        # 加噪后图片
        "noisy_image": image,
        # 设置大小后的 图片
        "resized_image": None,
        # 设置图片大小的缩放尺寸
        "resized_hw": None,
        #  设置mask 后的图片
        #  直线图、直线缩水图、曲线图、曲线缩水图
        "masked_image": None,
        #  图片的宽高
        "raw_image_size": [0, 0, w, h],
        #  标注文件名称
        "ann_path": path,
        #  原始图片路径
        "raw_image_path": path,
        #  实例的序号
        "instance_nm": None,
        #  多边形的点集
        "polygon_points": [],
        "resized_polygon_points": [],
        #  缩水多边形的点集
        "shrink_polygon_points": [],
        "resized_shrink_polygon_points": [],
        #  多边形类型
        "polygon_types": [],
        #  多边形的序号
        "polygon_ids": [],
        #  关键点是否可见
        # "visible": np.array([[2 for j in range(10)] for i in range(len(key_points))], dtype=np.float32),
        #  关键点
        "keypoints": [],
        "resized_keypoints": [],
        #  关键点的权重
        "kps_weights": [],
        #  生成的mask 后的实体
        "entity": [],
        # 实体的 中心点
        "entity_center": [],
        #  热力图
        "heatmap": [],
        #  外接矩形
        "bboxs": [],
        #  最小外接矩形
        "rbboxs": [],
        #  文字标注
        "recs": [],
        # resize 的 M
        "trans": [],
        #  anti resize 的 M
        "reverse_trans": []
    }

    instance = {
        "resized_image": None,
        "masked_image": [],
        "entity": [],
        "heatmap": [],
        "entity_center": [],
        "kps_weights": [],
        "entity_sn": [],
        "visible": [],
        "entity_type": [],
        "entity_id": []
    }
    return instance, target
