import math
import random
from typing import Tuple

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T


def flip_images(img):
    assert len(img.shape) == 4, 'images has to be [batch_size, channels, height, width]'
    img = torch.flip(img, dims=[3])
    return img


def flip_back(output_flipped, matched_parts):
    assert len(output_flipped.shape) == 4, 'output_flipped has to be [batch_size, num_joints, height, width]'
    output_flipped = torch.flip(output_flipped, dims=[3])

    for pair in matched_parts:
        tmp = output_flipped[:, pair[0]].clone()
        output_flipped[:, pair[0]] = output_flipped[:, pair[1]]
        output_flipped[:, pair[1]] = tmp

    return output_flipped


def get_max_preds(batch_heatmaps):
    """
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    """
    assert isinstance(batch_heatmaps, torch.Tensor), 'batch_heatmaps should be torch.Tensor'
    assert len(batch_heatmaps.shape) == 4, 'batch_images should be 4-ndim'

    batch_size, num_joints, h, w = batch_heatmaps.shape
    heatmaps_reshaped = batch_heatmaps.reshape(batch_size, num_joints, -1)
    maxvals, idx = torch.max(heatmaps_reshaped, dim=2)

    maxvals = maxvals.unsqueeze(dim=-1)
    idx = idx.float()

    preds = torch.zeros((batch_size, num_joints, 2)).to(batch_heatmaps)

    preds[:, :, 0] = idx % w  # column 对应最大值的x坐标
    preds[:, :, 1] = torch.floor(idx / w)  # row 对应最大值的y坐标

    pred_mask = torch.gt(maxvals, 0.0).repeat(1, 1, 2).float().to(batch_heatmaps.device)

    preds *= pred_mask
    return preds, maxvals


def affine_points(pt, t):
    ones = np.ones((pt.shape[0], 1), dtype=float)
    pt = np.concatenate([pt, ones], axis=1).T
    new_pt = np.dot(t, pt)
    return new_pt.T


def get_final_preds(batch_heatmaps: torch.Tensor,
                    trans: list = None,
                    post_processing: bool = False):
    assert trans is not None
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if post_processing:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                    diff = torch.tensor(
                        [
                            hm[py][px + 1] - hm[py][px - 1],
                            hm[py + 1][px] - hm[py - 1][px]
                        ]
                    ).to(batch_heatmaps.device)
                    coords[n][p] += torch.sign(diff) * .25

    preds = coords.clone().cpu().numpy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = affine_points(preds[i], trans[i])

    return preds, maxvals.cpu().numpy()




def get_final_polygon(polygons :list = None ,trans: list = None,result:list=None):
    for i,polygon in enumerate(polygons):
        tt = polygon.reshape(-1,2)
        result.append(affine_points(tt, trans[i]).reshape(-1,1,2))

    return result



def decode_keypoints(outputs, origin_hw, num_joints: int = 17):
    keypoints = []
    scores = []
    heatmap_h, heatmap_w = outputs.shape[-2:]
    for i in range(num_joints):
        pt = np.unravel_index(np.argmax(outputs[i]), (heatmap_h, heatmap_w))
        score = outputs[i, pt[0], pt[1]]
        keypoints.append(pt[::-1])  # hw -> wh(xy)
        scores.append(score)

    keypoints = np.array(keypoints, dtype=float)
    scores = np.array(scores, dtype=float)
    # convert to full image scale
    keypoints[:, 0] = np.clip(keypoints[:, 0] / heatmap_w * origin_hw[1],
                              a_min=0,
                              a_max=origin_hw[1])
    keypoints[:, 1] = np.clip(keypoints[:, 1] / heatmap_h * origin_hw[0],
                              a_min=0,
                              a_max=origin_hw[0])
    return keypoints, scores


def resize_pad(img: np.ndarray, size: tuple):
    h, w, c = img.shape
    src = np.array([[0, 0],       # 原坐标系中图像左上角点
                    [w - 1, 0],   # 原坐标系中图像右上角点
                    [0, h - 1]],  # 原坐标系中图像左下角点
                   dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    if h / w > size[0] / size[1]:
        # 需要在w方向padding
        wi = size[0] * (w / h)
        pad_w = (size[1] - wi) / 2
        dst[0, :] = [pad_w - 1, 0]            # 目标坐标系中图像左上角点
        dst[1, :] = [size[1] - pad_w - 1, 0]  # 目标坐标系中图像右上角点
        dst[2, :] = [pad_w - 1, size[0] - 1]  # 目标坐标系中图像左下角点
    else:
        # 需要在h方向padding
        hi = size[1] * (h / w)
        pad_h = (size[0] - hi) / 2
        dst[0, :] = [0, pad_h - 1]            # 目标坐标系中图像左上角点
        dst[1, :] = [size[1] - 1, pad_h - 1]  # 目标坐标系中图像右上角点
        dst[2, :] = [0, size[0] - pad_h - 1]  # 目标坐标系中图像左下角点

    trans = cv2.getAffineTransform(src, dst)  # 计算正向仿射变换矩阵
    # 对图像进行仿射变换
    resize_img = cv2.warpAffine(img,
                                trans,
                                size[::-1],  # w, h
                                flags=cv2.INTER_LINEAR)
    # import matplotlib.pyplot as plt
    # plt.imshow(resize_img)
    # plt.show()

    dst /= 4  # 网络预测的heatmap尺寸是输入图像的1/4
    reverse_trans = cv2.getAffineTransform(dst, src)  # 计算逆向仿射变换矩阵，方便后续还原

    return resize_img, reverse_trans


def adjust_box(xmin: float, ymin: float, w: float, h: float, fixed_size: Tuple[float, float]):
    """通过增加w或者h的方式保证输入图片的长宽比固定"""
    xmax = xmin + w
    ymax = ymin + h

    hw_ratio = fixed_size[0] / fixed_size[1]
    if h / w > hw_ratio:
        # 需要在w方向padding
        wi = h / hw_ratio
        pad_w = (wi - w) / 2
        xmin = xmin - pad_w
        xmax = xmax + pad_w
    else:
        # 需要在h方向padding
        hi = w * hw_ratio
        pad_h = (hi - h) / 2
        ymin = ymin - pad_h
        ymax = ymax + pad_h

    return xmin, ymin, xmax, ymax


def scale_box(xmin: float, ymin: float, w: float, h: float, scale_ratio: Tuple[float, float]):
    """根据传入的h、w缩放因子scale_ratio，重新计算xmin，ymin，w，h"""
    s_h = h * scale_ratio[0]
    s_w = w * scale_ratio[1]
    xmin = xmin - (s_w - w) / 2.
    ymin = ymin - (s_h - h) / 2.
    return xmin, ymin, s_w, s_h


def plot_heatmap(image, heatmap, kps, kps_weights):
    for kp_id in range(len(kps_weights)):
        if kps_weights[kp_id] > 0:
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.plot(*kps[kp_id].tolist(), "ro")
            plt.title("image")
            plt.subplot(1, 2, 2)
            plt.imshow(heatmap[kp_id], cmap=plt.cm.Blues)
            plt.colorbar(ticks=[0, 1])
            plt.title(f"kp_id: {kp_id}")
            plt.show()


class Compose(object):
    """组合多个transform函数"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, instance, target):
        for t in self.transforms:
            instance, target = t(instance, target)
        return instance, target


class ToTensor(object):
    """将PIL图像转为Tensor"""
    def __call__(self, instance, target):
        # cv2.imwrite("1.png",image)
        resized_image = F.to_tensor(instance['resized_image'])
        # 分割
        instance["resized_image"] = resized_image

        if len(instance["masked_image"]) != 0:
            instance["masked_image"] = F.to_tensor(target["masked_image"]).permute(1, 2, 0).contiguous()
            # 关键点检测
            target["entity"] = [F.to_tensor(t) for t in target["entity"]]
            instance["entity"] = torch.stack(target["entity"],dim=0)

            # target["poly_ids"] =  F.to_tensor(target["poly_ids"])
            instance["visible"] =  torch.as_tensor(target["visible"], dtype=torch.float32)
            instance["entity_center"] =  torch.as_tensor(np.array(target["entity_center"]), dtype=torch.float32)
            instance["bboxs"] =  torch.from_numpy(np.stack(target["bboxs"],axis=0))
            instance["entity_type"] = torch.tensor(target['polygon_types'])
            # instance["resized_polygon_points"] = torch.tensor(np.stack(target['resized_polygon_points']))





        return instance, target


class Normalize(object):
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, instance, target):
        instance['resized_image'] = F.normalize(instance['resized_image'] , mean=self.mean, std=self.std)
        if len(instance["masked_image"]) != 0 :
            instance["entity"] = F.normalize(instance["entity"], mean=self.mean, std=self.std)
        return instance, target



class HalfBody(object):
    def __init__(self, p: float = 0.3, upper_body_ids=None, lower_body_ids=None):
        assert upper_body_ids is not None
        assert lower_body_ids is not None
        self.p = p
        self.upper_body_ids = upper_body_ids
        self.lower_body_ids = lower_body_ids

    def __call__(self, image, target):
        if random.random() < self.p:
            kps = target["keypoints"]
            vis = target["visible"]
            upper_kps = []
            lower_kps = []

            # 对可见的keypoints进行归类
            for i, v in enumerate(vis):
                if v > 0.5:
                    if i in self.upper_body_ids:
                        upper_kps.append(kps[i])
                    else:
                        lower_kps.append(kps[i])

            # 50%的概率选择上或下半身
            if random.random() < 0.5:
                selected_kps = upper_kps
            else:
                selected_kps = lower_kps

            # 如果点数太少就不做任何处理
            if len(selected_kps) > 2:
                selected_kps = np.array(selected_kps, dtype=np.float32)
                xmin, ymin = np.min(selected_kps, axis=0).tolist()
                xmax, ymax = np.max(selected_kps, axis=0).tolist()
                w = xmax - xmin
                h = ymax - ymin
                if w > 1 and h > 1:
                    # 把w和h适当放大点，要不然关键点处于边缘位置
                    xmin, ymin, w, h = scale_box(xmin, ymin, w, h, (1.5, 1.5))
                    target["box"] = [xmin, ymin, w, h]

        return image, target


class AffineTransform(object):
    """scale+rotation"""
    def __init__(self,
                 scale: Tuple[float, float] = None,  # e.g. (0.65, 1.35)
                 rotation: Tuple[int, int] = None,   # e.g. (-45, 45)
                 fixed_size: Tuple[int, int] = (256, 256)):
        self.scale = scale
        self.rotation = rotation
        self.fixed_size = fixed_size

    def __call__(self, instance, target):
        target["resized_hw"] = [self.fixed_size[0],self.fixed_size[1]]
        borderValue = target["raw_image"][0][0]
        src_xmin, src_ymin, src_xmax, src_ymax = adjust_box(*target["raw_image_size"], fixed_size=self.fixed_size)
        src_w = src_xmax - src_xmin
        src_h = src_ymax - src_ymin
        src_center = np.array([(src_xmin + src_xmax) / 2, (src_ymin + src_ymax) / 2])
        src_p2 = src_center + np.array([0, -src_h / 2])  # top middle
        src_p3 = src_center + np.array([src_w / 2, 0])   # right middle

        dst_center = np.array([(self.fixed_size[1] - 1) / 2, (self.fixed_size[0] - 1) / 2])
        dst_p2 = np.array([(self.fixed_size[1] - 1) / 2, 0])  # top middle
        dst_p3 = np.array([self.fixed_size[1] - 1, (self.fixed_size[0] - 1) / 2])  # right middle

        if self.scale is not None:
            scale = random.uniform(*self.scale)
            src_w = src_w * scale
            src_h = src_h * scale
            src_p2 = src_center + np.array([0, -src_h / 2])  # top middle
            src_p3 = src_center + np.array([src_w / 2, 0])   # right middle

        if self.rotation is not None:
            angle = random.randint(*self.rotation)  # 角度制
            angle = angle / 180 * math.pi  # 弧度制
            src_p2 = src_center + np.array([src_h / 2 * math.sin(angle), -src_h / 2 * math.cos(angle)])
            src_p3 = src_center + np.array([src_w / 2 * math.cos(angle), src_w / 2 * math.sin(angle)])

        src = np.stack([src_center, src_p2, src_p3]).astype(np.float32)
        dst = np.stack([dst_center, dst_p2, dst_p3]).astype(np.float32)

        trans = cv2.getAffineTransform(src, dst)  # 计算正向仿射变换矩阵
        # dst /= 4  # 网络预测的heatmap尺寸是输入图像的1/4
        reverse_trans = cv2.getAffineTransform(dst, src)  # 计算逆向仿射变换矩阵，方便后续还原



        if len(target["polygon_types"]) != 0:
            '''
                "shrink_poly_points":shrink_poly_points,
                "poly_points":poly_points,
                "poly_ids":poly_ids,
            '''
            line_shrink_mask = np.zeros(target["noisy_image"].shape[:2], dtype=np.uint8)
            line_mask = line_shrink_mask.copy()
            oval_shrink_mask = line_shrink_mask.copy()
            oval_mask = line_shrink_mask.copy()

            for polygon_index,polygon_type in enumerate(target["polygon_types"]):
                # 2 是直线
                if polygon_type == 2:
                    cv2.fillPoly(line_shrink_mask, [target["shrink_polygon_points"][polygon_index]], color=(255, 255, 255))
                    cv2.fillPoly(line_mask, [target["polygon_points"][polygon_index]], color=(255, 255, 255))
                elif polygon_type == 5:
                    cv2.fillPoly(oval_shrink_mask, [target["shrink_polygon_points"][polygon_index]], color=(255, 255, 255))
                    cv2.fillPoly(oval_mask, [target["polygon_points"][polygon_index]], color=(255, 255, 255))
                else:
                    pass


            # 对每个 mask 进行变换
            line_mask = cv2.warpAffine(line_mask,trans,tuple(self.fixed_size[::-1]),flags=cv2.INTER_LINEAR)
            line_shrink_mask = cv2.warpAffine(line_shrink_mask,trans,tuple(self.fixed_size[::-1]),flags=cv2.INTER_LINEAR)
            oval_mask = cv2.warpAffine(oval_mask,trans,tuple(self.fixed_size[::-1]),flags=cv2.INTER_LINEAR)
            oval_shrink_mask = cv2.warpAffine(oval_shrink_mask,trans,tuple(self.fixed_size[::-1]),flags=cv2.INTER_LINEAR)

            #直线图、直线缩水图、曲线图、曲线缩水图
            target["masked_image"] = np.stack([line_mask,line_shrink_mask,oval_mask,oval_shrink_mask],axis=0)
            instance["masked_image"] = target["masked_image"].copy()

            # cv2.imwrite("line_mask.png",line_mask)
            # cv2.imwrite("line_shrink_mask.png",line_shrink_mask)
            # cv2.imwrite("oval_mask.png",oval_mask)
            # cv2.imwrite("oval_shrink_mask.png",oval_shrink_mask)


        # 对图像进行仿射变换
        resized_image = cv2.warpAffine(target["noisy_image"],
                                    trans,
                                    tuple(self.fixed_size[::-1]),  # [w, h]
                                    flags=cv2.INTER_LINEAR,borderValue=borderValue.tolist())

        target["resized_image"] = resized_image
        instance["resized_image"] = resized_image.copy()


        if len(target["polygon_types"]) != 0:
            # 多边形点的旋转
            for polygon_index,polygon_ids in enumerate(target["polygon_ids"]):
                kps = target["shrink_polygon_points"][polygon_index][0]
                mask = np.logical_and(kps[:, 0] != 0, kps[:, 1] != 0)
                kps[mask] = affine_points(kps[mask], trans)
                target["resized_shrink_polygon_points"].append(kps)

                kps = target["polygon_points"][polygon_index].reshape(200,2)
                mask = np.logical_and(kps[:, 0] != 0, kps[:, 1] != 0)
                kps[mask] = affine_points(kps[mask], trans)
                target["resized_polygon_points"].append(kps)



            # 获取旋转后的关键点
            for polygon_index,polygon_ids in enumerate(target["polygon_ids"]):
                kps = target["keypoints"][polygon_index].reshape(10,2)
                mask = np.logical_and(kps[:, 0] != 0, kps[:, 1] != 0)
                kps[mask] = affine_points(kps[mask], trans)
                target["resized_keypoints"].append(kps)

            # 对 文本实例进行同等旋转
            # 先绘制各个实例
            entity = np.zeros(self.fixed_size,dtype=np.uint8)
            for polygon_index,polygon_ids in enumerate(target["polygon_ids"]):
                entity_canvas = entity.copy()
                cv2.fillPoly(entity_canvas, [target["polygon_points"][polygon_index]], color=(255, 255, 255))
                # entity_canvas = cv2.add(resized_image, np.zeros(np.shape(resized_image), dtype=np.uint8), mask=entity_canvas)
                ######################################
                ######################################
                ######################################
                entity_canvas = cv2.merge([entity_canvas,entity_canvas,entity_canvas])
                # entity_canvas = np.stack([entity_canvas,entity_canvas,entity_canvas])
                ######################################
                ######################################
                ######################################
                # cv2.imwrite("entity-{0}.png".format(polygon_index),entity_canvas)
                target["entity"].append(entity_canvas)
                instance["entity"].append(entity_canvas.copy())
                cx = np.mean([target["polygon_points"][polygon_index].reshape(-1,2)[49:50,0] , target["polygon_points"][polygon_index].reshape(-1,2)[149:150,0]])
                cy = np.mean([target["polygon_points"][polygon_index].reshape(-1,2)[49:50,1] , target["polygon_points"][polygon_index].reshape(-1,2)[149:150,1]])

                # cv2.circle(resized_image,(int(cx),int(cy)),5,(255,0,0),-1)
                #
                # cv2.imwrite("ddwadsda.png",resized_image)


                target["entity_center"].append(np.array([cx,cy]))
                instance["entity_center"].append(np.array([cx,cy]))


        # import matplotlib.pyplot as plt
        # from draw_utils import draw_keypoints
        # # resize_img = draw_keypoints(resize_img.copy(), target["keypoints"])
        # draw_keypoints(resize_img, target["keypoints"])
        # plt.imshow(resize_img)
        # plt.show()

        target["trans"] = trans
        target["reverse_trans"] = reverse_trans
        return instance, target


class RandomHorizontalFlip(object):
    """随机对输入图片进行水平翻转，注意该方法必须接在 AffineTransform 后"""
    def __init__(self, p: float = 0.5, matched_parts: list = None):
        assert matched_parts is not None
        self.p = p
        self.matched_parts = matched_parts

    def __call__(self, image, target):
        if random.random() < self.p:
            # [h, w, c]
            image = np.ascontiguousarray(np.flip(image, axis=[1]))
            keypoints = target["keypoints"]
            visible = target["visible"]
            width = image.shape[1]

            # Flip horizontal
            keypoints[:, 0] = width - keypoints[:, 0] - 1

            # Change left-right parts
            for pair in self.matched_parts:
                keypoints[pair[0], :], keypoints[pair[1], :] = \
                    keypoints[pair[1], :], keypoints[pair[0], :].copy()

                visible[pair[0]], visible[pair[1]] = \
                    visible[pair[1]], visible[pair[0]].copy()

            target["keypoints"] = keypoints
            target["visible"] = visible

        return image, target


class KeypointToHeatMap(object):
    def __init__(self,
                 heatmap_hw: Tuple[int, int] = (256, 256),
                 gaussian_sigma: int = 4,
                 keypoints_weights=None):
        self.heatmap_hw = heatmap_hw
        self.sigma = gaussian_sigma
        self.kernel_radius = self.sigma * 3
        self.use_kps_weights = False if keypoints_weights is None else True
        self.kps_weights = keypoints_weights

        # generate gaussian kernel(not normalized)
        kernel_size = 2 * self.kernel_radius + 1
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        x_center = y_center = kernel_size // 2
        for x in range(kernel_size):
            for y in range(kernel_size):
                kernel[y, x] = np.exp(-((x - x_center) ** 2 + (y - y_center) ** 2) / (2 * self.sigma ** 2))
        # print(kernel)

        self.kernel = kernel

    def __call__(self, instance, target):
        for polygon_index,polygon_ids in enumerate(target["polygon_ids"]):
            kps = target["keypoints"][polygon_index].reshape(-1,2)
            num_kps = kps.shape[0]
            kps_weights = np.ones((num_kps,), dtype=np.float32)
            if "visible" in target:
                visible = target["visible"][polygon_index]
                kps_weights = visible

            heatmap = np.zeros((num_kps, self.heatmap_hw[0], self.heatmap_hw[1]), dtype=np.float32)
            # heatmap_kps = (kps / 4 + 0.5).astype(np.int)  # round
            heatmap_kps = (kps).astype(np.int)  # round
            for kp_id in range(num_kps):
                v = kps_weights[kp_id]
                if v < 0.5:
                    # 如果该点的可见度很低，则直接忽略
                    continue

                x, y = heatmap_kps[kp_id]
                ul = [x - self.kernel_radius, y - self.kernel_radius]  # up-left x,y
                br = [x + self.kernel_radius, y + self.kernel_radius]  # bottom-right x,y
                # 如果以xy为中心kernel_radius为半径的辐射范围内与heatmap没交集，则忽略该点(该规则并不严格)
                if ul[0] > self.heatmap_hw[1] - 1 or \
                        ul[1] > self.heatmap_hw[0] - 1 or \
                        br[0] < 0 or \
                        br[1] < 0:
                    # If not, just return the image as is
                    kps_weights[kp_id] = 0
                    continue

                # Usable gaussian range
                # 计算高斯核有效区域（高斯核坐标系）
                g_x = (max(0, -ul[0]), min(br[0], self.heatmap_hw[1] - 1) - ul[0])
                g_y = (max(0, -ul[1]), min(br[1], self.heatmap_hw[0] - 1) - ul[1])
                # image range
                # 计算heatmap中的有效区域（heatmap坐标系）
                img_x = (max(0, ul[0]), min(br[0], self.heatmap_hw[1] - 1))
                img_y = (max(0, ul[1]), min(br[1], self.heatmap_hw[0] - 1))

                if kps_weights[kp_id] > 0.5:
                    # 将高斯核有效区域复制到heatmap对应区域
                    heatmap[kp_id][img_y[0]:img_y[1] + 1, img_x[0]:img_x[1] + 1] = \
                        self.kernel[g_y[0]:g_y[1] + 1, g_x[0]:g_x[1] + 1]

            if self.use_kps_weights:
                kps_weights = np.multiply(kps_weights, self.kps_weights)

            # plot_heatmap(target["entity"][polygon_index], heatmap, kps, kps_weights)

            target["kps_weights"].append(kps_weights)
            target["heatmap"].append(heatmap)
            instance["heatmap"].append(torch.as_tensor(heatmap, dtype=torch.float32))
            instance["kps_weights"].append(torch.as_tensor(kps_weights, dtype=torch.float32))


        instance["visible"] = np.stack(instance["visible"],axis=0)
        instance["heatmap"] = torch.stack(instance["heatmap"],dim=0)
        instance["kps_weights"] = torch.stack(instance["kps_weights"],dim=0)
        return instance, target


'''
集成 
    灰色化
    颜色抖动
    随机对图像进行色调分离
    随机均衡给定图像的直方图
'''
class ColorAugmentation(object):
    def __call__(self, instance, target):
        nrr =np.random.randint(0,len(self.transforms)+5)
        oo = target['noisy_image']
        if nrr < len(self.transforms):
            noisy_image = np.array(self.transforms[nrr](Image.fromarray(target['noisy_image'])))
            if nrr == 0:
                noisy_image = cv2.merge([noisy_image,noisy_image,noisy_image])

            # plot([image],oo)
            target['noisy_image'] = noisy_image

        return instance, target


    def __init__(self,
                 grayscale=True,
                 colorjitter=True,
                 posterizer = True,
                 equalizes = True,
                 sharpness= True,
                 brightness=.5,
                 hue=.3,
                 bits =2,
                 sharpness_factor = 2

                 ):
        self.grayscale = grayscale
        self.colorjitter = colorjitter
        self.posterizer = posterizer
        self.equalizes = equalizes
        self.sharpness = sharpness


        self.transforms = []
        if self.grayscale:
            self.transforms.append(T.Grayscale())
        if self.colorjitter:
            self.transforms.append(T.ColorJitter(brightness=brightness, hue=hue))
        if self.posterizer:
            self.transforms.append( T.RandomPosterize(bits=bits))
        if self.equalizes:
            self.transforms.append(T.RandomEqualize())
        if self.sharpness:
            self.transforms.append(T.RandomAdjustSharpness(sharpness_factor=sharpness_factor))



#
# augmenter = T.AugMix()
# imgs = [augmenter(orig_img) for _ in range(4)]
# plot(imgs)
class BlurAugmentation(object):
    def __init__(self, blurrer = True,kernel_size=(5, 9),sigma=(0.1, 5)):

        # self.augmenter = augmenter
        self.blurrer = blurrer
        self.transforms = []
        # if self.augmenter:
        #     self.transforms.append(T.AugMix())

        if self.blurrer:
            self.transforms.append(T.GaussianBlur(kernel_size=kernel_size, sigma=sigma))


    def __call__(self, instance,target):
        nrr = np.random.randint(0, len(self.transforms)+5)
        # oo = image
        if nrr < len(self.transforms):
            target['noisy_image'] = np.array(self.transforms[nrr](Image.fromarray(target['noisy_image'])))

        return instance, target






