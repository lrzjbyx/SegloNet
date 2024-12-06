import datetime
import os.path

import torch.nn as nn
import torchvision.datasets.utils
from torch.nn import functional as F
import  torch
import  numpy as np
import matplotlib.pyplot as plt
import  cv2
from shapely.geometry import Point,Polygon
import pyclipper
import Polygon as plg
from torchvision.transforms import functional as FT
import time
import math
from core.dataset import transforms
from core.loss.loss import KpLoss
from core.models import u2net, hrnet
from core.tools.draw_utils import draw_keypoints
from core.train_utils.calculate import calculate_angle_ann, \
    calculate_angle_diff_ann, calculate_angle_draw


class SealNet(nn.Module):
    def __init__(self,u2net: nn.Module,hrnet: nn.Module,threshold):
        super().__init__()
        self.u2net = u2net
        self.hrnet = hrnet
        # 二值化阈值
        self.threshold = threshold
        # 候选数量
        self.proposal_num = 10
        # 最大扩充步数
        self.expand_step = 10
        self.step_stride = 2
        # 最小iou
        self.miniou = 0.8

        # 膨胀腐蚀kernal
        self.kernel_step = 3
        self.kernel = np.ones((self.kernel_step, self.kernel_step), np.uint8)
        # 针对于矩形的规则处理
        self.epsilon = 0.02
        self.minarea = 200
        self.line_optimize = True
        self.fixed_size = (256,256)
        self.heatmap_hw = (self.fixed_size[0]//4, self.fixed_size[1]//4)
        self.contain_area = 0.03


        self.mse = KpLoss()


    def compute_seg_losses(self,inputs,instances, targets):

        mask_target = instances["masked_image"]
        losses = [F.binary_cross_entropy_with_logits(inputs[i], mask_target) for i in range(len(inputs))]
        total_loss = sum(losses)

        return total_loss

    # def compute_kp_losses(self,inputs,instances,target):
    #     criterion = torch.nn.MSELoss(reduction='none')
    #     assert len(inputs.shape) == 4, 'logits should be 4-ndim'
    #     device = inputs.device
    #     bs = inputs.shape[0]
    #     # [num_kps, H, W] -> [B, num_kps, H, W]
    #     heatmaps = instances["heatmap"]
    #     # [num_kps] -> [B, num_kps]
    #     kps_weights = instances["kps_weights"]
    #
    #     # [B, num_kps, H, W] -> [B, num_kps]
    #     loss = criterion(inputs, heatmaps).mean(dim=[2, 3])
    #     loss = torch.sum(loss * kps_weights) / bs
    #
    #     return loss


    def expand_processing(self,pairs):
        result = []
        raw_result = []
        for pair in pairs:
            shrink_mask_poly = pair[0].buffer(0.01)
            mask_poly = pair[1].buffer(0.01)

            mask_poly_area = mask_poly.area
            # canvas = np.zeros((500, 500, 3), dtype=np.uint8)
            # cv2.fillPoly(canvas, [np.array(mask_poly.exterior.coords,dtype=np.int32)], color=(0, 255, 0))
            # cv2.fillPoly(canvas, [np.array(shrink_mask_poly.exterior.coords,dtype=np.int32)], color=(255, 0, 255))
            # cv2.imwrite("cnavas22.png", canvas)
            try:
                raw_shrink_mask_poly = np.array(shrink_mask_poly.exterior.coords,dtype=np.int32)
            except Exception as e:
                max_index = np.argmax(
                    np.array([cv2.contourArea(np.array(expand_a.exterior.coords).reshape(-1,1,2).astype(np.int32)) for expand_i, expand_a in enumerate(shrink_mask_poly.geoms)]))
                shrink_mask_poly = shrink_mask_poly.geoms[max_index]
                raw_shrink_mask_poly = np.array(shrink_mask_poly.exterior.coords,dtype=np.int32)

            n = 1
            # while not (mask_poly.union(shrink_mask_poly).area - mask_poly_area > mask_poly.area * self.contain_area) and n < self.expand_step:
            while not (mask_poly.union(shrink_mask_poly).area - mask_poly_area > 1) and n < self.expand_step:
                pco = pyclipper.PyclipperOffset()
                shrink_mask_poly = np.array(shrink_mask_poly.exterior.coords)
                pco.AddPath(shrink_mask_poly.reshape(-1, 2), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)

                offset = self.step_stride * n
                expand_array =  pco.Execute(offset)

                try:
                    shrink_mask_poly = Polygon(np.array(expand_array,dtype = object).reshape(-1, 2)).buffer(0.01)
                except Exception as e:
                    max_index = np.argmax(np.array(
                        [cv2.contourArea(np.array(expand_a).reshape(-1,1,2).astype(np.int32)) for expand_i, expand_a in enumerate(expand_array)]))
                    expand_array = expand_array[max_index]
                    shrink_mask_poly = Polygon(np.array(expand_array,dtype = object).reshape(-1, 2)).buffer(0.01)

                #
                # if len(expand_array) >1:
                #     max_index = np.argmax(np.array([cv2.contourArea(np.array(expand_a)) for expand_i,expand_a in enumerate(expand_array)]))
                #     expand_array = expand_array[max_index]
                #     # for expand_i,expand_a in enumerate(expand_array):
                #     #     canvas2 = np.zeros((1000, 1000, 3), dtype=np.uint8)
                #     #     cv2.fillPoly(canvas2, [np.array(expand_a, dtype=np.int32)], color=(0, 255,255))
                #     #     cv2.imwrite("fillPoly{0}.png".format(expand_i), canvas2)
                #
                # shrink_mask_poly = Polygon(np.array(expand_array).reshape(-1, 2)).buffer(0.01)
                n = n + 1

            # expand_mask_poly = np.array(shrink_mask_poly.exterior.coords).reshape(-1, 1, 2).astype(np.int32)
            raw_shrink_mask_poly = raw_shrink_mask_poly.reshape(-1, 1, 2).astype(np.int32)


            ###################
            try:
                expand_mask_poly = np.array(shrink_mask_poly.exterior.coords).reshape(-1, 1, 2).astype(np.int32)
            except Exception as e:
                max_index = np.argmax(np.array(
                    [cv2.contourArea(np.array(expand_a.exterior.coords).reshape(-1,1,2).astype(np.int32)) for expand_i, expand_a in enumerate(shrink_mask_poly.geoms)]))
                expand_mask_poly = shrink_mask_poly.geoms[max_index]
                expand_mask_poly = np.array(expand_mask_poly.exterior.coords).reshape(-1, 1, 2).astype(np.int32)

            try:
                raw_mask_poly = np.array(mask_poly.exterior.coords, dtype=np.int32).reshape(-1, 1, 2)
            except Exception as e:
                max_index = np.argmax(np.array(
                    [cv2.contourArea(np.array(expand_a.exterior.coords).reshape(-1,1,2).astype(np.int32)) for expand_i, expand_a in enumerate(mask_poly.geoms)]))
                raw_mask_poly = mask_poly.geoms[max_index]
                raw_mask_poly = np.array(raw_mask_poly.exterior.coords, dtype=np.int32).reshape(-1, 1, 2)


            # if (shrink_mask_poly.geom_type == 'MultiPolygon'):
            #     # if MultiPolygon, take the smallest convex Polygon containing all the points in the object
            #     expand_mask_poly = np.array(shrink_mask_poly.convex_hull,dtype=np.int32).reshape(-1, 1, 2)
            # else:
            #     expand_mask_poly = np.array(shrink_mask_poly.exterior.coords).reshape(-1, 1, 2).astype(np.int32)

            # if (mask_poly.geom_type == 'MultiPolygon'):
            #     # if MultiPolygon, take the smallest convex Polygon containing all the points in the object
            #     print(mask_poly.convex_hull)
            #     raw_mask_poly = np.array(mask_poly.convex_hull,dtype=np.int32).reshape(-1, 1, 2)
            # else:
            #     raw_mask_poly = np.array(mask_poly.exterior.coords,dtype=np.int32).reshape(-1, 1, 2)
            ####################


            result.append(expand_mask_poly)
            raw_result.append([raw_shrink_mask_poly,raw_mask_poly])

        return result,raw_result


    '''
    mask
    shrink_mask
    target
    sn      1 表示 弧形区域      2 表示矩形区域
    '''
    def pre_processing(self,mask,shrink_mask,target,tn=1):
        # 先二值化
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # shrink_mask = cv2.cvtColor(shrink_mask, cv2.COLOR_BGR2GRAY)

        # 先膨胀后腐蚀
        mask_dilation = cv2.dilate(mask, self.kernel, iterations=1)
        mask_erosion = cv2.erode(mask_dilation, self.kernel, iterations=1)
        shrink_mask_dilation = cv2.dilate(shrink_mask, self.kernel, iterations=1)
        shrink_mask_erosion = cv2.erode(shrink_mask_dilation, self.kernel, iterations=1)


        # 选出候选矩形
        shrink_mask_contours, _ = cv2.findContours(shrink_mask_erosion, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        mask_contours, _ = cv2.findContours(mask_erosion, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # 当候选区域过多  则先不进行扩充区域
        if len(shrink_mask_contours) > self.proposal_num:
            return []

        # canv = np.zeros((1000,1000),np.uint8)
        # canv2 = canv.copy()
        # 剔除小样本
        shrink_mask_proposals = []
        for contour in shrink_mask_contours:

            # cv2.fillPoly(canv,[contour],255)
            if tn == 2:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                contour = approx.reshape((-1,1, 2))

            # cv2.fillPoly(canv2, [contour], 255)
            #
            # cv2.imwrite("canv.png",canv)
            # cv2.imwrite("canv2.png",canv2)




            if cv2.contourArea(contour) >self.minarea and tn == 2:
                shrink_mask_proposals.append(Polygon(contour.reshape((-1,2))))

            if cv2.contourArea(contour)> self.minarea*2 and tn ==1:
                shrink_mask_proposals.append(Polygon(contour.reshape((-1, 2))))


        mask_proposals = []
        for contour in mask_contours:
            if tn == 2:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                contour = approx.reshape((-1,1, 2))

            if cv2.contourArea(contour) >self.minarea:
                mask_proposals.append(Polygon(contour.reshape((-1,2))))

        mask_pairs = []
        for shrink_mask_proposal in shrink_mask_proposals:
            for mask_proposal in mask_proposals:
                if not shrink_mask_proposal.is_valid:
                    shrink_mask_proposal = shrink_mask_proposal.buffer(0.01)
                if not mask_proposal.is_valid:
                    mask_proposal = mask_proposal.buffer(0.01)

                #
                # canvas = np.zeros((500, 500, 3), dtype=np.uint8)
                # cv2.fillPoly(canvas, [np.array(shrink_mask_proposal.exterior.coords, dtype=np.int32)], color=(0, 255, 0))
                # cv2.fillPoly(canvas, [np.array(mask_proposal.exterior.coords, dtype=np.int32)], color=(255, 0, 255))
                # cv2.imwrite("cnavas66.png", canvas)
                if mask_proposal.union(shrink_mask_proposal).area - mask_proposal.area < mask_proposal.area * self.contain_area:
                    mask_pairs.append([shrink_mask_proposal, mask_proposal])


        return mask_pairs



    '''
    x1      原始
    x2      预测
    '''
    def iou(self,x1,x2):
        x1 = Polygon(x1.reshape(-1, 2)).buffer(0.01)
        x2 = Polygon(x2.reshape(-1, 2)).buffer(0.01)

        union_area = x1.union(x2).area
        intersection_area = x1.intersection(x2).area

        iou = intersection_area/union_area

        return iou




    # 分配候选框
    '''
    pred_images     预测的图片
    targets         
    '''
    def assign_proposal(self,pred_images,instances,targets):
        device = instances['resized_image'].device
        preds = pred_images.to("cpu").detach().numpy()
        preds = np.where(preds > self.threshold, 255, 0)
        result = []
        result_ids = []
        result_types = []
        entity_polygons = []
        entity_shrink_polygons = []


        for pi,pred in enumerate(preds):
            line_mask = pred[0].astype(np.uint8)
            line_shrink_mask = pred[1].astype(np.uint8)
            oval_mask = pred[2].astype(np.uint8)
            oval_shrink_mask = pred[3].astype(np.uint8)

            # print("==============")
            # print(oval_shrink_mask.shape)
            # print("==============")

            cv2.imwrite("oval_shrink_mask.png",oval_shrink_mask)
            cv2.imwrite("oval_mask.png",oval_mask)
            cv2.imwrite("line_mask.png",line_mask)
            cv2.imwrite("line_shrink_mask.png",line_shrink_mask)


            oval_pairs = self.pre_processing(oval_mask, oval_shrink_mask, targets, 1)
            line_pairs = self.pre_processing(line_mask, line_shrink_mask, targets, 2)

            # 逐步扩展
            oval_masks,oval_raw_poly = self.expand_processing(oval_pairs)
            line_masks,line_raw_poly = self.expand_processing(line_pairs)

            if self.line_optimize:
                line_masks = [cv2.boxPoints(cv2.minAreaRect(line_mask)).reshape(-1,1,2).astype(np.int32) for line_mask in line_masks]


            canvas = np.zeros((targets[pi]['resized_hw'][0],targets[pi]['resized_hw'][1],3),dtype=np.uint8)
            for line_index, line_mask in enumerate(line_masks):
                entitys_canva = canvas.copy()
                # cv2.fillPoly(canvas, [line_mask], color=(255, 255, 0))
                result.append(FT.to_tensor(self.generate_entity(entitys_canva, line_mask)))
                result_ids.append(pi)
                result_types.append(2)
                entity_polygons.append(line_mask)
                # targets[pi]["resized_polygon_points"].append(line_raw_poly[line_index][1])
                # targets[pi]["resized_shrink_polygon_points"].append(line_raw_poly[line_index][0])


            for oval_index,oval_mask in enumerate(oval_masks):
                entitys_canva = canvas.copy()
                # cv2.fillPoly(canvas, [oval_mask], color=(255, 0, 255))
                result.append(FT.to_tensor(self.generate_entity(entitys_canva, oval_mask)))
                result_ids.append(pi)
                result_types.append(5)
                entity_polygons.append(oval_mask)
                # targets[pi]["resized_polygon_points"].append(oval_raw_poly[oval_index][1])
                # targets[pi]["resized_shrink_polygon_points"].append(oval_raw_poly[oval_index][0])


        # 将其分配到训练集合上


        retrains = []

        # 匹配到的多边形
        for r_in,result_id in enumerate(result_ids):
            # result[r_in]
            entity_centers = targets[result_id]['entity_center']
            for e_in,entity_center in enumerate(entity_centers):
                # 判断中心点是否在多边形区域内
                ic = cv2.pointPolygonTest(entity_polygons[r_in], entity_center, False)
                if ic > 0 :
                    poly_points = targets[result_id]['resized_polygon_points'][e_in]
                    if self.iou(entity_polygons[r_in],poly_points) > self.miniou:
                        # or_image = result[r_in]
                        selection = instances["entity_sn"] == result_id

                        item = {}
                        item["entity"] = FT.normalize(result[r_in], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        item["heatmap"] = instances["heatmap"][selection][e_in].clone()
                        item["entity_center"] = instances["entity_center"][selection][e_in].clone()
                        item["kps_weights"] = instances["kps_weights"][selection][e_in].clone()
                        item["visible"] = instances["visible"][selection][e_in].clone()
                        item["entity_sn"] = instances['entity_sn'][selection][e_in].clone()
                        item["entity_type"] = instances['entity_type'][selection][e_in].clone()
                        item["entity_id"] = instances['entity_id'][selection][e_in].clone()

                        # item["entitys"] = FT.to_tensor(self.generate_entity(or_image,result[r_in]))


                        retrains.append(item)

        if len(retrains) == 0:
            return instances,targets


        entity = torch.stack([rt["entity"] for rt in retrains], dim=0).to(device)
        heatmap = torch.stack([rt["heatmap"] for rt in retrains], dim=0).to(device)
        entity_center = torch.stack([rt["entity_center"] for rt in retrains], dim=0).to(device)
        kps_weights = torch.stack([rt["kps_weights"] for rt in retrains], dim=0).to(device)
        visible = torch.stack([rt["visible"] for rt in retrains], dim=0).to(device)
        entity_sn = torch.stack([rt["entity_sn"] for rt in retrains], dim=0).to(device)
        entity_type = torch.stack([rt["entity_type"] for rt in retrains], dim=0).to(device)
        entity_id = torch.stack([rt["entity_id"] for rt in retrains], dim=0).to(device)

        instances["entity"] = torch.concat((instances["entity"],entity),dim=0)
        instances["heatmap"] = torch.concat((instances["heatmap"],heatmap),dim=0)
        instances["entity_center"] = torch.concat((instances["entity_center"],entity_center),dim=0)
        instances["kps_weights"] = torch.concat((instances["kps_weights"],kps_weights),dim=0)
        instances["visible"] = torch.concat((instances["visible"],visible),dim=0)
        instances["entity_sn"] = torch.concat((instances["entity_sn"],entity_sn),dim=0)
        instances["entity_type"] = torch.concat((instances["entity_type"],entity_type),dim=0)
        instances["entity_id"] = torch.concat((instances["entity_id"],entity_id),dim=0)
        # torch.cat()

        return instances,targets


    # 阶梯透明色
    def generate_entity(self,image,pts):

        canvas = np.zeros(image.shape[:2],dtype=np.uint8)
        cv2.fillPoly(canvas, [pts], color=(255, 255, 255))
        # cv2.imwrite("dwdwdwdwdwd.png",image)
        image = cv2.add(image, np.zeros(np.shape(image), dtype=np.uint8), mask=canvas)
        # cv2.imwrite("dwdwsdwddw{0}.png".format(str(time.time())),image)
        # image = cv2.resize(image,dsize=self.fixed_size)
        ###########################################
        ###########################################
        image = cv2.merge([canvas,canvas,canvas])
        ###########################################
        ###########################################
        cv2.imwrite("dwadsdawdadw.png",image)
        return image

    '''
        测试阶段的分配
    '''
    def assign_proposal_test(self,pred_images,instances,targets):
        device = pred_images.device
        preds = pred_images.to("cpu").detach().numpy()
        preds = np.where(preds > self.threshold, 255, 0)
        result = []
        result_sns = []
        result_ids = []
        result_types = []
        for pi, pred in enumerate(preds):
            line_mask = pred[0].astype(np.uint8)
            line_shrink_mask = pred[1].astype(np.uint8)
            oval_mask = pred[2].astype(np.uint8)
            oval_shrink_mask = pred[3].astype(np.uint8)

            cv2.imwrite("line_mask.png",line_mask)
            cv2.imwrite("line_shrink_mask.png",line_shrink_mask)
            cv2.imwrite("oval_mask.png",oval_mask)
            cv2.imwrite("oval_shrink_mask.png",oval_shrink_mask)

            oval_pairs = self.pre_processing(oval_mask, oval_shrink_mask, targets, 1)
            line_pairs = self.pre_processing(line_mask, line_shrink_mask, targets, 2)


            # 逐步扩展
            oval_masks,oval_raw_poly = self.expand_processing(oval_pairs)
            line_masks,line_raw_poly = self.expand_processing(line_pairs)



            if self.line_optimize:
                line_masks = [cv2.boxPoints(cv2.minAreaRect(line_mask)).reshape(-1,1,2).astype(np.int32) for line_mask in line_masks]


            or_image = targets[pi]['resized_image']
            canvas = or_image.copy()
            id = 0
            for line_index,line_mask in enumerate(line_masks):
                entitys_canva = or_image.copy()
                # cv2.fillPoly(canvas, [line_mask], color=(255, 255, 0))
                cv2.polylines(canvas, [line_mask.astype(np.int32)], True, (0, 255, 0), thickness=2)
                result.append(FT.to_tensor(self.generate_entity(entitys_canva, line_mask)))
                result_sns.append(pi)
                result_ids.append(id)
                result_types.append(torch.as_tensor(2, dtype=torch.float32))
                # targets[pi]["resized_polygon_points"].append(line_raw_poly[line_index][1])
                targets[pi]["resized_polygon_points"].append(line_mask)
                targets[pi]["resized_shrink_polygon_points"].append(line_raw_poly[line_index][0])
                id+=1



            for oval_index,oval_mask in enumerate(oval_masks):
                entitys_canva = or_image.copy()
                # cv2.fillPoly(canvas, [oval_mask], color=(255, 0, 255))
                cv2.polylines(canvas, [oval_mask.astype(np.int32)], True, (0, 255, 0), thickness=2)
                result.append(FT.to_tensor(self.generate_entity(entitys_canva, oval_mask)))
                result_sns.append(pi)
                result_ids.append(id)
                result_types.append(torch.as_tensor(5, dtype=torch.float32))
                # targets[pi]["resized_polygon_points"].append(oval_raw_poly[oval_index][1])
                targets[pi]["resized_polygon_points"].append(oval_mask)
                targets[pi]["resized_shrink_polygon_points"].append(oval_raw_poly[oval_index][0])
                id += 1

            cv2.imwrite("error/{0}-{1}.png".format(os.path.basename(targets[0]["raw_image_path"]).split(".")[0],str(time.time())),canvas)
            # cv2.imwrite("ttttt/{0}.png".format(str(time.time())),canvas)


            if len(result) == 0:
                return instances,targets




        entity = torch.stack(result, dim=0).to(device)
        sn = torch.as_tensor(result_sns,dtype=torch.int64)
        instances["entity_sn"] = sn
        instances["entity"] = entity
        instances["entity"] = FT.normalize(instances["entity"], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        instances["entity_type"] = torch.as_tensor(result_types,dtype=torch.int64)
        instances["entity_id"] = torch.as_tensor(result_ids,dtype=torch.int64)

        return instances,targets




    def forward(self, instances , targets):


        if self.training:
            # 分割网络
            out_images, pred_images = self.u2net(instances['resized_image'])
            # 候选区域
            instances, targets = self.assign_proposal(pred_images, instances, targets)

            # 特征金字塔网络
            out_instances = self.hrnet(instances["entity"])

            seg_losses = self.compute_seg_losses(out_images,instances, targets)
            keys_losses = self.mse(out_instances, instances,targets)

            losses = {
                "seg_losses":seg_losses,
                "keys_losses":keys_losses
            }

            return losses

        else:
            # 分割网络
            out_images, pred_images = self.u2net(instances["resized_image"])


            if len(instances["entity"]) == 0:
                instances, targets = self.assign_proposal_test(pred_images, instances, targets)


            # function =  torchvision.transforms.ToPILImage()
            # png = function(instances["entitys"][0])
            # png.save("2222.png")
            #
            # png = function(instances["entitys"][1])
            # png.save("22221111.png")
            if len(instances["entity"]) == 0:
                pred_images = pred_images.to("cpu").detach().numpy()
                pred_images = np.where(pred_images > self.threshold, 255, 0).astype(np.uint8)

                predict_image = []
                batch_size = pred_images.shape[0]
                for b in range(batch_size):
                    reverse_trans = targets[b]["reverse_trans"]
                    image0 = cv2.warpAffine(pred_images[b][0], reverse_trans, tuple(targets[b]['raw_image_size'][2:4]),
                                            flags=cv2.INTER_LINEAR)
                    image1 = cv2.warpAffine(pred_images[b][1], reverse_trans, tuple(targets[b]['raw_image_size'][2:4]),
                                            flags=cv2.INTER_LINEAR)
                    image2 = cv2.warpAffine(pred_images[b][2], reverse_trans, tuple(targets[b]['raw_image_size'][2:4]),
                                            flags=cv2.INTER_LINEAR)
                    image3 = cv2.warpAffine(pred_images[b][3], reverse_trans, tuple(targets[b]['raw_image_size'][2:4]),
                                            flags=cv2.INTER_LINEAR)
                    # image = cv2.warpAffine(pred_images[b][0],reverse_trans,(200,200),flags=cv2.INTER_LINEAR)
                    predict_image.append(np.stack([image0, image1, image2, image3]))

                return predict_image, ([], [], [],[],[],[])

            # 特征金字塔网络
            out_instances = self.hrnet(instances["entity"])

            instances["entity_sn"].to("cpu").numpy()
            reverse_trans = [targets[t]["reverse_trans"] for t in instances["entity_sn"].to("cpu").numpy()]

            keypoints, scores = transforms.get_final_preds(out_instances, reverse_trans, post_processing=True)

            entity_sn = instances["entity_sn"].to("cpu").numpy()
            entity_type = instances["entity_type"].to("cpu").numpy()

            entity_polygons = []
            entity_shrink_polygons = []
            for t in targets:
                transforms.get_final_polygon(t['resized_polygon_points'],reverse_trans,entity_polygons)
            for t in targets:
                transforms.get_final_polygon(t['resized_shrink_polygon_points'],reverse_trans,entity_shrink_polygons)


            if len(instances["heatmap"]) != 0:
                return pred_images,(keypoints, scores, entity_sn,entity_type,entity_polygons,entity_shrink_polygons)

            # 复原
            pred_images = pred_images.to("cpu").detach().numpy()
            pred_images = np.where(pred_images > self.threshold, 255, 0).astype(np.uint8)

            predict_image = []
            batch_size = pred_images.shape[0]
            for b in range(batch_size):
                reverse_trans = targets[b]["reverse_trans"]
                image0 = cv2.warpAffine(pred_images[b][0],reverse_trans,tuple(targets[b]['raw_image_size'][2:4]),flags=cv2.INTER_LINEAR)
                image1 = cv2.warpAffine(pred_images[b][1],reverse_trans,tuple(targets[b]['raw_image_size'][2:4]),flags=cv2.INTER_LINEAR)
                image2 = cv2.warpAffine(pred_images[b][2],reverse_trans,tuple(targets[b]['raw_image_size'][2:4]),flags=cv2.INTER_LINEAR)
                image3 = cv2.warpAffine(pred_images[b][3],reverse_trans,tuple(targets[b]['raw_image_size'][2:4]),flags=cv2.INTER_LINEAR)
                # image = cv2.warpAffine(pred_images[b][0],reverse_trans,(200,200),flags=cv2.INTER_LINEAR)

                predict_image.append(np.stack([image0,image1,image2,image3]))




            # keypoints = np.squeeze(keypoints)
            # scores = np.squeeze(scores)

            # params1 = cv2.fitEllipseAMS(keypoints.astype(np.int32))
            # print(params1)
            # cv2.ellipse(img, params1, (0, 0, 255), 2)

            # for i,keypoint in enumerate(keypoints):
            #     img = targets[instances['entity_sn'][i]]["raw_image"]
            #     resized_polygon_points = targets[instances['entity_sn'][i]]['resized_polygon_points'][i]
            #     if instances['entity_type'][i] == 5:
            #         params1 = cv2.fitEllipseAMS(keypoint.astype(np.int32))
            #         cv2.ellipse(img, params1, (0, 0, 255), 2)
            #         # print(params1)
            #
            #         # cv2.circle(img,(int(params1[0][0]),int(params1[0][1])),5,(100,100,0),-1)
            #         # 计算向量和x轴的夹角
            #
            #         start_angle = calculate_angle_ann(params1[0], keypoint[0])
            #         end_angle = calculate_angle_ann(params1[0], keypoint[-1])
            #         span_angle =  calculate_angle_diff_ann(end_angle,start_angle)
            #
            #         start_angle = calculate_angle_draw(params1[0], keypoint[0])
            #         end_angle = calculate_angle_draw(params1[0], keypoint[-1])
            #         # span_angle =  calculate_angle_diff_ann(end_angle,start_angle)
            #
            #
            #         cv2.ellipse(img, (int(params1[0][0]),int(params1[0][1])), (int(params1[1][1]/2),int(params1[1][0]/2)), 0, int(start_angle), int(end_angle), (0, 255, 0), -1)
            #
            #         # arc_center = (np.mean(keypoint[4:6], axis=0)[0],np.mean(keypoint[4:6], axis=0)[1])
            #
            #         rotate_angle = params1[2]-90
            #
            #
            #         # cv2.circle(img, (int(keypoint[-1][0]), int(keypoint[-1][1])), 5, (100, 100, 0), -1)
            #         cv2.imwrite("{0}.png".format(time.time()),img)
            #
            #     else:
            #         # 使用 cv2.fitLine() 函数拟合直线
            #         vx, vy, x0, y0 = cv2.fitLine(keypoint.astype(np.int32), cv2.DIST_L2, 0, 0.01, 0.01)
            #         # 斜率
            #         slope = vy / vx
            #
            #         # # 计算直线的两个端点
            #         # lefty = int((-x0 * vy / vx) + y0)
            #         # righty = int(((100 - x0) * vy / vx) + y0)
            #
            #     plot_img = draw_keypoints(img, keypoint, scores[i], thresh=0.2, r=8)
            #     plt.imshow(plot_img)
            #     # plt.show()
            #     now = datetime.datetime.now()
            #     filename = now.strftime("source/%Y-%m-%d_%H-%M-%S-{0}-{1}.jpg").format(i,str(time.time()))
            #     plot_img.save(filename)
            #
            #
            #     # print(params1)
            #     # cv2.ellipse(img, params1, (0, 0, 255), 2)

            # torch.concat()
            # torch.cat()





            return predict_image,(keypoints, scores, entity_sn,entity_type,entity_polygons,entity_shrink_polygons)


def u2_hr_backbone(out_ch=4,base_channel=32,key_joints=10,threshold=0.5,cfg="xx"):


    if cfg == "xx":
        u2net_backbone = u2net.u2net_full(out_ch)
        hrnet_backbone = hrnet.HighResolutionNet(base_channel=base_channel,num_joints=key_joints)
    elif cfg =="sx":
        u2net_backbone = u2net.u2net_lite(out_ch)
        hrnet_backbone = hrnet.HighResolutionNet(base_channel=base_channel,num_joints=key_joints)


    return SealNet(u2net_backbone,hrnet_backbone,threshold)
