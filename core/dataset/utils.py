import json

import cv2
import  numpy as np
import pyclipper
import Polygon as plg
import math

def readJsonFile(path):
    with open(path, 'r',encoding='utf-8') as f:
        data = json.load(f)

    return data


def shrink(bboxes, min_shr):
    shrinked_bboxes = []
    for bbox in bboxes:
        area = plg.Polygon(bbox).area()
        # peri = perimeter(bbox)

        try:
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            offset = int(-min_shr)

            shrinked_bbox = pco.Execute(offset)
            if len(shrinked_bbox) == 0:
                shrinked_bboxes.append(bbox)
                continue

            shrinked_bbox = np.array(shrinked_bbox[0])
            if shrinked_bbox.shape[0] <= 2:
                shrinked_bboxes.append(bbox)
                continue

            shrinked_bboxes.append(shrinked_bbox)
        except Exception as e:
            print(type(shrinked_bbox), shrinked_bbox)
            # print('area:', area, 'peri:', peri)
            shrinked_bboxes.append(bbox)

    return shrinked_bboxes

################################################
def circle(x0,y0,r,t):
    x = x0 + r*np.cos(t)
    y = y0 - r*np.sin(t)
    return (x,y)
def circle_convert_polygon(x0,y0,r,t1,t2,h,rotation,num_control_points):
    if t1 == 5760:
        t1 = 0

    if t1 >180*16:
        t1 = t1 - 360*16

    t1 = math.radians(t1 / 16)-math.radians(rotation)
    t2 = t1 + math.radians(t2 / 16)
    spans = np.linspace(t1, t2, int(num_control_points/2))[::-1]
    top_polygon = []
    bottom_polygon = []
    for span in spans:
        (x,y) = circle(x0,y0,r+h/2,span)
        top_polygon.append(np.array([x,y]))

    for span in spans:
        (x,y) = circle(x0,y0,r-h/2,span)
        bottom_polygon.append(np.array([x, y]))


    top_polygon = np.array(top_polygon)
    bottom_polygon = np.array(bottom_polygon)[::-1]

    return np.concatenate((top_polygon,bottom_polygon),axis=0)
def circle_convert_key_points(x0,y0,r,t1,t2,rotation,num_control_points):
    if t1 == 5760:
        t1 = 0

    if t1 >180*16:
        t1 = t1 - 360*16

    t1 = math.radians(t1 / 16)-math.radians(rotation)
    t2 = t1 + math.radians(t2 / 16)
    spans = np.linspace(t1, t2, int(num_control_points))[::-1]
    top_polygon = []

    for span in spans:
        (x,y) = circle(x0,y0,r,span)
        top_polygon.append(np.array([x,y]))

    top_polygon = np.array(top_polygon)

    return top_polygon

################################################
def line(x0, y0, t, h):
    x1 = x0 + h * np.cos(math.radians(t))
    y1 = y0 + h * np.sin(math.radians(t))
    return (x1, y1)
def line_convert_polygon(x0, y0, t, h, l,num_control_points):
    spans = np.linspace(-l / 2, l / 2, int(num_control_points/2))
    top_polygon = []
    bottom_polygon = []
    tx = x0 - h / 2 * np.cos(math.pi / 2 - math.radians(t))
    ty = y0 + h / 2 * np.sin(math.pi / 2 - math.radians(t))
    for span in spans:
        (x, y) = line(tx, ty, t, span)
        top_polygon.append(np.array([x,y]))
    bx = x0 + h / 2 * np.cos(math.pi / 2 - math.radians(t))
    by = y0 - h / 2 * np.sin(math.pi / 2 - math.radians(t))
    for span in spans:
        (x, y) = line(bx, by, t, span)
        bottom_polygon.append(np.array([x, y]))

    top_polygon = np.array(top_polygon)
    # bottom_polygon = np.array(bottom_polygon)
    bottom_polygon = np.array(bottom_polygon)[::-1]

    return np.concatenate((top_polygon, bottom_polygon), axis=0)
def line_convert_key_points(x0, y0, t, l,num_control_points):
    spans = np.linspace(-l / 2, l / 2, int(num_control_points))
    top_polygon = []

    tx = x0
    ty = y0
    for span in spans:
        (x, y) = line(tx, ty, t, span)
        top_polygon.append(np.array([x,y]))


    top_polygon = np.array(top_polygon)


    return top_polygon

################################################
def oval(h, k, a, t, c, b):
    # x = h + a*np.cos(t)*np.cos(c) - b*np.sin(t)*np.sin(c)
    # y = k + a*np.cos(t)*np.sin(c) + b*np.sin(t)*np.cos(c)
    # x = h + a*np.cos(t)
    # y = k + b*np.sin(t)
    c = math.radians(c) + math.pi
    x = h + a * np.cos(t) * np.cos(c) - b * np.sin(t) * np.sin(c)
    y = k + a * np.cos(t) * np.sin(c) + b * np.sin(t) * np.cos(c)
    return (x, y)
def oval_convert_polygon(x0, y0, a, t1, t2, c, b, h,num_control_points):
    # 上边界
    if t1 == 5760:
        t1 = 0

    if t1 > 180 * 16:
        t1 = t1 - 360 * 16

    t1 = math.radians(t1 / 16)
    t2 = t1 + math.radians(t2 / 16)

    spans = np.linspace(t1, t2,  int(num_control_points/2))
    top_polygon = []
    bottom_polygon = []
    for span in spans:
        (x, y) = oval(x0, y0, a + h / 2, span, c, b + h / 2)
        top_polygon.append(np.array([x,y]))


    # 下边界
    for span in spans:
        (x, y) = oval(x0, y0, a - h / 2, span, c, b - h / 2)
        bottom_polygon.append(np.array([x, y]))

    top_polygon = np.array(top_polygon)
    # bottom_polygon = np.array(bottom_polygon)
    bottom_polygon = np.array(bottom_polygon)[::-1]

    return np.concatenate((top_polygon, bottom_polygon), axis=0)
def oval_convert_key_points(x0, y0, a, t1, t2, c, b, num_control_points):
    # 上边界
    if t1 == 5760:
        t1 = 0

    if t1 > 180 * 16:
        t1 = t1 - 360 * 16

    t1 = math.radians(t1 / 16)
    t2 = t1 + math.radians(t2 / 16)

    spans = np.linspace(t1, t2,  int(num_control_points))
    top_polygon = []
    for span in spans:
        (x, y) = oval(x0, y0, a, span, c, b)
        top_polygon.append(np.array([x,y]))

    top_polygon = np.array(top_polygon)


    return top_polygon




# 转成 mask
def polygon_mask(item,shrink_points,poly_points,bbox,rbbox,types):
    num_control_points = 200
    if item["type"] ==2:
        x0 = item["rect"][2] / 2 + item["x"] + item["rect"][0]
        y0 = item["rect"][3] / 2 + +item["y"] + item["rect"][1]
        t = item["rotation"]
        h = item["h"]
        l = item["l"]
        points = line_convert_polygon(x0, y0, t,h,l,num_control_points)

        # shrink
        pts_shrink = np.array(points, dtype=np.int32).reshape((1, -1, 2))
        pts_shrink = np.array(shrink(pts_shrink, item["h"] * 0.3))
        # 原始图片大小
        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        shrink_points.append(pts_shrink)
        poly_points.append(pts)
        bbox.append(cv2.boundingRect(pts))
        rbbox.append(cv2.boxPoints(cv2.minAreaRect(pts)))
        types.append(2)
        return



    elif item["type"] ==5:
        # 圆弧
        x0 = item["rect"][2] / 2 + item["x"] + item["rect"][0]
        y0 = item["rect"][3] / 2 + +item["y"] + item["rect"][1]
        r = item["r"]
        t1 = item["startAngle"]
        t2 = item["spanAngle"]
        h = item["h"]
        rotation = item["rotation"]
        points = circle_convert_polygon(x0, y0, r, t1, t2, h, rotation,num_control_points)

    elif item["type"] ==6:
        # 椭圆弧
        a = item["a"]
        b = item["b"]
        t1 = item["startAngle"]
        t2 = item["spanAngle"]
        c = item["rotation"]
        h = item["h"]
        x0 = item["rect"][2] / 2 + item["x"] + item["rect"][0]
        y0 = item["rect"][3] / 2 + +item["y"] + item["rect"][1]
        points = oval_convert_polygon(x0, y0, a, t1, t2, c, b, h,num_control_points)

    # shrink
    pts_shrink = np.array(points, dtype=np.int32).reshape((1, -1, 2))
    pts_shrink = np.array(shrink(pts_shrink, item["h"] * 0.3))

    # 原始图片大小
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

    shrink_points.append(pts_shrink)
    poly_points.append(pts)
    bbox.append(cv2.boundingRect(pts))
    rbbox.append(cv2.boxPoints(cv2.minAreaRect(pts)))
    types.append(5)

def polygon_points(item,key_points):
    if item["type"] ==2:
        x0 = item["rect"][2] / 2 + item["x"] + item["rect"][0]
        y0 = item["rect"][3] / 2 + +item["y"] + item["rect"][1]
        t = item["rotation"]
        h = item["h"]
        l = item["l"]
        # points = line_convert_polygon(x0, y0, t,h,l,10)
        points = line_convert_key_points(x0, y0, t,l,10)


    elif item["type"] ==5:
        # 圆弧
        x0 = item["rect"][2] / 2 + item["x"] + item["rect"][0]
        y0 = item["rect"][3] / 2 + +item["y"] + item["rect"][1]
        r = item["r"]
        t1 = item["startAngle"]
        t2 = item["spanAngle"]
        h = item["h"]
        rotation = item["rotation"]
        # points = circle_convert_polygon(x0, y0, r, t1, t2, h, rotation,10)
        points = circle_convert_key_points(x0, y0, r, t1, t2,  rotation,10)

    elif item["type"] ==6:
        # 椭圆弧
        a = item["a"]
        b = item["b"]
        t1 = item["startAngle"]
        t2 = item["spanAngle"]
        c = item["rotation"]
        h = item["h"]
        x0 = item["rect"][2] / 2 + item["x"] + item["rect"][0]
        y0 = item["rect"][3] / 2 + +item["y"] + item["rect"][1]
        # points = oval_convert_polygon(x0, y0, a, t1, t2, c, b, h,10)
        points = oval_convert_key_points(x0, y0, a, t1, t2, c, b, 10)
        category_id = 3

    points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

    key_points.append(points)


    return key_points


def convertMask(annotation_data):
    shrink_points = []
    points = []
    id_s = []
    bbox = []
    rbbox = []
    types = []
    sns = 0
    annotation_labels = annotation_data["label"]
    for annotation_label in annotation_labels:
        for group_index,annotation_group in enumerate(annotation_label["groups"]):
            for link_index,link in enumerate(annotation_group["links"]):
                polygon_mask(link["entity"], shrink_points,points,bbox,rbbox,types)
                id_s.append(sns)
                sns+=1

    # 缩水多边形点、原点、序号点
    return shrink_points,points,id_s,bbox,rbbox,types


# 回去关键点
def convertPoints(annotation_data):
    key_points = []
    annotation_labels = annotation_data["label"]
    for annotation_label in annotation_labels:
        for group_index,annotation_group in enumerate(annotation_label["groups"]):
            for link_index,link in enumerate(annotation_group["links"]):
                polygon_points(link["entity"], key_points)

    return key_points