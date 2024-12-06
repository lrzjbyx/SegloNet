import cv2
import numpy as np
import math
import matplotlib.path as mplPath
from core.train_utils.calculate import calculate_angle_ann, counter_clockwise_subtract, counter_clockwise_difference, \
    RowHeightCalculator, calculate_distance, determine_angles, find_start_end_angles
from core.align.align import Align
# 圆弧行高计算
rhc = RowHeightCalculator()

def convert_to_arc(keypoint,image,polygons):

    # poly_path = mplPath.Path(polygons.reshape(-1,2))
    #
    # inside = []
    # outside = []
    #
    # for kp in keypoint:
    #     if poly_path.contains_point(kp):
    #         inside.append(kp)
    #     else:
    #         outside.append(kp)
    #
    # inside = np.array(inside)
    params = cv2.fitEllipse(keypoint.astype(np.int32))
    params = cv2.fitEllipseDirect(keypoint.astype(np.int32))
    # params = cv2.fitEllipseAMS(keypoint.astype(np.int32))
    # 计算开始角度
    # print((-params1[2]+360+90)%360)
    rotated_angle = -params[2] + 90
    # print((params1[2]-90+360)%360)
    # start_angle = calculate_angle_ann([params[0][0], params[0][1]], keypoint[-1])
    # end_angle = calculate_angle_ann([params[0][0], params[0][1]], keypoint[0])



    polys = polygons.copy().reshape(-1,2)
    angles = []

    for poly in polys:
        angles.append(calculate_angle_ann([params[0][0], params[0][1]], poly))
        # calculate_angle_ann([params[0][0], params[0][1]], poly)

    # print(angles)
    # angles = np.array(angles).astype(np.int32)
    # mid_angle = calculate_angle_ann([params[0][0], params[0][1]], np.mean(polys, axis=0))

    start_angle, end_angle = find_start_end_angles(angles)

    # start_angle,end_angle = determine_angles([start_angle,end_angle],mid_angle)

    span_angle = counter_clockwise_subtract(end_angle, start_angle)
    # print("start_angle:{0}".format(start_angle))
    # print("end_angle:{0}".format(end_angle))
    # print("旋转角度:{0}".format(rotated_angle))
    # print("过度角度:{0}".format(span_angle))
    # print("开始角度:{0}".format(start_angle))
    diff = counter_clockwise_difference(rotated_angle, start_angle)
    # print("开始角度:{0}".format((diff + 360) % 360))
    # hh = rhc.calculate(image, params[0], keypoint[4], 2 * max(params[1][0] / 2, params[1][1] / 2),
    #                    polygons.astype(np.int32))
    center, size, angle = cv2.minAreaRect(polygons.astype(np.int32))
    # calculate(self,image, start_point, end_point, length,poly)
    hh = rhc.calculate(image, params[0], center, 2 * max(params[1][0] / 2, params[1][1] / 2),
                       polygons.astype(np.int32))

    #"面积过小"
    if params[1][0]*params[1][1]< 20:
        item = convert_to_line(keypoint, image, polygons)
        return item

    # center,size,angle = cv2.minAreaRect(polygons.astype(np.int32))

    # print(cv2.contourArea(polygons.astype(np.int32))/(size[0]*size[1]))

    # 矩形
    if cv2.contourArea(polygons.astype(np.int32))/(size[0]*size[1]) >= 0.8:
        item = convert_to_line(keypoint, image, polygons)
        return item



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
            params[0][0] - params[1][0] / 2, params[0][1] - params[1][1] / 2, params[1][0],
            params[1][1]
        ],
        "rotation": -rotated_angle * 16,
        "text": "",
        "type": 6,
        "sequence": "从左到右",
        "startAngle": ((diff + 360) % 360) * 16,
        "spanAngle": span_angle * 16,
        "b": params[1][0] / 2,
        "a": params[1][1] / 2,
        "h": hh*0.9

    }
    return item

def convert_to_line(keypoint,image,polygons):
    # rotated_angle = calculate_angle_ann(keypoint[0],keypoint[-1])
    # distance = calculate_distance(keypoint[0],keypoint[-1])
    center,size,angle = cv2.minAreaRect(polygons.astype(np.int32))
    angle_rad = np.deg2rad(angle)
    vx, vy, x, y = cv2.fitLine(keypoint, cv2.DIST_L1, 0, 0.01, 0.01)
    # vx, vy, x, y = cv2.fitLine(keypoint, cv2.DIST_WELSCH, 0, 0.01, 0.01)
    angle_a_deg = np.degrees(np.arctan(vy / vx))[0]

    if size[0] < size[1]:
        ll = size[1]
        hh = size[0]
        angle_rad += np.pi / 2
    else:
        ll = size[0]
        hh = size[1]

    # 将夹角转换为角度制
    angle_a_deg = np.rad2deg(angle_rad)


    item = {
        'x': 0.0,
        'y': 0.0,
        'rect': [center[0]-ll/2, center[1]-hh/2, ll, hh],
        'rotation': angle_a_deg*16,
        'text': '',
        'type': 2,
        'sequence': '从左到右',
        'l': ll,
        'h': hh,
        'la': 0,
        'mu': 1
    }

    return item



def centerline_to_poly(item,num_control_points = 100):
    if item["type"] == 2:
        return line_to_poly(item,num_control_points)
    elif item["type"] == 6:
        return arc_to_poly(item,num_control_points)
    else:
        return None


def arc_to_poly(item,num_control_points):
    start_angle = math.radians(item["startAngle"] / 16)
    span_angle = math.radians(item["spanAngle"] / 16)

    # 中心点
    cc = [item["rect"][2] / 2 + item["x"] + item["rect"][0],
          item["rect"][3] / 2 + +item["y"] + item["rect"][1]]
    # 角度
    xx = np.linspace(start_angle, start_angle + span_angle, num_control_points)
    aa = np.linspace(item["a"] - (item["h"] / 2), item["a"] + (item["h"] / 2), 2)[::-1]
    bb = np.linspace(item["b"] - (item["h"] / 2), item["b"] + (item["h"] / 2), 2)[::-1]
    # ro = math.radians(math.degrees(item["rotation"] / 16))
    ro = math.radians(item["rotation"] / 16)

    top_polygon = []
    bottom_polygon = []
    for i,x in enumerate(xx):
        (x, y) = Align.oval(cc[0], cc[1], aa[0], x, ro, bb[0])
        top_polygon.append(np.array([x,y]))

    for i,x in enumerate(xx):
        (x, y) = Align.oval(cc[0], cc[1], aa[1], x, ro, bb[1])
        bottom_polygon.append(np.array([x,y]))

    top_polygon = np.array(top_polygon)
    bottom_polygon = np.array(bottom_polygon)[::-1]

    return np.concatenate((top_polygon, bottom_polygon), axis=0)

def line_to_poly(item,num_control_points):
    l = item["l"]
    x0 = item["rect"][2] / 2 + item["x"] + item["rect"][0]
    y0 = item["rect"][3] / 2 + +item["y"] + item["rect"][1]
    h = item["h"]
    t = item["rotation"] / 16

    hh = np.linspace(-h / 2, h / 2, 2)
    ll = np.linspace(-l / 2, l / 2, num_control_points)


    top_polygon = []
    bottom_polygon = []

    tx = x0 - hh[0] * np.cos(math.pi / 2 - math.radians(t))
    ty = y0 + hh[0] * np.sin(math.pi / 2 - math.radians(t))
    for i,l in enumerate(ll):
        x, y = Align.line(tx, ty, t, l)
        top_polygon.append(np.array([x,y]))

    tx = x0 - hh[1] * np.cos(math.pi / 2 - math.radians(t))
    ty = y0 + hh[1] * np.sin(math.pi / 2 - math.radians(t))
    for i,l in enumerate(ll):
        x, y = Align.line(tx, ty, t, l)
        bottom_polygon.append(np.array([x,y]))


    top_polygon = np.array(top_polygon)
    bottom_polygon = np.array(bottom_polygon)[::-1]


    return np.concatenate((top_polygon, bottom_polygon), axis=0)


def circle_to_poly(item,num_control_points):

    x0 = item["rect"][2] / 2 + item["x"] + item["rect"][0]
    y0 = item["rect"][3] / 2 + +item["y"] + item["rect"][1]
    r = item["r"]
    t1 = item["startAngle"]
    t2 = item["spanAngle"]
    h = item["h"]
    rotation = item["rotation"]

    t1 = math.radians(t1 / 16)-math.radians(rotation/16)
    t2 = t1 + math.radians(t2 / 16)
    spans = np.linspace(t1, t2, int(num_control_points/2))[::-1]

    top_polygon = []
    bottom_polygon = []
    for span in spans:
        (x,y) = Align.circle(x0,y0,r+h/2,span)
        top_polygon.append(np.array([x,y]))

    for span in spans:
        (x,y) = Align.circle(x0,y0,r-h/2,span)
        bottom_polygon.append(np.array([x, y]))


    top_polygon = np.array(top_polygon)
    bottom_polygon = np.array(bottom_polygon)[::-1]

    return np.concatenate((top_polygon,bottom_polygon),axis=0)

def convert_polygon(item):
    num_control_points = 200
    item["rotation"] = item["rotation"] * 16
    if item["type"] == 2:
        # 直线
        points = line_to_poly(item,num_control_points)
    elif item["type"] ==5:
        # 圆弧
        points = circle_to_poly(item,num_control_points)
    elif item["type"] ==6:
        # 椭圆弧
        points = arc_to_poly(item,num_control_points)


    pts = np.array(points).reshape((-1, 1, 2))

    return pts