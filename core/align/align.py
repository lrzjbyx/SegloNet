import math
import numpy as np
import cv2
from scipy import ndimage


class Align:

    '''
        直线坐标
    '''
    @staticmethod
    def line(x0, y0, t, h):
        x1 = x0 + h * np.cos(math.radians(t))
        y1 = y0 + h * np.sin(math.radians(t))
        return (x1, y1)

    '''
        椭圆坐标
    '''
    @staticmethod
    def oval(h, k, a, t, c, b):
        x = h + a * np.cos(t) * np.cos(c) + b * np.sin(t) * np.sin(c)
        y = k + a * np.cos(t) * np.sin(c) - b * np.sin(t) * np.cos(c)
        return (x, y)

    '''
        椭圆坐标
    '''
    @staticmethod
    def circle(x0, y0, r, t):
        x = x0 + r * np.cos(t)
        y = y0 - r * np.sin(t)
        return (x, y)


    '''
        计算弧长
    '''
    @staticmethod
    def elliptical_arc_length(a, b, theta1, theta2):
        h = ((a - b) / (a + b)) ** 2
        L = math.pi * (a + b) * (1 + ((3 * h) / (10 + math.sqrt(4 - 3 * h))))
        return L * abs(theta2 - theta1) / (2 * math.pi)


    def __init__(self,w,h):
        self.w = w
        self.h = h
        self.canvas = None



    def line_align(self,image,item):

        # 计算宽度
        scaled = item["l"]/item["h"]
        w  = int(self.h*scaled)

        l = item["l"]
        x0 = item["rect"][2] / 2 + item["x"] + item["rect"][0]
        y0 = item["rect"][3] / 2 + +item["y"] + item["rect"][1]
        h = item["h"]
        t = item["rotation"]/16

        hh = np.linspace(-h / 2, h / 2, self.h)
        ll = np.linspace(-l / 2, l / 2, w)

        self.canvas = np.zeros((self.h, w, 3), dtype=np.uint8)

        for row in range(self.canvas.shape[0]):
            tx = x0 - hh[row] * np.cos(math.pi / 2 - math.radians(t))
            ty = y0 + hh[row] * np.sin(math.pi / 2 - math.radians(t))
            for col in range(self.canvas.shape[1]):
                x, y = Align.line(tx, ty, t, ll[col])
                coord = np.array([[y], [x]])
                self.canvas[row, col, 0] = ndimage.map_coordinates(image[:, :, 0], coord, order=1)
                self.canvas[row, col, 1] = ndimage.map_coordinates(image[:, :, 1], coord, order=1)
                self.canvas[row, col, 2] = ndimage.map_coordinates(image[:, :, 2], coord, order=1)

        return self.canvas

    def arc_align(self,image, item):

        start_angle = math.radians(item["startAngle"] / 16)
        span_angle = math.radians(item["spanAngle"] / 16)

        # 计算弧线的长度
        arc_length = Align.elliptical_arc_length(item["a"],item["b"],start_angle,start_angle + span_angle)
        # print(arc_length)
        scaled = arc_length/item["h"]
        w = int(self.h*scaled)


        cc = [item["rect"][2] / 2 + item["x"] + item["rect"][0],
                         item["rect"][3] / 2 + +item["y"] + item["rect"][1]]
        xx = np.linspace(start_angle, start_angle + span_angle, w)
        aa = np.linspace(item["a"] - (item["h"] / 2), item["a"] + (item["h"] / 2), self.h)[::-1]
        bb = np.linspace(item["b"] - (item["h"] / 2), item["b"] + (item["h"] / 2), self.h)[::-1]
        # ro = math.radians(math.degrees(item["rotation"] / 16))
        ro = math.radians(item["rotation"] / 16)



        self.canvas = np.zeros((self.h, w, 3), dtype=np.uint8)
        for row in range(self.canvas.shape[0]):
            for col in range(self.canvas.shape[1]):
                x, y = Align.oval(cc[0], cc[1], aa[row], xx[col], ro, bb[row])
                # b = bilinear_interpolation(image[:, :, 0], (x, y))
                # g = bilinear_interpolation(image[:, :, 1], (x, y))
                # r = bilinear_interpolation(image[:, :, 2], (x, y))

                coord = np.array([[y], [x]])
                self.canvas[row, col, 0] = ndimage.map_coordinates(image[:, :, 0], coord, order=1)
                self.canvas[row, col, 1] = ndimage.map_coordinates(image[:, :, 1], coord, order=1)
                self.canvas[row, col, 2] = ndimage.map_coordinates(image[:, :, 2], coord, order=1)


                # self.canvas[row, col, :] = image[int(y), int(x), :]

        return cv2.flip(self.canvas, 1)


    def run(self,image,item):
        la = item["la"]
        mu = item["mu"]

        if la == 1:
            return self.arc_align(image,item)
        elif mu == 1:
            return self.line_align(image,item)

