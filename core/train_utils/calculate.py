import math
import numpy as np
from shapely.geometry import Polygon
import cv2

def calculate_angle_ann(aa,bb):
    ax,ay = aa[0],aa[1]
    bx,by = bb[0],bb[1]
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy==0:
        angle = 0
    else:
        angle = math.atan2(-dy, dx)
        angle = math.degrees(angle)

    return (angle+360)%360


def calculate_angle_draw(aa,bb):
    ax,ay = aa[0],aa[1]
    bx,by = bb[0],bb[1]
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy==0:
        angle = 0
    else:
        angle = math.atan2(dy, dx)
        angle = math.degrees(angle)

    return (angle+360)%360


def calculate_angle_diff_ann(angle1, angle2):
    # 计算角度差值
    angle_diff = angle2 - angle1
    # 将差值转换到0到360度之间
    angle_diff = angle_diff % 360
    # 如果差值为负数，将其转换为正数
    if angle_diff < 0:
        angle_diff += 360
    return angle_diff

'''
逆时针角度减法
'''
def counter_clockwise_subtract(angle, subtract_angle):
    angle = np.deg2rad(angle)  # 将角度转换为弧度
    subtract_angle = np.deg2rad(subtract_angle)
    result = (angle - subtract_angle) % (2 * np.pi)  # 逆时针减小角度，确保结果在 0 到 2*pi 范围内
    result = np.rad2deg(result)  # 将弧度转换为角度
    return result

def counter_clockwise_difference(angle1, angle2):
    angle1 = angle1 % 360  # 确保角度值在 0 到 359 范围内
    angle2 = angle2 % 360
    difference = (angle2 - angle1) % 360  # 逆时针差值
    if difference > 180:
        difference -= 360  # 调整为负数
    return difference

''' 
    行高计算
'''
class RowHeightCalculator:
    def __init__(self, thickness=10):
        self.thickness = thickness

    def angle_with_x_axis(self,point_a, point_b):
        dx = point_b[0] - point_a[0]
        dy = point_b[1] - point_a[1]

        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    def calculate_rectangle(self,image, start_point, end_point, length):
        canvas = np.zeros(image.shape[:2], dtype=np.uint8)

        angle = self.angle_with_x_axis(start_point,end_point)

        # 计算中心线的结束位置
        end_point = (
        start_point[0] + length * np.cos(np.deg2rad(angle)), start_point[1] + length * np.sin(np.deg2rad(angle)))
        thickness = 10
        cv2.line(canvas,
                 (int(start_point[0]), int(start_point[1])),
                 (int(end_point[0]), int(end_point[1])),
                 255,
                 thickness
                 )

        # cv2.imwrite("sdacanvas.png",canvas)
        # 使用阈值化将图像转换为二值图像
        _, binary = cv2.threshold(canvas, 127, 255, cv2.THRESH_BINARY)

        # 查找轮廓
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rect = cv2.minAreaRect(contours[0])
        points = cv2.boxPoints(rect)
        points = np.int0(points)
        # print(cv2.contourArea(points)/thickness)
        return points

    def calculate_sector_row_height(self,poly, rect_points):

        dwdw = np.zeros((1000,1000,3),dtype=np.uint8)
        rect_points = rect_points.reshape(-1,1,2)
        cv2.fillPoly(dwdw,[poly.astype(np.int32)],(0,255,153))
        cv2.fillPoly(dwdw,[rect_points.astype(np.int32)],(50,255,153))

        cv2.imwrite("sdawdxc.png",dwdw)

        poly = Polygon(np.array(poly, dtype=object).reshape(-1, 2)).buffer(0.01)
        points = Polygon(np.array(rect_points, dtype=object).reshape(-1, 2)).buffer(0.01)
        hh = poly.intersection(points).area / self.thickness
        return hh

    def calculate(self,image, start_point, end_point, length,poly):
        points = self.calculate_rectangle(image, start_point, end_point, length)
        hh = self.calculate_sector_row_height(poly,points)
        return hh


def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    return distance


'''
a_c_range [a,c]
b
'''
def determine_angles(a_c_range, b):
    # 规范化 b 的角度到 0 到 360 度之间
    b = b % 360

    # 获取数组中 a 和 c 的值
    a, c = a_c_range

    # 根据 a 和 c 的关系确定角度的位置
    if c < a:
        c += 360

    if b > c:
        a += 360

    return a%360, c%360


def find_start_end_angles(angles):
    start_angle = angles[0]
    end_angle = angles[0]
    max_gap = 0

    # 将列表排序
    sorted_angles = sorted(angles)

    # 计算相邻角度之间的最大间隔
    for i in range(len(sorted_angles) - 1):
        gap = sorted_angles[i+1] - sorted_angles[i]
        if gap > max_gap:
            max_gap = gap
            start_angle = sorted_angles[i+1]
            end_angle = sorted_angles[i]

    # 特殊情况：处理 359 和 0 之间的间隔
    gap = 360 - sorted_angles[-1] + sorted_angles[0]
    if gap > max_gap:
        start_angle = sorted_angles[0]
        end_angle = sorted_angles[-1]

    return start_angle, end_angle