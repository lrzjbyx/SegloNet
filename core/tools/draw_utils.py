import numpy as np
from numpy import ndarray
import PIL
from PIL import ImageDraw, ImageFont
from PIL.Image import Image


point_name = ["0", "1", "2",
              "3", "4","5",
              "6","7", "8",
              "9"]

point_color = [(240, 2, 127), (169, 209, 142), (152, 2, 152),
               (52, 127, 12), (211, 2, 52),(255, 255, 51),
               (255, 255, 51),(5, 197, 41), (44, 215, 184),
               (66, 195, 194)]

line_name = ["curve","line"]
line_color = [(192,168,2),(3,12,152)]

# def draw_line(img: Image,item: dict):
#     if isinstance(img, ndarray):
#         img = PIL.Image.fromarray(img)
#
#     draw = ImageDraw.Draw(img)
#     if item["type"] !=2:
#         draw.arc((0, 0, radius * 2, radius * 2), start, end, fill=color, width=width)
#     pass







def draw_keypoints(img: Image,
                   keypoints: ndarray,
                   scores: ndarray = None,
                   thresh: float = 0.2,
                   r: int = 5,
                   draw_text: bool = False,
                   font: str = 'arial.ttf',
                   font_size: int = 10):
    if isinstance(img, ndarray):
        img = PIL.Image.fromarray(img)

    if scores is None:
        scores = np.ones(keypoints.shape[0])

    if draw_text:
        try:
            font = ImageFont.truetype(font, font_size)
        except IOError:
            font = ImageFont.load_default()

    draw = ImageDraw.Draw(img)
    for i, (point, score) in enumerate(zip(keypoints, scores)):
        if score > thresh and np.max(point) > 0:
            draw.ellipse([point[0] - r, point[1] - r, point[0] + r, point[1] + r],
                         fill=point_color[i],
                         outline=(255, 255, 255))
            if draw_text:
                draw.text((point[0] + r, point[1] + r), text=point_name[i], font=font)

    return img
