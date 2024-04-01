import cv2
import os
import numpy as np
from read import read

def read_orientation(image_path):
    boxes = []
    # Carregar a imagem usando OpenCV
    image = cv2.imread(image_path)

    height, width, _ = image.shape
    blank_image = np.ones((height, width, 3), np.uint8) * 255  # preencher com branco
    blank_image = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)

    data = read(image_path)

    for i in range(len(data)):

        box = np.intp(data[i])
        top_right, top_left, bottom_left, bottom_right = box[0], box[1], box[2], box[3]
                    
        # calcula as coordenadas das extremidades da reta horizontal superior
        x1, y1, x2, y2= top_left[0], top_left[1], top_right[0], top_right[1]

        angle = np.arctan2(y1 - y2, x1 - x2) * 180 / np.pi
        if angle < -45: angle += 90
        area = cv2.contourArea(box)
        boxes.append((area, angle * area, abs(x1-x2), angle))

        cv2.drawContours(blank_image, [box], 0, (0, 255, 0), 2)

    boxes_ordenadas = sorted(boxes, key=lambda x: x[2], reverse=True)[:20]
    total_areas, total_weighted_angle, total_angle = 0, 0, 0
    for t in boxes_ordenadas:
        total_areas, total_weighted_angle, total_angle = total_areas+t[0], total_weighted_angle+t[1], total_angle+t[3]

    average_weighted_angle = total_weighted_angle / total_areas
    # average_weighted_angle = total_angle / len(boxes_ordenadas)

    return average_weighted_angle

if __name__ == "__main__":
    print(read_orientation('/home/dg/aleatorio/ufv/detect-text-orientation-with-craft-detector/a.jpg'))