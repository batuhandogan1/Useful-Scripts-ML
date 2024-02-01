import cv2
import os
from yolo_show_annos import get_anno
import argparse

parser = argparse.ArgumentParser(description='A tool for cleaning a dataset quickly (YOLO format only)')
parser.add_argument('--path', type=str, required=True, help='Path for dataset containing images and annotations')

args = parser.parse_args()

path = args.path

files = os.listdir(path)
paths = []
bbox = []

for file in files:
    arr = file.split('.')

    if arr[0] not in paths:
        paths.append(arr[0])

for i, plain_path in enumerate(paths):
    image = get_anno(path + plain_path, i)
    cv2.imshow('image', image)

    k = cv2.waitKey(0)
    if k == 115:
        os.remove(path + plain_path + '.jpg')
        os.remove(path + plain_path + '.txt')
    if k == 107:
        continue

cv2.destroyAllWindows()