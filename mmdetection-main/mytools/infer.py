from mmdet.apis import DetInferencer
import os
import pandas as pd
import numpy as np
import cv2

# Choose to use a config
config_path = './configs/pascal_voc/faster-rcnn_r50_fpn_1x_voc0712.py'
# Setup a checkpoint file to load
checkpoint = './work_dirs/faster-rcnn_r50_fpn_1x_voc0712/epoch_12.pth'
# Initialize the DetInferencer
device = 'cuda:0'
inferencer = DetInferencer(model=config_path, weights=checkpoint, device=device)

#测试图片路径
#test_img = "./datasets/VOCdevkit/VOC2007/JPEGImages/000300.jpg"
test_img = "img.jpg"

result = inferencer(test_img)["predictions"][0]
print(result)
labels = result.get("labels")
scores = result.get("scores")
bboxes = result.get("bboxes")

src = cv2.imread(test_img)
img = cv2.imread('rpn_results.jpg')
img = cv2.resize(img, (src.shape[1], src.shape[0]))

for i in range(len(labels)):
    label = labels[i]
    score = scores[i]
    bbox = bboxes[i]
    left_top = (int(bbox[0]), int(bbox[1]))
    right_bottom = (int(bbox[2]), int(bbox[3]))
    cv2.rectangle(img, left_top, right_bottom, (0, 0, 255), 2)
    cv2.putText(img, f"{score:.2f}", left_top, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cv2.imwrite("result.jpg", img)