from algorithm.object_detector import YOLOv8
from utils.detections import draw
import json
import cv2

yolov8 = YOLOv8()
yolov8.load('coco.weights', classes='coco.yaml', device='cpu') # use 'gpu' for CUDA GPU inference
webcam = cv2.VideoCapture(0)

if webcam.isOpened() == False:
	print('[!] error opening the webcam')

try:
    while webcam.isOpened():
        ret, frame = webcam.read()
        if ret == True:
            detections = yolov8.detect(frame)
            detected_frame = yolov8.draw(frame, detections)
            cv2.imshow('webcam', detected_frame)
            cv2.waitKey(1)
        else:
            break
except KeyboardInterrupt:
    pass

webcam.release()
print('[+] webcam closed')
yolov8.unload()