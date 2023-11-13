from algorithm.object_detector import YOLOv8
import json
import cv2

yolov8 = YOLOv8()
yolov8.load('coco.weights', classes='coco.yaml', device='cpu') # use 'gpu' for CUDA GPU inference
image = cv2.imread('image.jpg')
detections = yolov8.detect(image)
detected_image = yolov8.draw(image, detections)
cv2.imwrite('detected_image.jpg', detected_image)
print(json.dumps(detections, indent=4))