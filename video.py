from algorithm.object_detector import YOLOv8
from tqdm import tqdm
import cv2

yolov8 = YOLOv8()
yolov8.load('coco.weights', classes='coco.yaml', device='cpu') # use 'gpu' for CUDA GPU inference

video = cv2.VideoCapture('video.mp4')
width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))
frames_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
output = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

if video.isOpened() == False:
	print('[!] error opening the video')

print('[+] detecting video...\n')
pbar = tqdm(total=frames_count, unit=' frames', dynamic_ncols=True, position=0, leave=True)

try:
    while video.isOpened():
        ret, frame = video.read()
        if ret == True:
            detections = yolov8.detect(frame)
            detected_frame = yolov8.draw(frame, detections)
            output.write(detected_frame)
            pbar.update(1)
        else:
            break
except KeyboardInterrupt:
    pass

pbar.close()
video.release()
output.release()
yolov8.unload()