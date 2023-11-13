from algorithm.object_detector import YOLOv8
from multiprocessing import Pool
import json
import cv2

WEIGHTS = 'coco.weights'
CLASSES = 'coco.yaml'
DEVICE  = 'cpu'         # use 'gpu' for CUDA GPU inference
STREAMS = 'streams.txt'

yolov8 = YOLOv8()
yolov8.load(WEIGHTS, classes=CLASSES, device=DEVICE) 

def detect(url):
    stream = cv2.VideoCapture(url)

    if stream.isOpened() == False:
        print(f'[!] error opening {url}')

    try:
        while stream.isOpened():
            ret, frame = stream.read()
            if ret == True:
                detections = yolov8.detect(frame)
                detected_frame = yolov8.draw(frame, detections)
                print(f'\n{url}:\n', json.dumps(detections, indent=4))
                cv2.imshow(url, detected_frame)
                cv2.waitKey(1)
            else:
                break
    except KeyboardInterrupt:
        pass

    stream.release()
    print(f'[+] stream {url} closed')

if __name__ == '__main__':
    with open(STREAMS) as file:
        streams_urls = [line.rstrip() for line in file]
        stream_count = len(streams_urls)
        print(f'[+] found {stream_count} stream urls')
        print('[+] starting live detection...')
        
        with Pool(stream_count) as process_pool:
            process_pool.map(detect, streams_urls)

    yolov8.unload()