from collections import OrderedDict
from utils.detections import draw
from ultralytics import YOLO
import json
import yaml


class YOLOv8:
    def __init__(self, version='nano'):
        self.version = version
        self.img_size = 640
        self.conf = 0.25
        self.iou = 0.7
        self.classes = []
        if version == 'nano':
            self.model_yaml = 'yolov8n-pose.yaml'
        elif version == 'small':
            self.model_yaml = 'yolov8s-pose.yaml'
        elif version == 'medium':
            self.model_yaml = 'yolov8m-pose.yaml'
        elif version == 'large':
            self.model_yaml = 'yolov8l-pose.yaml'
        elif version == 'extra-large':
            self.model_yaml = 'yolov8x-pose.yaml'

    def load(self, weights, classes, device='cpu'):
        self.weights = weights
        if device == 'cpu':
            self.to_cpu()
        elif device == 'gpu':
            self.to_gpu()

        with open(classes, 'r') as f:
            classes = yaml.load(f, Loader=yaml.FullLoader)
            classes = classes['classes']
            parsed_classes = []
            keypoints_to_indices = {}
            for class_ in classes:
                if 'skeleton' in class_:
                    keypoints_to_indices[class_['name']] = {}
                    if 'keypoints' in class_['skeleton']:
                        for i, keypoint in enumerate(class_['skeleton']['keypoints']):
                            keypoints_to_indices[class_['name']][keypoint['id']] = i
            for class_ in classes:
                new_class = {
                    'name': class_['name'],
                    'color': class_['color']
                }
                
                if 'skeleton' in class_:
                    new_class['skeleton'] = {
                        'keypoints': [],
                        'connections': []
                    }
                    if 'connections' in class_['skeleton']:
                        for connection in class_['skeleton']['connections']:
                            new_class['skeleton']['connections'].append({
                                'from': keypoints_to_indices[class_['name']][connection['from']],
                                'to': keypoints_to_indices[class_['name']][connection['to']],
                                'color': connection['color']
                            })
                    
                    if 'keypoints' in class_['skeleton']:
                        for keypoint in class_['skeleton']['keypoints']:
                            new_class['skeleton']['keypoints'].append(keypoint)
                
                parsed_classes.append(new_class)
            
            self.classes = parsed_classes

    def to_gpu(self):
        self.model = YOLO(self.weights, device='cuda')

    def to_cpu(self):
        self.model = YOLO(self.weights, device='cpu')

    def unload(self):
        pass
    
    def detect(self, image):
        results = self.model(image, conf=self.conf, iou=self.iou)
        detections = []

        for r in results:
            result = json.loads(r.tojson())
            for box in result:
                position = box['box']
                keypoints = []
                box_class = [class_ for class_ in self.classes if class_['name'] == box['name']][0]
                if 'keypoints' in box:
                    for i, keypoint_x in enumerate(box['keypoints']['x']):
                        keypoints.append({
                            'x': int(keypoint_x),
                            'y': int(box['keypoints']['y'][i]),
                            'visible': round(box['keypoints']['visible'][i], 2)
                        })
                box_dict = dict(OrderedDict([
                    ('class', box['name']),
                    ('confidence', round(box['confidence'], 2)),
                    ('x', int(position['x1'])),
                    ('y', int(position['y1'])),
                    ('width', int(position['x2'] - position['x1'])),
                    ('height', int(position['y2'] - position['y1'])),
                    ('keypoints', keypoints),
                    ('color', box_class['color'])
                ]))
                detections.append(box_dict)

        return detections

    def draw(self, image, detections):
        for detection in detections:
            for class_ in self.classes:
                if detection['class'] == class_['name'] and 'skeleton' in class_:
                    connections = class_['skeleton']['connections']
                    detection['connections'] = connections
                    break
        return draw(image, detections)

    def set(self, inference_config):
        if 'conf_thres' in inference_config:
            self.conf = inference_config['conf_thres']
        if 'iou_thres' in inference_config:
            self.iou = inference_config['iou_thres']