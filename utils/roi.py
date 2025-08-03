import json
import cv2
import numpy as np

class ROI:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.roi = np.array(json.load(f)["roi"], dtype=np.int32)

    def is_inside(self, point):
        return cv2.pointPolygonTest(self.roi, point, False) >= 0

    def draw(self, frame):
        cv2.polylines(frame, [self.roi], isClosed=True, color=(0, 255, 255), thickness=2)
