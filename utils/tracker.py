import cv2
from filterpy.kalman import KalmanFilter
import numpy as np

class Track:
    def __init__(self, id, bbox):
        self.id = id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.hits = 0
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.x[:4] = self.bbox_center(bbox)
        self.kf.F[:4, :4] = np.eye(4)
        self.kf.H = np.eye(4, 7)
    
    def bbox_center(self, bbox):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return np.array([cx, cy, w, h, 0, 0, 0])
    
    def update(self, bbox):
        self.bbox = bbox
        self.kf.update(self.bbox_center(bbox))
        self.hits += 1

class Tracker:
    def __init__(self):
        self.tracks = []
        self.next_id = 1

    def iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

    def update(self, detections):
        updated_tracks = []
        for det in detections:
            matched = False
            for track in self.tracks:
                if self.iou(track.bbox, det[:4]) > 0.3:
                    track.update(det[:4])
                    updated_tracks.append((track.id, track.bbox))
                    matched = True
                    break
            if not matched:
                new_track = Track(self.next_id, det[:4])
                self.tracks.append(new_track)
                updated_tracks.append((self.next_id, det[:4]))
                self.next_id += 1

        return updated_tracks
