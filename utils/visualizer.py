import cv2

class Visualizer:
    def __init__(self, roi):
        self.roi = roi

    def draw_box(self, frame, box, track_id, in_roi):
        x1, y1, x2, y2 = map(int, box)
        color = (255, 255, 255)  # white box
        label = f"ID: {track_id}"

        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Shadow text background
        cv2.putText(frame, label, (x1 + 1, y1 - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        # Text foreground
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    def draw_roi(self, frame):
        self.roi.draw(frame)

    def draw_count(self, frame, count):
        text = f"People Waiting: {count}"
        cv2.putText(frame, text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
