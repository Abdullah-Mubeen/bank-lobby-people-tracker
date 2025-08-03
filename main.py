import cv2
import os
from ultralytics import YOLO
from utils.tracker import Tracker 
from utils.roi import ROI
from utils.visualizer import Visualizer

# Load model
model = YOLO("yolov8m.pt")

# Initialize ROI
roi = ROI("roi_config.json")
visualizer = Visualizer(roi)

# Setup video
video_path = "bank_lobby.mp4"  # Or use 0 for webcam
cap = cv2.VideoCapture(video_path)

# Create output directory
os.makedirs("output", exist_ok=True)

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
out = cv2.VideoWriter("output/result_tracking.mp4", fourcc, fps, (frame_width, frame_height))

# Log file
log_file = open("output/waiting_log.txt", "w")
log_file.write("frame,waiting_count\n")

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 Tracking (built-in tracker with persistent IDs)
    results = model.track(frame, persist=True, conf=0.5, iou=0.5, classes=[0])
    dets = results[0].boxes

    roi_count = 0

    if dets is not None and dets.id is not None:
        boxes = dets.xyxy.cpu().numpy()
        ids = dets.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            inside_roi = roi.is_inside((cx, cy))

            if inside_roi:
                roi_count += 1

            visualizer.draw_box(frame, box, track_id, inside_roi)

    visualizer.draw_roi(frame)
    visualizer.draw_count(frame, roi_count)

    # Save frame to output video
    out.write(frame)

    # Log count
    log_file.write(f"{frame_count},{roi_count}\n")
    frame_count += 1

    # Show live output
    cv2.imshow("Bank People Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
out.release()
log_file.close()
cv2.destroyAllWindows()
