import streamlit as st
import numpy as np
import datetime
import cv2
import tempfile
from ultralytics import YOLO
from collections import deque

from deep_sort.deep_sort.tracker import Tracker
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.tools import generate_detections as gdet


st.sidebar.title("Navigation")
app_mode = st.sidebar.radio(
    "Choose an option",
    ["Road Surveillance", "Team Members"]
)

if app_mode == "Team Members":
    st.header("👥 Team Members")
    st.write("""
    - **CH. SURYA SAI MANIKANTA VTU21726**  
    - **K.SAI CHARAN TEJA VTU23822** 
    - **T. OM SEKHAR VTU21950**  
    """)

elif app_mode == "Road Surveillance":
    st.title("Traffic surveillance system with vehicle counting and speed detection")
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        video_path = tfile.name
        
        conf_threshold = 0.5
        max_cosine_distance = 0.4
        nn_budget = None
        points = [deque(maxlen=32) for _ in range(1000)]
        counted_ids = set()
        total_count = 0

        video_cap = cv2.VideoCapture(video_path)
        model = YOLO("yolov8s.pt")
        encoder = gdet.create_box_encoder("config/mars-small128.pb", batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        tracker = Tracker(metric)

        with open("config/coco.names", "r") as f:
            class_names = f.read().strip().split("\n")
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(class_names), 3))

        stframe = st.empty()

        while video_cap.isOpened():
            start = datetime.datetime.now()
            ret, frame = video_cap.read()
            if not ret:
                break

            results = model(frame)
            for result in results:
                bboxes, confidences, class_ids = [], [], []
                for data in result.boxes.data.tolist():
                    x1, y1, x2, y2, confidence, class_id = data
                    if confidence > conf_threshold:
                        x, y = int(x1), int(y1)
                        w, h = int(x2 - x1), int(y2 - y1)
                        bboxes.append([x, y, w, h])
                        confidences.append(confidence)
                        class_ids.append(int(class_id))

            names = [class_names[i] for i in class_ids]
            features = encoder(frame, bboxes)
            dets = [Detection(b, c, n, f) for b, c, n, f in zip(bboxes, confidences, names, features)]

            tracker.predict()
            tracker.update(dets)

            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                bbox = track.to_tlbr()
                track_id = track.track_id
                class_name = track.get_class()
                x1, y1, x2, y2 = map(int, bbox)

                class_id = class_names.index(class_name)
                color = colors[class_id]
                B, G, R = int(color[0]), int(color[1]), int(color[2])

                if track_id not in counted_ids:
                    counted_ids.add(track_id)
                    total_count += 1

                pixels_per_meter = 140 / 3.5
                fps_video = video_cap.get(cv2.CAP_PROP_FPS)
                if len(points[track_id]) >= 2:
                    dx = points[track_id][-1][0] - points[track_id][-2][0]
                    dy = points[track_id][-1][1] - points[track_id][-2][1]
                    distance_pixels = np.sqrt(dx ** 2 + dy ** 2)
                    distance_meters = distance_pixels / pixels_per_meter
                    speed_mps = distance_meters * fps_video
                    speed_kmph = speed_mps * 3.6
                else:
                    speed_kmph = 0.0

                speed_text = f"{speed_kmph:.1f} km/h"
                cv2.putText(frame, speed_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (B, G, R), 2)

                label = f"{track_id} - {class_name}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 3)
                cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(label) * 12, y1), (B, G, R), -1)
                cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                points[track_id].append((center_x, center_y))
                cv2.circle(frame, (center_x, center_y), 4, (0, 255, 0), -1)

                for i in range(1, len(points[track_id])):
                    if points[track_id][i - 1] is None or points[track_id][i] is None:
                        continue
                    cv2.line(frame, points[track_id][i - 1], points[track_id][i], (0, 255, 0), 2)

            fps = f"FPS: {1 / (datetime.datetime.now() - start).total_seconds():.2f}"
            cv2.putText(frame, fps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.rectangle(frame, (10, 50), (270, 100), (0, 0, 0), -1)
            cv2.putText(frame, f"Total Vehicles: {total_count}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        
        video_cap.release()
