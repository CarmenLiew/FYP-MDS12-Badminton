import matplotlib
matplotlib.use('TkAgg')  # Set this to an interactive backend suitable for your system
import matplotlib.pyplot as plt
import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
from sort import *
import os
import math
import torch
from utils.general import *
from test import predict_location, get_ensemble_weight, generate_inpaint_mask
from dataset import Shuttlecock_Trajectory_Dataset, Video_IterableDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from predict import predict
from collections import deque

class ObjectDetection():
    def __init__(self, capture, result, court, tracknet_file):
        self.capture = capture
        self.result = result
        self.court = court  # court coordinates (xmin, ymin, xmax, ymax)
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names
        self.player_colors = {} # dictionary to store all colours for the players

        # Load TrackNet model for shuttlecock tracking
        tracknet_ckpt = torch.load(tracknet_file)
        self.tracknet_seq_len = tracknet_ckpt['param_dict']['seq_len']
        self.bg_mode = tracknet_ckpt['param_dict']['bg_mode']
        self.tracknet = get_model('TrackNet', self.tracknet_seq_len, self.bg_mode).cuda()
        self.tracknet.load_state_dict(tracknet_ckpt['model'])
        self.tracknet.eval()

    def load_model(self):
        model = YOLO("yolov8l.pt")
        model.fuse()
        return model

    def predict(self, img):
        results = self.model(img, stream=True)
        return results

    def plot_boxes(self, results, img, detections):
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                # Classname
                cls = int(box.cls[0])
                currentClass = self.CLASS_NAMES_DICT[cls]

                # Confidence score
                conf = math.ceil(box.conf[0] * 100) / 100

                # Check if bounding box intersects with the court
                if conf > 0.5 and self.is_within_court(x1, y1, x2, y2):
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))
                    # Optionally, draw the bounding box
                    # cvzone.putTextRect(img, f'class: {currentClass}', (x1, y1), scale=1, thickness=1, colorR=(0, 0, 255))
                    # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=1, colorR=(255, 0, 255))

        return detections, img

    def is_within_court(self, x1, y1, x2, y2):
        xmin, ymin, xmax, ymax = self.court
        # Check if any part of the bounding box is within the court
        return not (x2 < xmin or x1 > xmax or y2 < ymin or y1 > ymax)

    def track_detect(self, detections, tracker, img):
        resultTracker = tracker.update(detections)

        # Predefined colors (you can choose any colors you like)
        color_list = [
            (0, 0, 255),   # Red
            (0, 255, 0),   # Green
            (255, 0, 0),   # Blue
            (255, 255, 0)  # Yellow
        ]

        for res in resultTracker:
            x1, y1, x2, y2, id = res
            x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
            w, h = x2 - x1, y2 - y1

            cx, cy = x1 + w // 2, y1 + h // 2

            # Assign a color to the player ID if it hasn't been assigned yet
            if id not in self.player_colors:
                if len(self.player_colors) < len(color_list):
                    self.player_colors[id] = color_list[len(self.player_colors)]
                else:
                    # If more than 4 players, recycle colors (or add more unique colors)
                    self.player_colors[id] = color_list[id % len(color_list)]

            # Get the assigned color for this player
            color = self.player_colors[id]

            cvzone.putTextRect(img, f'Player: {id}', (x1, y1), scale=1, thickness=1, colorR=color)
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=1, colorR=color)

            print(f"Player ID: {id}, Coordinates: ({cx}, {cy})")

            cv2.circle(img, (cx, cy), 5, color, cv2.FILLED)

        return img

    # def track_shuttlecock(self, img, img_scaler):
        # Implement shuttlecock tracking using TrackNet
        # frame_list = generate_frames(self.capture)
        # dataset = Shuttlecock_Trajectory_Dataset(seq_len=self.tracknet_seq_len, sliding_step=1, data_mode='heatmap', bg_mode=self.bg_mode,
        #                                          frame_arr=np.array(frame_list)[:, :, :, ::-1])
        # data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)

        # for step, (i, x) in enumerate(tqdm(data_loader)):
        #     x = x.float().cuda()
        #     with torch.no_grad():
        #         y_pred = self.tracknet(x).detach().cpu()

        #     # Predict shuttlecock location
        #     pred_dict = predict(i, y_pred=y_pred, img_scaler=img_scaler)
        #     for frame, x, y, vis in zip(pred_dict['Frame'], pred_dict['X'], pred_dict['Y'], pred_dict['Visibility']):
        #         if vis:
        #             cv2.circle(img, (x, y), 5, (0, 255, 255), cv2.FILLED)  # Yellow circle for shuttlecock

        # return img
    def track_shuttlecock(self, img, img_scaler):
        # Implement shuttlecock tracking using TrackNet
        frame_list = generate_frames(self.capture)
        dataset = Shuttlecock_Trajectory_Dataset(seq_len=self.tracknet_seq_len, sliding_step=1, data_mode='heatmap', bg_mode=self.bg_mode,
                                                frame_arr=np.array(frame_list)[:, :, :, ::-1])
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)

        traj = deque(maxlen=20)  # Adjust maxlen to control the length of the trajectory

        for step, (i, x) in enumerate(tqdm(data_loader)):
            x = x.float().cuda()
            with torch.no_grad():
                y_pred = self.tracknet(x).detach().cpu()

            # Predict shuttlecock location
            pred_dict = predict(i, y_pred=y_pred, img_scaler=img_scaler)
            for frame, x, y, vis in zip(pred_dict['Frame'], pred_dict['X'], pred_dict['Y'], pred_dict['Visibility']):
                if vis:
                    traj.append((x, y))

                    # Print the shuttlecock coordinates
                    print(f"Frame: {frame}, X: {x}, Y: {y}")

        # Draw the trajectory on the image
        img = draw_traj(img, traj, radius=3, color='yellow')

        return img
        

    def __call__(self):
        cap = cv2.VideoCapture(self.capture)
        assert cap.isOpened(), "Failed to open video capture"

        result_path = os.path.join(self.result, 'results.avi')

        codec = cv2.VideoWriter_fourcc(*'XVID')
        vid_fps = int(cap.get(cv2.CAP_PROP_FPS))
        vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = cv2.VideoWriter(result_path, codec, vid_fps, (vid_width, vid_height))

        tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

        if not os.path.exists(self.result):
            os.makedirs(self.result)
            print("Result folder created successfully")
        else:
            print("Result folder already exists")

        # Set up the matplotlib figure and axes
        plt.ion()  # Interactive mode on
        fig, ax = plt.subplots(figsize=(12, 8))

        img_scaler = (vid_width / WIDTH, vid_height / HEIGHT)

        while True:
            ret, img = cap.read()
            if not ret:
                break

            assert img.shape[0] == vid_height and img.shape[1] == vid_width, "Frame dimensions do not match video dimensions"

            detections = np.empty((0, 5))
            results = self.predict(img)
            detections, frames = self.plot_boxes(results, img, detections)
            detect_frame = self.track_detect(detections, tracker, frames)
            detect_frame = self.track_shuttlecock(detect_frame, img_scaler)

            out.write(detect_frame)

            # Convert BGR to RGB for displaying with matplotlib
            img_rgb = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)
            ax.clear()
            ax.imshow(img_rgb)
            plt.draw()
            plt.pause(0.001)  # Small pause to allow for image display

            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        out.release()
        plt.ioff()  # Turn off interactive mode

detector = ObjectDetection(capture="test.mp4", result='result', court=(450, 390, 1500, 1000), tracknet_file='ckpts/TrackNet_best.pt')
detector()