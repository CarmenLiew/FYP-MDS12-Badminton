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

class ObjectDetection:
    def __init__(self, capture, result, court):
        self.capture = capture
        self.result = result
        self.court = court
        self.model = self.load_model()  # Load YOLO or your chosen model for player detection
        self.player_colors = {}  # Store colors for each player
        
        # Load TrackNet model for shuttlecock tracking
        tracknet_ckpt = torch.load("ckpts/TrackNet_best.pt")
        tracknet_seq_len = tracknet_ckpt['param_dict']['seq_len']

    def __call__(self):
        # Open video capture and setup video output
        cap = cv2.VideoCapture(self.capture)
        out = self.prepare_output_video(cap)

        # Initialize plot
        plt.ion()  # Enable interactive mode for plotting
        fig, ax = plt.subplots(figsize=(12, 8))

        # Initialize tracking for players (DeepSORT)
        tracker = self.initialize_tracker()

        while True:
            ret, img = cap.read()
            if not ret:
                break

            # Detect players in the frame using YOLO or another model
            results = self.predict(img)  
            detections, frames = self.plot_boxes(results, img)  
            detect_frame = self.track_detect(detections, tracker, frames)  # Track players

            # Perform shuttlecock tracking
            shuttlecock_predictions = self.track_shuttlecock(detect_frame)
            self.draw_shuttlecock(detect_frame, shuttlecock_predictions)

            # Write the frame to output video
            out.write(detect_frame)

            # Visualize in matplotlib
            img_rgb = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)
            ax.clear()
            ax.imshow(img_rgb)
            plt.draw()
            plt.pause(0.001)  # Small pause to allow for image display

        # Clean up
        cap.release()
        out.release()
        plt.ioff()
        plt.show()

    def load_model(self):
        """Load the YOLO or other model for player detection."""
        # Load your pre-trained YOLO model here
        model = YOLO("yolov8l.pt")
        model.fuse()
        return model

    def predict(self, img):
        """Predict players using the loaded model."""
        # Forward pass the image through the model to get predictions
        results = self.model(img)
        return results

    def plot_boxes(self, results, img):
        """Draw bounding boxes for players."""
        detections = np.empty((0, 5))  # Placeholder for detected players
        for result in results:
            x1, y1, x2, y2, conf, label = result
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            detections = np.vstack((detections, [x1, y1, x2, y2, conf]))
        return detections, img

    def initialize_tracker(self):
        """Initialize DeepSORT tracker."""
        # Initialize your DeepSORT tracker here
        tracker = None  # Replace with actual tracker initialization
        return tracker

    def track_detect(self, detections, tracker, frames):
        """Track players using DeepSORT and draw their bounding boxes."""
        # Use the DeepSORT tracker to track detected players in frames
        # Update tracker with new detections and return the updated frame
        track_frame = frames  # Placeholder, update with tracking logic
        return track_frame

    def track_shuttlecock(self, frame):
        """Track the shuttlecock using TrackNet."""
        # You can extract features/heatmaps from the frame to predict the shuttlecock
        indices = self.get_shuttlecock_indices(frame)  # Placeholder function to get indices for TrackNet
        shuttlecock_pred = self.predict_shuttlecock(indices)
        return shuttlecock_pred

    def get_shuttlecock_indices(self, frame):
        """Extract indices/features from frame to input into TrackNet."""
        # You can use preprocessing or feature extraction to get the right input format for TrackNet
        indices = np.random.rand(224, 224, 3)  # Placeholder, replace with actual extraction logic
        return indices

    def predict_shuttlecock(self, indices):
        """Predict shuttlecock position using TrackNet."""
        # Input indices into TrackNet to get shuttlecock position
        with torch.no_grad():
            shuttlecock_pred = self.tracknet(torch.tensor(indices).unsqueeze(0))
        
        # Process output to get coordinates and visibility
        x, y, visibility = shuttlecock_pred['X'], shuttlecock_pred['Y'], shuttlecock_pred['Visibility']
        return {'X': x, 'Y': y, 'Visibility': visibility}

    def draw_shuttlecock(self, img, shuttlecock_pred):
        """Draw the shuttlecock on the frame."""
        for x, y, vis in zip(shuttlecock_pred['X'], shuttlecock_pred['Y'], shuttlecock_pred['Visibility']):
            if vis:
                cv2.circle(img, (int(x), int(y)), 5, (0, 255, 255), cv2.FILLED)  # Draw yellow circle for the shuttlecock

    def prepare_output_video(self, cap):
        """Prepare the output video file."""
        result_path = os.path.join(self.result, 'results.avi')
        codec = cv2.VideoWriter_fourcc(*'XVID')
        vid_fps = int(cap.get(cv2.CAP_PROP_FPS))
        vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return cv2.VideoWriter(result_path, codec, vid_fps, (vid_width, vid_height))

detector = ObjectDetection(capture="test.mp4", result='result', court=(450, 390, 1500, 1000))
detector()