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
        self.capture_path = capture  # Store the path to the video file
        self.result = result
        self.court = court  # court coordinates (xmin, ymin, xmax, ymax)
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names
        self.player_colors = {}  # dictionary to store all colours for the players
        self.previous_player_positions = {}  # Store player positions for velocity calculation
        self.previous_shuttlecock_pos = None  # Track previous shuttlecock position

        # Load TrackNet model for shuttlecock tracking
        tracknet_ckpt = torch.load(tracknet_file)
        self.tracknet_seq_len = tracknet_ckpt['param_dict']['seq_len']
        self.bg_mode = tracknet_ckpt['param_dict']['bg_mode']
        self.tracknet = get_model('TrackNet', self.tracknet_seq_len, self.bg_mode).cuda()
        self.tracknet.load_state_dict(tracknet_ckpt['model'])
        self.tracknet.eval()
        
        # Predefined colors for specific players
        self.color_list = {
            1: (0, 255, 0),   # Green for Player 1
            2: (255, 255, 0), # Yellow for Player 2
            3: (0, 0, 255),   # Red for Player 3 (default if more players are present)
            4: (255, 0, 0)    # Blue for Player 4 (default if more players are present)
        }

        # keep track of the current closest player to the shuttlecock for drawing the black bounding box
        self.current_closest = None

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

        return detections, img

    def is_within_court(self, x1, y1, x2, y2):
        xmin, ymin, xmax, ymax = self.court
        # Check if any part of the bounding box is within the court
        return not (x2 < xmin or x1 > xmax or y2 < ymin or y1 > ymax)

    def track_detect(self, detections, tracker, img):
        resultTracker = tracker.update(detections)
        player_positions = {}  # Store player ID and coordinates

        for res in resultTracker:
            x1, y1, x2, y2, id = res
            x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2  # Center of player

            

            # Assign predefined colors to players based on their ID

            # Assign a color to the player ID
            if id <= 4:
                if self.current_closest is not None and id == self.current_closest: # if the current box being drawn is the closest player's
                    color = (0,0,0) # make the bounding box black
                else: # else keep their original colour
                    self.player_colors[id] = self.color_list[id]
                    color = self.player_colors[id]

            else:
                color = self.color_list.get(id, (0, 0, 255))  # Default to Red if no specific color


            # Save player center coordinates
            player_positions[id] = (cx, cy)

            # Draw the player box and ID
            cvzone.putTextRect(img, f'Player: {id}', (x1, y1), scale=1, thickness=1, colorR=color)
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=1, colorR=color)
            cv2.circle(img, (cx, cy), 5, color, cv2.FILLED)

        return img, player_positions



    def calculate_velocity(self, player_positions, frame_diff=1):
        """ Calculate player velocity based on change in position """
        velocities = {}
        for player_id, pos in player_positions.items():
            if player_id in self.previous_player_positions:
                prev_pos = self.previous_player_positions[player_id]
                velocity = math.sqrt((pos[0] - prev_pos[0]) ** 2 + (pos[1] - prev_pos[1]) ** 2) / frame_diff
                velocities[player_id] = velocity
        return velocities


    def calculate_shuttlecock_velocity(self, previous_pos, current_pos, frame_diff=1):
        """ Calculate shuttlecock velocity """
        velocity = math.sqrt((current_pos[0] - previous_pos[0]) ** 2 + (current_pos[1] - previous_pos[1]) ** 2) / frame_diff
        return velocity


    def is_moving_toward_shuttlecock(self, player_pos, shuttlecock_pos, shuttlecock_prev_pos):
        """ Check if the player is moving toward the shuttlecock based on trajectory """
        shuttlecock_dir = (shuttlecock_pos[0] - shuttlecock_prev_pos[0], shuttlecock_pos[1] - shuttlecock_prev_pos[1])
        player_dir = (player_pos[0] - shuttlecock_pos[0], player_pos[1] - shuttlecock_pos[1])
        dot_product = player_dir[0] * shuttlecock_dir[0] + player_dir[1] * shuttlecock_dir[1]
        return dot_product > 0  # Positive means aligned with shuttlecock trajectory


    def predict_hit(self, player_positions, shuttlecock_pos, shuttlecock_velocity):
        """ Predict which player is most likely to hit the shuttlecock based on position and velocity """
        if self.previous_shuttlecock_pos is None:
            return None  # Skip if no previous shuttlecock position
        min_dist = float('inf')
        closest_player_id = None
        for player_id, pos in player_positions.items():
            if self.is_moving_toward_shuttlecock(pos, shuttlecock_pos, self.previous_shuttlecock_pos):
                distance = math.sqrt((pos[0] - shuttlecock_pos[0]) ** 2 + (pos[1] - shuttlecock_pos[1]) ** 2)
                if distance < min_dist:
                    min_dist = distance
                    closest_player_id = player_id
        return closest_player_id


    def track_shuttlecock(self, img_scaler):
        # Initialise variable to track previous closest player ID
        previous_closest_player_id = None
        
        # Implement shuttlecock tracking using TrackNet
        frame_list = generate_frames(self.capture_path)
        dataset = Shuttlecock_Trajectory_Dataset(
            seq_len=self.tracknet_seq_len, sliding_step=1, data_mode='heatmap', 
            bg_mode=self.bg_mode, frame_arr=np.array(frame_list)[:, :, :, ::-1]
        )
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)

        traj = deque(maxlen=8)  # Adjust maxlen to control the length of the trajectory
        shuttlecock_pos = None  # Initialize the shuttlecock position to None

        # Set up video writer
        result_path = os.path.join(self.result, 'results.avi')
        codec = cv2.VideoWriter_fourcc(*'XVID')
        vid_fps = int(self.capture.get(cv2.CAP_PROP_FPS))
        vid_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(result_path, codec, vid_fps, (vid_width, vid_height))

        tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

        skipped_frames = self.tracknet_seq_len - 1

        for step, (i, x) in enumerate(tqdm(data_loader)):
            x = x.float().cuda()
            with torch.no_grad():
                y_pred = self.tracknet(x).detach().cpu()

            # Predict shuttlecock location
            pred_dict = predict(i, y_pred=y_pred, img_scaler=img_scaler)
            for frame, x, y, vis in zip(pred_dict['Frame'], pred_dict['X'], pred_dict['Y'], pred_dict['Visibility']):
                actual_frame_index = frame + skipped_frames
                if vis:  # Only process visible shuttlecock locations
                    traj.append((x, y))
                    shuttlecock_pos = (x, y)
                    # Print the coordinates for debugging
                    print(f"Frame: {frame}, X: {x}, Y: {y}")

            # Player detection for the current frame
            img = frame_list[step + skipped_frames]
            detections = np.empty((0, 5))
            results = self.predict(img)
            detections, img = self.plot_boxes(results, img, detections)
            img, player_positions = self.track_detect(detections, tracker, img)

            # Compute distances between players and shuttlecock
            if shuttlecock_pos:  # Only calculate if the shuttlecock position is available
                min_dist = float('inf')
                closest_player_id = None
                
                for player_id, player_pos in player_positions.items():
                    if player_id in [1, 2, 3, 4]:  # Assuming these are valid player IDs
                        print(f"Player ID: {player_id}, Position: {player_pos}")
                        # Calculate Euclidean distance between player and shuttlecock
                        distance = math.sqrt((player_pos[0] - shuttlecock_pos[0]) ** 2 + (player_pos[1] - shuttlecock_pos[1]) ** 2)
                        if distance < min_dist:
                            min_dist = distance
                            closest_player_id = player_id

                # Print or store the closest player
                if closest_player_id is not None:
                    print(f"Player {closest_player_id} is closest to the shuttlecock at frame {frame}")
                    # for drawing the black bounding box later
                    self.current_closest = closest_player_id

                    # Change the color of the current closest player to black
                    if closest_player_id in self.player_colors:
                        self.player_colors[closest_player_id] = (0, 0, 0)  # Set to black

                    # Revert the previous closest player to their original color
                    if previous_closest_player_id is not None and previous_closest_player_id != closest_player_id:
                        if previous_closest_player_id in self.player_colors:
                            self.player_colors[previous_closest_player_id] = self.color_list[previous_closest_player_id]

                    # Update the previous closest player ID
                    previous_closest_player_id = closest_player_id

            # Draw the shuttlecock trajectory on the image
            img = draw_traj(img, traj, radius=3, color='red')

            # Write the frame to the video file
            out.write(img)

        # Release the video writer
        out.release()

        return result_path, shuttlecock_pos










    def __call__(self):
        self.capture = cv2.VideoCapture(self.capture_path)  # Initialize the video capture
        assert self.capture.isOpened(), "Failed to open video capture"

        # Get video frame dimensions
        vid_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate image scaler based on expected dimensions (e.g., 512x288 for TrackNet)
        w_scaler = vid_width / WIDTH
        h_scaler = vid_height / HEIGHT
        img_scaler = (w_scaler, h_scaler)

        print(f"Image scaler: {img_scaler}")

        # Initialize the SORT tracker
        tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

        # Call the shuttlecock tracking function and retrieve the result
        result_path = self.track_shuttlecock(img_scaler)
        print(f"Result video saved at {result_path}")

        # Video processing loop
        frame_count = 0
        while True:
            ret, img = self.capture.read()
            if not ret:
                break  # End of the video

            # Player detection for the current frame
            detections = np.empty((0, 5))
            results = self.predict(img)
            detections, img = self.plot_boxes(results, img, detections)

            # Track players using SORT tracker
            img, player_positions = self.track_detect(detections, tracker, img)

            # Check if we have a previous shuttlecock position to calculate velocity
            if self.previous_shuttlecock_pos:
                shuttlecock_velocity = self.calculate_shuttlecock_velocity(self.previous_shuttlecock_pos, shuttlecock_pos)

                # Predict which player is most likely to hit the shuttlecock
                closest_player_id = self.predict_hit(player_positions, shuttlecock_pos, shuttlecock_velocity)

                if closest_player_id is not None:
                    print(f"Player {closest_player_id} is most likely to hit the shuttlecock.")


            # Update the previous player positions and shuttlecock position for the next frame 
            # self.previous_player_positions = player_positions
            # self.previous_shuttlecock_pos = shuttlecock_pos

            frame_count += 1

        self.capture.release()
        print("Tracking finished.")






# Example usage
detector = ObjectDetection(capture="test.mp4", result='result', court=(450, 390, 1500, 1000), tracknet_file='ckpts/TrackNet_best.pt')
detector()





