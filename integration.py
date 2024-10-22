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
    def __init__(self, capture, result, court, tracknet_file, scale_factor=59.7, progress_callback=None):
        self.capture_path = capture  # Store the path to the video file
        self.result = result
        self.court = court  # court coordinates (xmin, ymin, xmax, ymax)
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names
        self.player_colors = {}  # dictionary to store all colours for the players
        self.previous_player_positions = {}  # Store player positions for velocity calculation
        self.previous_shuttlecock_pos = None  # Track previous shuttlecock position
        self.progress_callback = progress_callback # Add progress callback

        # Load TrackNet model for shuttlecock tracking
        tracknet_ckpt = torch.load(tracknet_file)
        self.tracknet_seq_len = tracknet_ckpt['param_dict']['seq_len']
        self.bg_mode = tracknet_ckpt['param_dict']['bg_mode']
        self.tracknet = get_model('TrackNet', self.tracknet_seq_len, self.bg_mode).cuda()
        self.tracknet.load_state_dict(tracknet_ckpt['model'])
        self.tracknet.eval()
        
        # Predefined colors for specific players
        self.default_color =  (255, 0, 0) # set default colour to dark blue

        # keep track of the current closest player to the shuttlecock for drawing the black bounding box
        self.current_closest = None

        self.scale_factor = scale_factor

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
        # Update the tracker with the current detections and retrieve the results
        resultTracker = tracker.update(detections)
        
        # Initialize a dictionary to store player ID and their corresponding center coordinates
        player_positions = {}

        # Iterate over the tracking results to process each detected player
        for res in resultTracker:
            # Unpack the results into bounding box coordinates and player ID
            x1, y1, x2, y2, id = res
            
            # Convert the bounding box coordinates and ID to integers
            x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
            
            # Calculate the width and height of the bounding box
            w, h = x2 - x1, y2 - y1
            
            # Calculate the center coordinates (cx, cy) of the player based on the bounding box
            cx, cy = x1 + w // 2, y1 + h // 2  # Center of player

            # Assign a color to the player based on their ID
            if id <= 4:  # Check if the player ID is within the expected range
                # Check if the current player is the closest player to the shuttlecock
                if self.current_closest is not None and id == self.current_closest:
                    color = (0, 0, 0)  # Set the bounding box color to black for the closest player
                else:
                    # If not the closest, use the player's original color stored in player_colors
                    self.player_colors[id] = self.default_color
                    color = self.player_colors[id]  # Get the color for the player

                # Save the player's center coordinates in the player_positions dictionary
                player_positions[id] = (cx, cy)

                # Draw the player's ID as text on the image at the top-left corner of the bounding box
                cvzone.putTextRect(img, f'Player: {id}', (x1, y1), scale=1, thickness=1, colorR=color)
                # Draw a rectangle around the player's bounding box with rounded corners
                cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=1, colorR=color)
                # Draw a filled circle at the center of the player for visual reference
                cv2.circle(img, (cx, cy), 5, color, cv2.FILLED)

        # Return the annotated image and the dictionary of player positions
        return img, player_positions


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
        # Generate a list of frames from the video capture path
        frame_list = generate_frames(self.capture_path)
        
        # Create a dataset for shuttlecock trajectory using the frames generated
        dataset = Shuttlecock_Trajectory_Dataset(
            seq_len=self.tracknet_seq_len, sliding_step=1, data_mode='heatmap',
            bg_mode=self.bg_mode, frame_arr=np.array(frame_list)[:, :, :, ::-1]  # Convert BGR to RGB
        )
        
        # Create a DataLoader for batching and shuffling the dataset
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)

        traj = deque(maxlen=8)  # Initialize trajectory buffer with a maximum length of 8
        shuttlecock_pos = None  # Initialize the shuttlecock position to None

        # Set up video writer for saving output video with tracked movements
        result_path = os.path.join(self.result, 'results.avi')
        codec = cv2.VideoWriter_fourcc(*'XVID')
        vid_fps = int(self.capture.get(cv2.CAP_PROP_FPS))  # Get video frame rate
        vid_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get video width
        vid_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get video height
        out = cv2.VideoWriter(result_path, codec, vid_fps, (vid_width, vid_height))  # Create video writer object

        # Initialize SORT tracker for tracking players
        tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

        skipped_frames = self.tracknet_seq_len - 1  # Determine how many frames to skip for accurate tracking

        # Track previous positions and distances for velocity and distance tracking
        previous_player_positions = {}
        total_distances = {1: 0, 2: 0, 3: 0, 4: 0}  # Dictionary to track total distance for each player
        player_speeds = {1: 0, 2: 0, 3: 0, 4: 0}  # Dictionary to track velocity for each player

        # Initialize a deque to store player velocities for smoothing calculations
        velocity_history = {1: deque(maxlen=5), 2: deque(maxlen=5), 3: deque(maxlen=5), 4: deque(maxlen=5)}

        total_frames = len(data_loader)  # Get the total number of frames in the data loader
        for step, (i, x) in enumerate(tqdm(data_loader)):  # Iterate over each frame in the data loader
            x = x.float().cuda()  # Convert the input tensor to float and move to GPU
            with torch.no_grad():  # Disable gradient tracking for inference
                y_pred = self.tracknet(x).detach().cpu()  # Get predictions from TrackNet

            # Predict shuttlecock location based on model predictions
            pred_dict = predict(i, y_pred=y_pred, img_scaler=img_scaler)
            for frame, x, y, vis in zip(pred_dict['Frame'], pred_dict['X'], pred_dict['Y'], pred_dict['Visibility']):
                actual_frame_index = frame + skipped_frames  # Adjust frame index considering skipped frames
                if vis:  # Only process visible shuttlecock locations
                    traj.append((x, y))  # Append the current shuttlecock position to the trajectory
                    shuttlecock_pos = (x, y)  # Update the current position of the shuttlecock
                    # Print the coordinates for debugging
                    print(f"Frame: {frame}, X: {x}, Y: {y}")

            # Player detection for the current frame
            img = frame_list[step + skipped_frames]  # Get the current frame image
            detections = np.empty((0, 5))  # Initialize empty detections array
            results = self.predict(img)  # Get detection results for the current frame
            detections, img = self.plot_boxes(results, img, detections)  # Plot detected boxes on the image
            img, player_positions = self.track_detect(detections, tracker, img)  # Track detected players in the image

            # Compute distances between players and the shuttlecock
            if shuttlecock_pos:  # Only calculate if the shuttlecock position is available
                min_dist = float('inf')  # Initialize minimum distance to infinity
                closest_player_id = None  # Variable to track the closest player ID

                for player_id, player_pos in player_positions.items():
                    if player_id in [1, 2, 3, 4]:  # Assuming these are valid player IDs
                        print(f"Player ID: {player_id}, Position: {player_pos}")

                        # Calculate Euclidean distance between player and shuttlecock
                        distance = math.sqrt((player_pos[0] - shuttlecock_pos[0]) ** 2 + (player_pos[1] - shuttlecock_pos[1]) ** 2)
                        if distance < min_dist:  # Update if this player is closer than the previous closest
                            min_dist = distance
                            closest_player_id = player_id

                        # Calculate the total distance moved by the player
                        if player_id in previous_player_positions:  # Check if we have a previous position to compare
                            prev_pos = previous_player_positions[player_id]
                            frame_distance = math.sqrt((player_pos[0] - prev_pos[0]) ** 2 + (player_pos[1] - prev_pos[1]) ** 2)
                            total_distances[player_id] += frame_distance / self.scale_factor  # Convert to meters

                            # Calculate the velocity and add to history
                            player_velocity = (frame_distance * vid_fps) / self.scale_factor  # Convert to meters
                            velocity_history[player_id].append(player_velocity)  # Store the velocity in history

                            # Compute the average velocity from history for smoothing
                            if velocity_history[player_id]:
                                player_speeds[player_id] = sum(velocity_history[player_id]) / len(velocity_history[player_id])

                        # Update previous position for the player
                        previous_player_positions[player_id] = player_pos

                # Print or store the closest player
                if closest_player_id is not None:
                    print(f"Player {closest_player_id} is closest to the shuttlecock at frame {frame}")
                    self.current_closest = closest_player_id  # Update the current closest player

                    # Change the color of the current closest player to black for highlighting
                    if closest_player_id in self.player_colors:
                        self.player_colors[closest_player_id] = (0, 0, 0)  # Set to black

                    # Revert the previous closest player to their original color
                    if previous_closest_player_id is not None and previous_closest_player_id != closest_player_id:
                        if previous_closest_player_id in self.player_colors:
                            self.player_colors[previous_closest_player_id] = self.default_color  # Restore original color

                    # Update the previous closest player ID
                    previous_closest_player_id = closest_player_id

            # Draw the shuttlecock trajectory on the image
            img = draw_traj(img, traj, radius=3, color='red')  # Draw the trajectory in red

            # Display the player stats (velocity and total distance moved)
            for player_id in [1, 2, 3, 4]:
                player_text = f"Player {player_id}"
                speed_text = f"Speed: {player_speeds[player_id]:.2f} m/s"  # Changed to m/s
                distance_text = f"Distance: {total_distances[player_id]:.2f} m"  # Changed to m
                print(speed_text)  # Print speed for debugging
                print(distance_text)  # Print distance for debugging

                # Get the width of the image
                img_height, img_width = img.shape[:2]

                # Set a margin from the right side for text placement
                margin = 20

                # Calculate the positions for the top-right placement of text
                text_position_x = img_width - margin
                text_position_y_speed = 30 + player_id * 30  # Adjust for speed text placement
                text_position_y_distance = 50 + player_id * 30  # Adjust for distance text placement

                # Combine speed and distance into one string for display
                combined_text = f"{player_text} {speed_text} | {distance_text}"

                # Set font parameters for drawing text
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_thickness = 2

                # Get text size (width, height) and the baseline for proper placement
                (text_width, text_height), baseline = cv2.getTextSize(combined_text, font, font_scale, font_thickness)

                # Define the top-left and bottom-right coordinates for the rectangle background of the text
                top_left = (text_position_x - text_width - 20, text_position_y_speed - text_height - 10)  # Add padding to fit the text
                bottom_right = (text_position_x + 20, text_position_y_speed + 10)

                # Draw a filled black rectangle for the background to improve text visibility
                cv2.rectangle(img, top_left, bottom_right, (0, 0, 0), thickness=cv2.FILLED)

                # Draw the text on the image
                cv2.putText(img, combined_text, (text_position_x - text_width, text_position_y_speed), font, font_scale, (255, 255, 255), font_thickness)

            # Write the frame with annotations to the output video
            out.write(img)

            # Update progress
            if self.progress_callback:
                # Calculate the progress percentage based on the current step and total number of frames
                progress = int((step + 1) / total_frames * 100)
                
                # Invoke the progress callback function to notify about the current progress
                self.progress_callback(progress)

            # Release the video writer
            out.release()  # Ensure the video writer is properly closed to finalize the output file

            # Return the path to the resulting video file
            return result_path  # Return the path where the processed video is saved


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

        
        print("Tracking finished.")

        return result_path

# Example usage
# detector = ObjectDetection(capture="test.mp4", result='result', court=(450, 390, 1500, 1000), tracknet_file='ckpts/TrackNet_best.pt')
# detector()





