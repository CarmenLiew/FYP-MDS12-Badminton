from flask import Flask, request, render_template, jsonify, url_for
from flask_socketio import SocketIO, emit
import os
import ffmpeg
from integration import ObjectDetection  # Import your ObjectDetection class
import threading
import shutil

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static/results'  # Ensure results are served from a static folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def run_object_detection(filepath, result_folder, court, tracknet_file):
    """
    Run object detection on the specified video file and save the results.

    Args:
        filepath (str): The path to the input video file for object detection.
        result_folder (str): The folder where the results will be saved.
        court (str): The type of court (e.g., badminton court) used in the detection.
        tracknet_file (str): The path to the TrackNet model file for shuttlecock tracking.

    Returns:
        str: The path to the result video file after processing.
    """
    
    def progress_callback(progress):
        """
        Callback function to report progress of the object detection process.

        Args:
            progress (int): The current progress percentage (0 to 100).
        """
        print(f'Progress: {progress}%')  # Print the progress for debugging
        socketio.emit('progress', {'progress': progress})  # Emit the progress via socket for real-time updates

    # Initialize the ObjectDetection class with provided parameters
    detector = ObjectDetection(
        capture=filepath,          # Path to the input video file
        result=result_folder,      # Directory to save the results
        court=court,               # Court type for detection
        tracknet_file=tracknet_file,  # Path to the TrackNet model file
        progress_callback=progress_callback  # Set the progress callback function
    )
    
    # Run the object detection process and store the result path
    result_path = detector()  # Call the detector instance to perform detection

    return result_path  # Return the path of the result video file


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        if os.path.exists(RESULT_FOLDER):
            shutil.rmtree(RESULT_FOLDER)
        os.makedirs(RESULT_FOLDER)

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Run the object detection
        result_path = run_object_detection(
            filepath, 
            RESULT_FOLDER, 
            court=(450, 390, 1500, 1000), 
            tracknet_file='ckpts/TrackNet_best.pt'
        )

        if result_path is None:
            return jsonify({'error': 'Error processing video'}), 500

        # Convert the AVI result to MP4 using ffmpeg
        mp4_result_path = os.path.splitext(result_path)[0] + ".mp4"
        try:
            ffmpeg.input(result_path).output(mp4_result_path).run()
        except Exception as e:
            return jsonify({'error': f'Video conversion failed: {str(e)}'}), 500

        # Return the correct URL for the video in MP4 format
        video_url = url_for('static', filename=f'results/{os.path.basename(mp4_result_path)}')
        print(f"Video URL: {video_url}")  # Debugging statement
        return jsonify({'video_url': video_url})

@app.route('/result')
def result():
    video_url = request.args.get('video_url')
    return render_template('result.html', video_url=video_url)

if __name__ == '__main__':
    socketio.run(app, debug=True)
