from flask import Flask, request, render_template, jsonify, url_for
import os
from integration import ObjectDetection  # Import your ObjectDetection class

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static/results'  # Ensure results are served from a static folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

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
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Run the object detection
        detector = ObjectDetection(
            capture=filepath, 
            result=RESULT_FOLDER, 
            court=(450, 390, 1500, 1000), 
            tracknet_file='ckpts/TrackNet_best.pt'
        )
        result_path = detector()

        if result_path is None:
            return jsonify({'error': 'Error processing video'}), 500

        # Return the correct URL for the video
        video_url = url_for('static', filename=f'results/{os.path.basename(result_path)}')
        print(f"Video URL: {video_url}")  # Debugging statement
        return jsonify({'video_url': video_url})

if __name__ == '__main__':
    app.run(debug=True)