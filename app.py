from flask import Flask, request, jsonify
import mediapipe as mp
import numpy as np
import cv2

app = Flask(__name__)
pose = mp.solutions.pose.Pose(static_image_mode=True)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    result = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    keypoints = {}

    if result.pose_landmarks:
        h, w, _ = image.shape
        for i, lm in enumerate(result.pose_landmarks.landmark):
            keypoints[f"{mp.solutions.pose.PoseLandmark(i).name.lower()}"] = {
                "x": int(lm.x * w),
                "y": int(lm.y * h),
                "z": lm.z,
                "visibility": lm.visibility
            }

    return jsonify({"keypoints": keypoints})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
