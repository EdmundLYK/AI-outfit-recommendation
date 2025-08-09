from flask import Flask, request, jsonify
import mediapipe as mp
import numpy as np
import cv2

app = Flask(__name__)
pose = mp.solutions.pose.Pose(static_image_mode=True)

# ✅ drop these from the output entirely
EXCLUDED_LANDMARKS = {
    "left_ankle", "left_foot_index", "left_heel", "left_knee",
    "right_ankle", "right_foot_index", "right_heel", "right_knee"
}

def calc_distance(p1, p2):
    if not p1 or not p2:
        return None
    try:
        dx = (p1.get("x", 0) - p2.get("x", 0))
        dy = (p1.get("y", 0) - p2.get("y", 0))
        return int(np.sqrt(dx * dx + dy * dy))
    except Exception:
        return None

def extract_skin_color(image, landmarks, h, w):
    try:
        face_points = []
        for landmark_name in ['nose', 'left_eye', 'right_eye']:
            if landmark_name in landmarks:
                lm = landmarks[landmark_name]
                face_points.append((int(lm['x']), int(lm['y'])))

        if not face_points:
            return {"error": "No face landmarks for skin detection"}

        center_x = int(np.mean([p[0] for p in face_points]))
        center_y = int(np.mean([p[1] for p in face_points]))

        region_size = 30
        x1, x2 = max(0, center_x - region_size), min(w, center_x + region_size)
        y1, y2 = max(0, center_y - region_size), min(h, center_y + region_size)

        face_region = image[y1:y2, x1:x2]
        if face_region.size == 0:
            return {"error": "Empty region for skin color"}

        face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
        avg_color = np.mean(face_rgb, axis=(0, 1))

        hex_color = '#{:02x}{:02x}{:02x}'.format(*[int(c) for c in avg_color])
        r, g, b = avg_color
        if r > 220 and g > 200 and b > 180:
            tone = "Light"
        elif r > 180 and g > 140 and b > 100:
            tone = "Medium"
        elif r > 120 and g > 80 and b > 60:
            tone = "Medium-Dark"
        else:
            tone = "Dark"

        return {"hex": hex_color, "rgb": [int(r), int(g), int(b)], "tone_category": tone}
    except Exception as e:
        return {"error": f"Skin extraction failed: {str(e)}"}

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    result = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    keypoints = {}
    analysis_accuracy = {}

    if result.pose_landmarks:
        h, w, _ = image.shape

        # Only consider these as "key landmarks" for accuracy (upper body)
        key_landmarks = ['nose', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']

        for i, lm in enumerate(result.pose_landmarks.landmark):
            name = mp.solutions.pose.PoseLandmark(i).name.lower()
            # 🚫 skip excluded lower-body points
            if name in EXCLUDED_LANDMARKS:
                continue

            keypoints[name] = {
                "x": int(lm.x * w),
                "y": int(lm.y * h),
                "z": lm.z,
                "visibility": lm.visibility
            }

            if name in key_landmarks:
                analysis_accuracy[name] = {
                    "visibility": round(lm.visibility, 5),
                    "confidence": "High" if lm.visibility > 0.8 else "Medium" if lm.visibility > 0.5 else "Low"
                }

        # --- Body width (kept as-is) ---
        body_width_data = {"error": "Missing landmarks"}
        if keypoints.get('left_shoulder') and keypoints.get('right_shoulder'):
            shoulder_width = calc_distance(keypoints['left_shoulder'], keypoints['right_shoulder'])
            if keypoints.get('left_hip') and keypoints.get('right_hip'):
                hip_width = calc_distance(keypoints['left_hip'], keypoints['right_hip'])
                if shoulder_width and hip_width and hip_width > 0:
                    ratio = round(shoulder_width / hip_width, 2)
                    body_width_data = {
                        "shoulder_px": shoulder_width,
                        "hip_px": hip_width,
                        "shoulder_to_hip_ratio": ratio,
                        "body_shape": "Inverted Triangle" if ratio > 1.2 else "Pear" if ratio < 0.8 else "Rectangle"
                    }

        # accuracy score (upper-body only)
        visible_landmarks = [kp for kp in key_landmarks if kp in keypoints and keypoints[kp]['visibility'] > 0.5]
        overall_confidence = len(visible_landmarks) / len(key_landmarks)

        return jsonify({
            "keypoints": keypoints,
            "skin_color": extract_skin_color(image, keypoints, h, w),
            "body_width": body_width_data,
            "model_accuracy": {
                "overall_confidence": round(overall_confidence, 2),
                "key_landmarks_detected": len(visible_landmarks),
                "total_key_landmarks": len(key_landmarks),
                "landmark_accuracy": analysis_accuracy
            }
        })

    else:
        return jsonify({
            "keypoints": {},
            "skin_color": {"error": "No face detected"},
            "body_width": {"error": "No pose detected"},
            "model_accuracy": {
                "overall_confidence": 0.0,
                "key_landmarks_detected": 0,
                "total_key_landmarks": 5,
                "landmark_accuracy": {}
            }
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
