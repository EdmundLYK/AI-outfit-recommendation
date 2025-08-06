from flask import Flask, request, jsonify
import mediapipe as mp
import numpy as np
import cv2

app = Flask(__name__)
pose = mp.solutions.pose.Pose(static_image_mode=True)

def calc_distance(p1, p2):
    if not p1 or not p2:
        return None
    dx = p1["x"] - p2["x"]
    dy = p1["y"] - p2["y"]
    return int(np.sqrt(dx * dx + dy * dy))

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
        for i, lm in enumerate(result.pose_landmarks.landmark):
            keypoint_name = mp.solutions.pose.PoseLandmark(i).name.lower()
            keypoints[keypoint_name] = {
                "x": int(lm.x * w),
                "y": int(lm.y * h),
                "z": lm.z,
                "visibility": lm.visibility
            }

            if keypoint_name in ['nose', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']:
                analysis_accuracy[keypoint_name] = {
                    "visibility": round(lm.visibility, 3),
                    "confidence": "High" if lm.visibility > 0.8 else "Medium" if lm.visibility > 0.5 else "Low"
                }

        # Measurements
        shoulder_width_px = calc_distance(keypoints.get("left_shoulder"), keypoints.get("right_shoulder"))
        waist_width_px = calc_distance(keypoints.get("left_hip"), keypoints.get("right_hip"))

        # ✅ Improved torso length (try left, right, or midpoint)
        torso_length_px = None
        if "left_shoulder" in keypoints and "left_hip" in keypoints:
            torso_length_px = calc_distance(keypoints["left_shoulder"], keypoints["left_hip"])
        elif "right_shoulder" in keypoints and "right_hip" in keypoints:
            torso_length_px = calc_distance(keypoints["right_shoulder"], keypoints["right_hip"])
        elif all(k in keypoints for k in ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]):
            mid_shoulder = {
                "x": (keypoints["left_shoulder"]["x"] + keypoints["right_shoulder"]["x"]) // 2,
                "y": (keypoints["left_shoulder"]["y"] + keypoints["right_shoulder"]["y"]) // 2
            }
            mid_hip = {
                "x": (keypoints["left_hip"]["x"] + keypoints["right_hip"]["x"]) // 2,
                "y": (keypoints["left_hip"]["y"] + keypoints["right_hip"]["y"]) // 2
            }
            torso_length_px = calc_distance(mid_shoulder, mid_hip)

        # Body shape classification
        body_shape = None
        ratio = None
        if shoulder_width_px and waist_width_px:
            ratio = shoulder_width_px / waist_width_px if waist_width_px > 0 else None
            if ratio and ratio > 1.2:
                body_shape = "Inverted Triangle (broad shoulders)"
            elif ratio and ratio < 0.8:
                body_shape = "Pear (narrow shoulders, wider waist)"
            else:
                body_shape = "Balanced (proportional shoulders and waist)"
        else:
            body_shape = "Unknown"

        skin_analysis = extract_skin_color(image, keypoints, h, w)

        # Model confidence
        key_landmarks = ['nose', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        visible_landmarks = [kp for kp in key_landmarks if kp in keypoints and keypoints[kp]['visibility'] > 0.5]
        overall_confidence = len(visible_landmarks) / len(key_landmarks)

        return jsonify({
            "keypoints": keypoints,
            "skin_color": skin_analysis,
            "body_width": {
                "shoulder_px": shoulder_width_px,
                "waist_px": waist_width_px,
                "shoulder_to_waist_ratio": round(ratio, 2) if ratio else None,
                "torso_length_px": torso_length_px,
                "body_shape": body_shape
            },
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
            "body_width": {"body_shape": "Unknown"},
            "model_accuracy": {
                "overall_confidence": 0.0,
                "key_landmarks_detected": 0,
                "total_key_landmarks": 5
            }
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
