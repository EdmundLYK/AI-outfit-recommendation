from flask import Flask, request, jsonify
import mediapipe as mp
import numpy as np
import cv2

app = Flask(__name__)
pose = mp.solutions.pose.Pose(static_image_mode=True)


def calc_distance(p1, p2):
    """Helper: calculate Euclidean distance between 2 keypoints"""
    if not p1 or not p2:
        return None
    dx = p1["x"] - p2["x"]
    dy = p1["y"] - p2["y"]
    return int(np.sqrt(dx * dx + dy * dy))


def extract_skin_color(image, landmarks, h, w):
    """Extract average skin color from face region"""
    try:
        face_points = []
        for landmark_name in ['nose', 'left_eye', 'right_eye']:
            if landmark_name in landmarks:
                lm = landmarks[landmark_name]
                face_points.append((int(lm['x']), int(lm['y'])))

        if len(face_points) == 0:
            return {
                "hex": None,
                "rgb": None,
                "tone_category": "Unknown",
                "error": "No face landmarks for skin detection"
            }

        center_x = int(np.mean([p[0] for p in face_points]))
        center_y = int(np.mean([p[1] for p in face_points]))

        region_size = 30
        x1 = max(0, center_x - region_size)
        x2 = min(w, center_x + region_size)
        y1 = max(0, center_y - region_size)
        y2 = min(h, center_y + region_size)

        face_region = image[y1:y2, x1:x2]
        if face_region.size == 0:
            return {
                "hex": None,
                "rgb": None,
                "tone_category": "Unknown",
                "error": "Empty region for skin color"
            }

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

        return {
            "hex": hex_color,
            "rgb": [int(r), int(g), int(b)],
            "tone_category": tone
        }
    except Exception as e:
        return {
            "hex": None,
            "rgb": None,
            "tone_category": "Unknown",
            "error": f"Skin extraction failed: {str(e)}"
        }


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

            if keypoint_name in ['nose', 'left_shoulder', 'right_shoulder', 'left_ankle', 'right_ankle']:
                analysis_accuracy[keypoint_name] = {
                    "visibility": round(lm.visibility, 3),
                    "confidence": "High" if lm.visibility > 0.8 else "Medium" if lm.visibility > 0.5 else "Low"
                }

        # Extract skin color
        skin_analysis = extract_skin_color(image, keypoints, h, w)

        # Body width: shoulders & waist
        shoulder_width_px = None
        waist_width_px = None
        shoulder_waist_ratio = None

        if "left_shoulder" in keypoints and "right_shoulder" in keypoints:
            shoulder_width_px = calc_distance(keypoints["left_shoulder"], keypoints["right_shoulder"])

        if "left_hip" in keypoints and "right_hip" in keypoints:
            waist_width_px = calc_distance(keypoints["left_hip"], keypoints["right_hip"])

        if shoulder_width_px and waist_width_px and waist_width_px > 0:
            shoulder_waist_ratio = round(shoulder_width_px / waist_width_px, 2)

        body_width_result = {
            "shoulder_px": shoulder_width_px if shoulder_width_px else 0,
            "waist_px": waist_width_px if waist_width_px else 0,
            "shoulder_to_waist_ratio": shoulder_waist_ratio if shoulder_waist_ratio else 0
        }

        # Model confidence
        key_landmarks = ['nose', 'left_shoulder', 'right_shoulder', 'left_ankle', 'right_ankle']
        visible_landmarks = [kp for kp in keypoints if kp in key_landmarks and keypoints[kp]['visibility'] > 0.5]
        overall_confidence = len(visible_landmarks) / len(key_landmarks)

        return jsonify({
            "keypoints": keypoints,
            "skin_color": skin_analysis,
            "body_width": body_width_result,
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
            "skin_color": {
                "hex": None,
                "rgb": None,
                "tone_category": "Unknown",
                "error": "No face detected"
            },
            "body_width": {
                "shoulder_px": 0,
                "waist_px": 0,
                "shoulder_to_waist_ratio": 0
            },
            "model_accuracy": {
                "overall_confidence": 0.0,
                "key_landmarks_detected": 0,
                "total_key_landmarks": 5
            }
        })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
