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

def classify_fitzpatrick_scale(r, g, b):
    """Classify skin tone using Fitzpatrick scale (I-VI)"""
    brightness = (r + g + b) / 3
    
    if brightness > 210:
        return {"scale": "I", "description": "Very Fair - Always burns, never tans"}
    elif brightness > 190:
        return {"scale": "II", "description": "Fair - Usually burns, tans minimally"}
    elif brightness > 160:
        return {"scale": "III", "description": "Light - Sometimes burns, gradually tans"}
    elif brightness > 130:
        return {"scale": "IV", "description": "Moderate - Rarely burns, always tans"}
    elif brightness > 100:
        return {"scale": "V", "description": "Dark - Very rarely burns, tans very easily"}
    else:
        return {"scale": "VI", "description": "Very Dark - Never burns, deeply pigmented"}

def detect_undertone(r, g, b):
    """Detect skin undertone (warm, cool, neutral)"""
    # Convert to different color spaces for better undertone detection
    yellow_factor = (r + g) / (2 * b) if b > 0 else 1
    red_factor = r / g if g > 0 else 1
    
    # Analyze undertone based on color ratios
    if yellow_factor > 1.15 and red_factor > 1.1:
        return {"undertone": "Warm", "description": "Golden, yellow, or peach undertones"}
    elif yellow_factor < 0.95 and (b / r) > 0.85:
        return {"undertone": "Cool", "description": "Pink, red, or blue undertones"}
    else:
        return {"undertone": "Neutral", "description": "Balanced warm and cool undertones"}

def get_descriptive_category(r, g, b):
    """Get descriptive skin tone category for fashion/beauty context"""
    brightness = (r + g + b) / 3
    
    # More granular categories based on fashion/beauty industry standards
    if brightness > 220:
        return "Porcelain"
    elif brightness > 205:
        return "Fair"
    elif brightness > 190:
        return "Light"
    elif brightness > 175:
        return "Light-Medium"
    elif brightness > 160:
        return "Medium"
    elif brightness > 145:
        return "Medium-Tan"
    elif brightness > 130:
        return "Tan"
    elif brightness > 115:
        return "Deep"
    elif brightness > 100:
        return "Rich"
    else:
        return "Ebony"

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
        
        # Get all classification systems
        fitzpatrick = classify_fitzpatrick_scale(r, g, b)
        undertone = detect_undertone(r, g, b)
        descriptive = get_descriptive_category(r, g, b)
        
        # Detailed simple tone categories
        brightness = (r + g + b) / 3
        
        if brightness > 220:
            simple_tone = "Very Light"
        elif brightness > 200:
            simple_tone = "Light"
        elif brightness > 180:
            simple_tone = "Fair"
        elif brightness > 160:
            simple_tone = "Light Medium"
        elif brightness > 140:
            simple_tone = "Medium"
        elif brightness > 120:
            simple_tone = "Medium Dark"
        elif brightness > 100:
            simple_tone = "Dark"
        elif brightness > 80:
            simple_tone = "Deep"
        else:
            simple_tone = "Very Dark"

        return {
            "hex": hex_color, 
            "rgb": [int(r), int(g), int(b)], 
            "tone_category": simple_tone,  # Keep for backward compatibility
            "descriptive_category": descriptive,
            "fitzpatrick_scale": fitzpatrick,
            "undertone": undertone,
            "brightness_score": round((r + g + b) / 3, 1)
        }
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
