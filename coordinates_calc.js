//this is used in the n8n coordinates calc node

const skinColor = items[0].json.skin_color || {};
const bodyWidth = items[0].json.body_width || {};
const keypoints = items[0].json.keypoints || {};

const MIN_VIS = 0.5;

function hasPt(p) {
  return p && typeof p.x === "number" && typeof p.y === "number";
}

function visible(p) {
  return (
    hasPt(p) && (typeof p.visibility !== "number" || p.visibility >= MIN_VIS)
  );
}

function dist(p1, p2) {
  if (!hasPt(p1) || !hasPt(p2)) return null;
  const dx = p1.x - p2.x;
  const dy = p1.y - p2.y;
  return Math.round(Math.sqrt(dx * dx + dy * dy));
}

/* -------- Torso length (with fallbacks) -------- */
let torsoLengthPx = null;
if (visible(keypoints.left_shoulder) && visible(keypoints.left_hip)) {
  torsoLengthPx = dist(keypoints.left_shoulder, keypoints.left_hip);
}
if (
  !torsoLengthPx &&
  visible(keypoints.right_shoulder) &&
  visible(keypoints.right_hip)
) {
  torsoLengthPx = dist(keypoints.right_shoulder, keypoints.right_hip);
}
if (
  !torsoLengthPx &&
  visible(keypoints.left_shoulder) &&
  visible(keypoints.right_shoulder) &&
  visible(keypoints.left_hip) &&
  visible(keypoints.right_hip)
) {
  const midShoulder = {
    x: (keypoints.left_shoulder.x + keypoints.right_shoulder.x) / 2,
    y: (keypoints.left_shoulder.y + keypoints.right_shoulder.y) / 2,
  };
  const midHip = {
    x: (keypoints.left_hip.x + keypoints.right_hip.x) / 2,
    y: (keypoints.left_hip.y + keypoints.right_hip.y) / 2,
  };
  torsoLengthPx = dist(midShoulder, midHip);
}

/* -------- Shoulder & Waist (Hip) width -------- */
// Start with server values if present
let shoulderPx = bodyWidth?.shoulder_px ?? null;
// Your API uses hip_px; treat it as waist width
let waistPx = bodyWidth?.waist_px ?? bodyWidth?.hip_px ?? null;

// Compute from keypoints if missing
if (
  !shoulderPx &&
  visible(keypoints.left_shoulder) &&
  visible(keypoints.right_shoulder)
) {
  shoulderPx = dist(keypoints.left_shoulder, keypoints.right_shoulder);
}
if (!waistPx && visible(keypoints.left_hip) && visible(keypoints.right_hip)) {
  waistPx = dist(keypoints.left_hip, keypoints.right_hip);
}

/* -------- Ratio & Body Shape -------- */
let ratio = null;
let bodyShape = "Unknown";
if (shoulderPx && waistPx && waistPx > 0) {
  ratio = shoulderPx / waistPx;
  if (ratio > 1.2) {
    bodyShape = "Inverted Triangle (broad shoulders)";
  } else if (ratio < 0.8) {
    bodyShape = "Pear (narrow shoulders, wider waist)";
  } else {
    bodyShape = "Balanced (proportional shoulders & waist)";
  }
}

return [
  {
    json: {
      // Focused features
      skin_tone: skinColor.tone_category || "Unknown",
      skin_hex: skinColor.hex || null,
      body_shape: bodyShape,
      shoulder_px: shoulderPx ?? null,
      waist_px: waistPx ?? null,
      shoulder_to_waist_ratio: ratio != null ? ratio.toFixed(2) : null,
      torso_length_px: torsoLengthPx ?? null,

      // Raw for debugging
      skin_color: skinColor,
      body_width: bodyWidth,

      // Pass through model accuracy
      model_accuracy: items[0].json.model_accuracy || {},
    },
  },
];
