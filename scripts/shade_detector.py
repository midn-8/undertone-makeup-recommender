import cv2
import numpy as np


LIGHT_THRESHOLD = 110  
MEDIUM_THRESHOLD = 60 

PATCH_SIZE = 8 

def extract_skin_lab(frame_bgr, landmarks, indices):
    h, w = frame_bgr.shape[:2]
    all_pixels = []

    for idx in indices:
        lm = landmarks[idx]
        x, y = int(lm.x * w), int(lm.y * h)

        y_min, y_max = max(0, y - PATCH_SIZE), min(h, y + PATCH_SIZE)
        x_min, x_max = max(0, x - PATCH_SIZE), min(w, x + PATCH_SIZE)
        
        patch = frame_bgr[y_min:y_max, x_min:x_max]
        if patch.size == 0: continue

        lab_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
  
        all_pixels.append(lab_patch.reshape(-1, 3))

    if not all_pixels: return None


    combined_pixels = np.vstack(all_pixels)

    median_lab = np.median(combined_pixels, axis=0)
    
    return {
        "L": float(median_lab[0]),
        "A": float(median_lab[1]),
        "B": float(median_lab[2])
    }

def classify_depth(L_value, ambient_brightness):

    adj = 0
    if ambient_brightness < 100:
        adj = 20 
        
    if L_value >= (LIGHT_THRESHOLD - adj):
        return "LIGHT"
    elif L_value >= (MEDIUM_THRESHOLD - adj):
        return "MEDIUM"
    else:
        return "DEEP"

def get_shade_depth(image, landmarks, draw=False):

    indices = [10, 123, 352] 

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ambient_brightness = np.mean(gray)

    skin_lab = extract_skin_lab(image, landmarks, indices)
    
    if skin_lab is None:
        return {"depth": "UNKNOWN", "confidence": 0, "L_value": 0}

    L = skin_lab["L"]
    depth = classify_depth(L, ambient_brightness)

    confidence = min(abs(L - MEDIUM_THRESHOLD) / 50.0, 1.0)

    if draw:
        h, w = image.shape[:2]
        for idx in indices:
            pt = landmarks[idx]
            cv2.circle(image, (int(pt.x * w), int(pt.y * h)), 5, (0, 255, 0), -1)
            cv2.putText(image, f"L:{int(L)}", (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    return {
        "depth": depth,
        "confidence": round(confidence, 2),
        "L_value": round(L, 1)
    }