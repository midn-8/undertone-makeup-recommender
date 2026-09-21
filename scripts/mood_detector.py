import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp

class FERModel(nn.Module):
    def __init__(self, num_classes=4):
        super(FERModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
   
        self.fc1 = nn.Linear(4608, 256) 
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 4608)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FERModel(num_classes=4)
model.load_state_dict(torch.load("fer_model.pth", map_location=device))
model.eval()

EMOTIONS = ["angry", "happy", "sad", "neutral"]

def mood_makeup(emotion, features):
    bright_score = 0
    bold_score = 0

    if emotion == "happy":
        bright_score += 3
    elif emotion == "neutral":
        bright_score += 1 
    elif emotion in ["sad", "angry"]:
        bold_score += 3

    if features["big_smile"] and features["big_eyes"]:
        bright_score += 2
    if features["big_smile"] and not features["eyebrows_down"]:
        bright_score += 1
    if features["small_eyes"] and features["eyebrows_down"]:
        bold_score += 1
    if features["pout"] and features["eyebrows_down"]:
        bold_score += 2

    return "bright" if bright_score >= bold_score else "bold"


def get_mood(frame, landmarks=None, draw_landmarks=False):

    if landmarks is None:
        return "bright"
        
    try:
        h, w, _ = frame.shape

        mouth_open = abs(landmarks[13].y - landmarks[14].y) * h
        eye_open = abs(landmarks[159].y - landmarks[145].y) * h
        brow_dist = abs(landmarks[70].y - landmarks[159].y) * h
        
        features = {
            "big_smile": mouth_open > 7,
            "big_eyes": eye_open > 5,
            "eyebrows_down": brow_dist < 15,
            "small_eyes": eye_open < 3,
            "pout": 0.5 < mouth_open < 3
        }

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
        y1, y2 = int(landmarks[10].y * h), int(landmarks[152].y * h)
        x1, x2 = int(landmarks[234].x * w), int(landmarks[454].x * w)
        
        face_roi = gray[max(0,y1):y2, max(0,x1):x2]
        
        emotion = "neutral"
        if face_roi.size > 0:
            face_roi = cv2.resize(face_roi, (48, 48)) / 255.0
            img_tensor = torch.FloatTensor(face_roi).unsqueeze(0).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(img_tensor)
                emotion = EMOTIONS[torch.argmax(output).item()]

        if draw_landmarks:
            cv2.putText(frame, f"Emotion: {emotion}", (w-200, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
 
            for idx in [13, 14, 159, 145, 70]:
                pt = landmarks[idx]
                cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 3, (0, 255, 0), -1)

        return mood_makeup(emotion, features)

    except Exception as e:
        print(f"Mood Detector Error: {e}")
        return "bright"