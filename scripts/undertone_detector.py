import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
import mediapipe as mp
import numpy as np
from PIL import Image

IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["COOL", "NEUTRAL", "WARM"]

from torchvision.models import resnet18

model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(
    torch.load("undertone_model.pth", map_location=DEVICE)
)
model.to(DEVICE)
model.eval()

transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


CHEEK_POINTS = [234, 454]

def extract_skin_patch(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        
        return cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

    landmarks = results.multi_face_landmarks[0].landmark
    h, w = frame.shape[:2]

    patches = []
    for idx in CHEEK_POINTS:
        lm = landmarks[idx]
        x, y = int(lm.x * w), int(lm.y * h)

        size = 40
        patch = frame[
            max(0, y-size):min(h, y+size),
            max(0, x-size):min(w, x+size)
        ]
        if patch.size > 0:
            patches.append(patch)

    if not patches:
        return None

    combined = np.concatenate(patches, axis=1)
    return cv2.resize(combined, (IMG_SIZE, IMG_SIZE))
    
def get_undertone(frame, draw_points=False):
    patch = extract_skin_patch(frame)
    if patch is None:
        return None

    pil_img = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
    input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        pred = torch.argmax(outputs, dim=1).item()

    return CLASS_NAMES[pred]

