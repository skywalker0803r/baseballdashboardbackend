import numpy as np
import torch
import torch.nn as nn
import copy

# 假設你的模型
class SimpleTCN(nn.Module):
    def __init__(self, input_size=4000 ,num_classes=2):
        super(SimpleTCN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.fc(x)

# 工具函式
def get_landmark_vector(lm, idx):
    try:
        return np.array([lm[idx].x, lm[idx].y, lm[idx].z])
    except Exception as e:
        print(f"[ERROR] get_landmark_vector failed at idx {idx}: {e}")
        return np.zeros(3)

def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def calc_stride_angle(lm):
    return calculate_angle(get_landmark_vector(lm, 24), get_landmark_vector(lm, 26), get_landmark_vector(lm, 23))

def calc_throwing_angle(lm):
    return calculate_angle(get_landmark_vector(lm, 12), get_landmark_vector(lm, 14), get_landmark_vector(lm, 16))

def calc_arm_symmetry(lm):
    return 1 - abs(lm[15].y - lm[16].y)

def calc_hip_rotation(lm):
    return abs(lm[23].z - lm[24].z)

def calc_elbow_height(lm):
    return lm[14].y

def calc_ankle_height(lm):
    return lm[28].y

def calc_shoulder_rotation(lm):
    return abs(lm[11].z - lm[12].z)

def calc_torso_tilt_angle(lm):
    return calculate_angle(get_landmark_vector(lm, 11), get_landmark_vector(lm, 23), get_landmark_vector(lm, 24))

def calc_release_distance(lm):
    return np.linalg.norm(get_landmark_vector(lm, 16) - get_landmark_vector(lm, 12))

def calc_shoulder_to_hip(lm):
    return abs(lm[12].x - lm[24].x)

# 主預測函數
def predict_function(landmarks_path, model_path='model/model.pth'):
    frame_landmarks = landmarks_path
    total_frames = len(frame_landmarks)
    margin = min(50, total_frames // 3)

    velocities = [
        np.linalg.norm(get_landmark_vector(frame_landmarks[i], 16) - get_landmark_vector(frame_landmarks[i - 1], 16))
        for i in range(1, total_frames)
    ]
    release_idx = int(np.argmax(velocities)) + 1
    indices = [release_idx - 20, release_idx - 5, release_idx, release_idx + 10]
    names = ["foot", "arm", "release", "hip"]

    features = []
    for name, idx in zip(names, indices):
        start = max(idx - margin, 0)
        end = min(idx + margin, total_frames)
        seq = list(frame_landmarks[start:end])

        pad_len = 100 - len(seq)
        if pad_len > 0:
            seq += [copy.deepcopy(seq[-1]) for _ in range(pad_len)]
        seq = seq[:100]

        f1 = [calc_stride_angle(lm) for lm in seq]
        f2 = [calc_throwing_angle(lm) for lm in seq]
        f3 = [calc_arm_symmetry(lm) for lm in seq]
        f4 = [calc_hip_rotation(lm) for lm in seq]
        f5 = [calc_elbow_height(lm) for lm in seq]
        f6 = [calc_ankle_height(lm) for lm in seq]
        f7 = [calc_shoulder_rotation(lm) for lm in seq]
        f8 = [calc_torso_tilt_angle(lm) for lm in seq]
        f9 = [calc_release_distance(lm) for lm in seq]
        f10 = [calc_shoulder_to_hip(lm) for lm in seq]

        features.append([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10])

    features = np.array(features).reshape(1, 4000)
    features = torch.tensor(features, dtype=torch.float32)

    model = SimpleTCN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        output = model(features)
        pred = output.argmax(dim=1).item()
        return "好球" if pred == 1 else "壞球"
