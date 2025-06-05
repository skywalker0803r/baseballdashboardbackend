from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import uuid
import cv2
import mediapipe as mp
import base64
import time
import numpy as np
from predict import predict_function

def get_landmark_vector(lm, idx):
    return np.array([lm[idx].x, lm[idx].y, lm[idx].z])  # ✅ 改為 NumPy 陣列


def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


# 1.跨步角度
def calc_stride_angle(lm):
    return calculate_angle(
        get_landmark_vector(lm, 24),
        get_landmark_vector(lm, 26),
        get_landmark_vector(lm, 23),
    )

# 2.投擲角度
def calc_throwing_angle(lm):
    return calculate_angle(
        get_landmark_vector(lm, 12),
        get_landmark_vector(lm, 14),
        get_landmark_vector(lm, 16),
    )

# 3.雙手對稱性
def calc_arm_symmetry(lm):
    return 1 - abs(lm[15].y - lm[16].y)

# 4.髖部旋轉角度
def calc_hip_rotation(lm):
    return abs(lm[23].z - lm[24].z)

# 5.右手手肘的高度
def calc_elbow_height(lm):
    return lm[14].y

# 6
def calc_ankle_height(lm):
    return lm[28].y

# 7
def calc_shoulder_rotation(lm):
    return abs(lm[11].z - lm[12].z)

# 8
def calc_torso_tilt_angle(lm):
    return calculate_angle(
        get_landmark_vector(lm, 11),
        get_landmark_vector(lm, 23),
        get_landmark_vector(lm, 24),
    )

# 9
def calc_release_distance(lm):
    return np.linalg.norm(
        get_landmark_vector(lm, 16) - get_landmark_vector(lm, 12)
    )  # ✅ 修正 list 相減

# 10
def calc_shoulder_to_hip(lm):
    return abs(lm[12].x - lm[24].x)


app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

sessions = {}

def process_and_stream(video_path, sid):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = 1.0 / fps if fps > 0 else 1.0 / 25
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_landmarks = []
        stride_angles = []
        throwing_angles = []
        arm_symmetrys = []
        hip_rotations = []
        elbow_heights = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            stride_angle = None
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                frame_landmarks.append(lm)
                
                # 計算各種特徵
                stride_angles.append(calc_stride_angle(lm))
                throwing_angles.append(calc_throwing_angle(lm))
                arm_symmetrys.append(calc_arm_symmetry(lm))
                hip_rotations.append(calc_hip_rotation(lm))
                elbow_heights.append(calc_elbow_height(lm))
                
                # 畫骨架圖
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            else:
                frame_landmarks.append(None)
                stride_angles.append(None)
                throwing_angles.append(None)
                arm_symmetrys.append(None)
                hip_rotations.append(None)
                elbow_heights.append(None)
                
            _, buffer = cv2.imencode('.jpg', frame)
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            socketio.emit('frame', {
                'image': frame_b64,
                'stride_angle': stride_angles[-1],
                'throwing_angle': throwing_angles[-1],
                'arm_symmetry': arm_symmetrys[-1],
                'hip_rotation': hip_rotations[-1],
                'elbow_height': elbow_heights[-1],
            }, room=sid)
        # 影片結束
        # 預測
        predict = predict_function(frame_landmarks,'model/model.pth')
        # 打印預測結果
        socketio.emit('frame', {
            'image': frame_b64,
            'stride_angle': stride_angles[-1],
            'throwing_angle': throwing_angles[-1],
            'arm_symmetry': arm_symmetrys[-1],
            'hip_rotation': hip_rotations[-1],
            'elbow_height': elbow_heights[-1],
            'predict':predict,
        }, room=sid)
    cap.release()
    socketio.emit('done', {}, room=sid)
    # 刪除影片檔案
    if os.path.exists(video_path):
        os.remove(video_path)

@app.route('/api/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400
    file = request.files['video']
    session_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_FOLDER, f"{session_id}.mp4")
    file.save(save_path)
    sessions[session_id] = save_path
    return jsonify({'session_id': session_id})

@socketio.on('start_stream')
def handle_start_stream(data):
    session_id = data.get('session_id')
    sid = request.sid
    video_path = sessions.get(session_id)
    if video_path:
        socketio.start_background_task(process_and_stream, video_path, sid)
    else:
        emit('error', {'message': 'Invalid session_id'})

if __name__ == '__main__':
    import eventlet
    import eventlet.wsgi
    socketio.run(app, host='0.0.0.0', port=10000)