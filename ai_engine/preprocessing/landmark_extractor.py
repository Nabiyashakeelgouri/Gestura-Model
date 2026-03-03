import mediapipe as mp
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def extract_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_results = hands.process(frame_rgb)
    face_results = face_mesh.process(frame_rgb)

    hands_landmarks = []
    face_landmarks = []

    if hand_results.multi_hand_landmarks:
        for hand in hand_results.multi_hand_landmarks:
            hand_points = []
            for lm in hand.landmark:
                hand_points.append([lm.x, lm.y, lm.z])
            hands_landmarks.append(hand_points)

    if face_results.multi_face_landmarks:
        for face in face_results.multi_face_landmarks:
            face_points = []
            for lm in face.landmark:
                face_points.append([lm.x, lm.y, lm.z])
            face_landmarks.append(face_points)

    return {
        "hands": hands_landmarks,
        "face": face_landmarks
    }