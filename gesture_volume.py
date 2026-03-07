import cv2
import math
import time
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pycaw.pycaw import AudioUtilities


device = AudioUtilities.GetSpeakers()
volume = device.EndpointVolume

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
)

landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam.")
    raise SystemExit

prev_distance = None

min_distance = 20
max_distance = 200

while True:
    success, frame = cap.read()
    if not success:
        print("Could not read frame from webcam.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    timestamp_ms = int(time.time() * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if result.hand_landmarks:
        hand_landmarks = result.hand_landmarks[0]

        thumb = hand_landmarks[4]
        index = hand_landmarks[8]

        x1, y1 = int(thumb.x * w), int(thumb.y * h)
        x2, y2 = int(index.x * w), int(index.y * h)

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(frame, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(frame, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(frame, (cx, cy), 8, (0, 255, 0), cv2.FILLED)

        distance = math.hypot(x2 - x1, y2 - y1)

        if prev_distance is None:
            smooth_distance = distance
        else:
            smooth_distance = 0.8 * prev_distance + 0.2 * distance
        prev_distance = smooth_distance

        vol_scalar = np.interp(smooth_distance, [min_distance, max_distance], [0.0, 1.0])
        vol_scalar = np.clip(vol_scalar, 0.0, 1.0)

        volume.SetMasterVolumeLevelScalar(float(vol_scalar), None)

        actual_scalar = volume.GetMasterVolumeLevelScalar()
        vol_percent = int(actual_scalar * 100)
        vol_bar = np.interp(actual_scalar, [0.0, 1.0], [400, 150])

        cv2.rectangle(frame, (50, 150), (85, 400), (0, 255, 0), 3)
        cv2.rectangle(frame, (50, int(vol_bar)), (85, 400), (0, 255, 0), cv2.FILLED)
        cv2.putText(
            frame,
            f"{vol_percent}%",
            (25, 440),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3
        )

    cv2.imshow("Gesture Volume Control", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()