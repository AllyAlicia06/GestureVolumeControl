import cv2
import time
import math
import ctypes
import numpy as np
import mediapipe as mp

from mediapipe.tasks.python import vision
from pycaw.pycaw import AudioUtilities


device = AudioUtilities.GetSpeakers()
volume = device.EndpointVolume

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
VK_MEDIA_PLAY_PAUSE = 0xB3
VK_MEDIA_NEXT_TRACK = 0xB0
VK_MEDIA_PREV_TRACK = 0xB1

def send_media_key(vk_code):
    ctypes.windll.user32.keybd_event(vk_code, 0, KEYEVENTF_EXTENDEDKEY, 0)
    ctypes.windll.user32.keybd_event(vk_code, 0, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0)

def finger_is_up(landmarks, tip_idx, pip_idx, margin=0.02):
    return landmarks[tip_idx].y < (landmarks[pip_idx].y - margin)

def distance_norm(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def palm_size_norm(landmarks):
    return max(distance_norm(landmarks[0], landmarks[9]), 1e-6)

def get_finger_states(landmarks):
    index_up = finger_is_up(landmarks, 8, 6)
    middle_up = finger_is_up(landmarks, 12, 10)
    ring_up = finger_is_up(landmarks, 16, 14)
    pinky_up = finger_is_up(landmarks, 20, 18)
    return index_up, middle_up, ring_up, pinky_up

def classify_pose(landmarks):
    index_up, middle_up, ring_up, pinky_up = get_finger_states(landmarks)
    other_three_down = not middle_up and not ring_up and not pinky_up

    thumb_index_ratio = distance_norm(landmarks[4], landmarks[8]) / palm_size_norm(landmarks)
    thumb_index_close = thumb_index_ratio < 0.45

    if other_three_down and (index_up or thumb_index_close):
        return "volume"

    if index_up and middle_up and not ring_up and not pinky_up:
        return "swipe"

    if index_up and middle_up and ring_up and pinky_up:
        return "open_palm"

    return "idle"

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

filtered_target_scalar = volume.GetMasterVolumeLevelScalar()
last_set_scalar = filtered_target_scalar

current_pose = "idle"
pose_since = 0.0

PLAY_PAUSE_HOLD = 0.35
PLAY_PAUSE_COOLDOWN = 0.9
last_play_pause_time = 0.0
open_palm_triggered_this_hold = False

SWIPE_ARM_DELAY = 0.20
SWIPE_COOLDOWN = 0.90
last_swipe_time = 0.0
swipe_origin_x = None
smoothed_wrist_x = None
swipe_recenter_required = False

while True:
    success, frame = cap.read()
    if not success:
        print("Could not read frame from webcam.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms = int(time.perf_counter() * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    now = time.time()
    gesture_status = "No hand"

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]
        pose = classify_pose(landmarks)

        if pose != current_pose:
            current_pose = pose
            pose_since = now
            open_palm_triggered_this_hold = False
            swipe_origin_x = None
            smoothed_wrist_x = None
            swipe_recenter_required = False

        wrist_x = landmarks[0].x * w
        wrist_y = landmarks[0].y * h
        thumb_x = int(landmarks[4].x * w)
        thumb_y = int(landmarks[4].y * h)
        index_x = int(landmarks[8].x * w)
        index_y = int(landmarks[8].y * h)

        cv2.circle(frame, (int(wrist_x), int(wrist_y)), 7, (0, 255, 255), cv2.FILLED)
        cv2.circle(frame, (thumb_x, thumb_y), 8, (255, 0, 255), cv2.FILLED)
        cv2.circle(frame, (index_x, index_y), 8, (255, 0, 255), cv2.FILLED)
        cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), (255, 0, 255), 2)

        if current_pose == "volume":
            ratio = distance_norm(landmarks[4], landmarks[8]) / palm_size_norm(landmarks)
            target_scalar = np.interp(ratio, [0.25, 1.35], [0.0, 1.0])
            target_scalar = float(np.clip(target_scalar, 0.0, 1.0))

            filtered_target_scalar = 0.82 * filtered_target_scalar + 0.18 * target_scalar

            if volume.GetMute():
                volume.SetMute(0, None)

            if abs(filtered_target_scalar - last_set_scalar) >= 0.012:
                volume.SetMasterVolumeLevelScalar(filtered_target_scalar, None)
                last_set_scalar = filtered_target_scalar

            gesture_status = f"Volume control ({int(filtered_target_scalar * 100)}%)"

        elif current_pose == "swipe":
            if smoothed_wrist_x is None:
                smoothed_wrist_x = wrist_x
            else:
                smoothed_wrist_x = 0.80 * smoothed_wrist_x + 0.20 * wrist_x

            if now - pose_since < SWIPE_ARM_DELAY:
                gesture_status = "Swipe mode..."
            else:
                if swipe_origin_x is None:
                    swipe_origin_x = smoothed_wrist_x
                    gesture_status = "Swipe armed"
                else:
                    dx = smoothed_wrist_x - swipe_origin_x
                    swipe_threshold = w * 0.12
                    recenter_window = w * 0.05
                    cooldown_done = (now - last_swipe_time) >= SWIPE_COOLDOWN

                    if swipe_recenter_required:
                        if abs(smoothed_wrist_x - swipe_origin_x) <= recenter_window:
                            swipe_recenter_required = False
                            gesture_status = "Swipe re-armed"
                        else:
                            gesture_status = "Return to center"
                    else:
                        if cooldown_done and dx >= swipe_threshold:
                            send_media_key(VK_MEDIA_NEXT_TRACK)
                            last_swipe_time = now
                            swipe_recenter_required = True
                            gesture_status = "Next track"
                        elif cooldown_done and dx <= -swipe_threshold:
                            send_media_key(VK_MEDIA_PREV_TRACK)
                            last_swipe_time = now
                            swipe_recenter_required = True
                            gesture_status = "Previous track"
                        else:
                            gesture_status = "Swipe armed"

                    center_x = int(swipe_origin_x)
                    left_x = int(swipe_origin_x - swipe_threshold)
                    right_x = int(swipe_origin_x + swipe_threshold)

                    cv2.line(frame, (center_x, 100), (center_x, h - 40), (120, 120, 120), 1)
                    cv2.line(frame, (left_x, 140), (left_x, h - 80), (80, 80, 255), 1)
                    cv2.line(frame, (right_x, 140), (right_x, h - 80), (80, 255, 80), 1)

        elif current_pose == "open_palm":
            held_long_enough = (now - pose_since) >= PLAY_PAUSE_HOLD
            cooldown_done = (now - last_play_pause_time) >= PLAY_PAUSE_COOLDOWN

            if held_long_enough and cooldown_done and not open_palm_triggered_this_hold:
                send_media_key(VK_MEDIA_PLAY_PAUSE)
                last_play_pause_time = now
                open_palm_triggered_this_hold = True
                gesture_status = "Play / Pause"
            else:
                gesture_status = "Open palm"

        else:
            gesture_status = "Volume frozen"

    else:
        current_pose = "idle"
        swipe_origin_x = None
        smoothed_wrist_x = None
        swipe_recenter_required = False
        open_palm_triggered_this_hold = False

    actual_scalar = volume.GetMasterVolumeLevelScalar()
    muted = bool(volume.GetMute())
    shown_percent = 0 if muted else int(actual_scalar * 100)
    vol_bar = np.interp(actual_scalar, [0.0, 1.0], [400, 150])

    cv2.rectangle(frame, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(frame, (50, int(vol_bar)), (85, 400), (0, 255, 0), cv2.FILLED)

    cv2.putText(frame, f"{shown_percent}%", (25, 440),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    if muted:
        cv2.putText(frame, "MUTED", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

    cv2.putText(frame, "Index only = volume (thumb-index distance)", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
    cv2.putText(frame, "Index+Middle = swipe | Return to center after each swipe", (10, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
    cv2.putText(frame, "Open palm = play/pause", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
    cv2.putText(frame, gesture_status, (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

    cv2.imshow("Gesture Volume Control", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()