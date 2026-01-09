import face_recognition
import cv2
import numpy as np
import os
import time

# ===================== LOAD EXISTING DATA =====================
known_encodings = list(np.load("encodings.npy", allow_pickle=True))
known_names = list(np.load("names.npy", allow_pickle=True))

TARGET = "yuu"
SAVE_DIR = "faces/yuu"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===================== QUICK TEST PARAMETERS =====================
RECOG_THRESHOLD = 0.45      # display only
LEARN_THRESHOLD = 0.42      # relaxed learning gate
TEMPORAL_TIME = 0.3         # seconds face must stay stable
CONSENSUS_RATIO = 0.40      # used AFTER bootstrap
BOOTSTRAP_MIN = 10          # until this many encodings, skip consensus
SAVE_INTERVAL = 0.8         # seconds between saves
MAX_SESSION_LEARN = 20      # auto stop

# ===================== STATE =====================
cap = cv2.VideoCapture(0)

stable_start = None
last_saved = 0
session_learned = 0
flash_time = 0

# ===================== MAIN LOOP =====================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, locations)

    for (top, right, bottom, left), face_encoding in zip(locations, encodings):

        distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_dist = np.min(distances)

        votes = np.sum(distances < LEARN_THRESHOLD)
        ratio = votes / len(distances)

        # ===== Bootstrap-aware consensus =====
        if len(known_encodings) < BOOTSTRAP_MIN:
            ratio_ok = True
            consensus_state = "BOOTSTRAP"
        else:
            ratio_ok = ratio >= CONSENSUS_RATIO
            consensus_state = "CONSENSUS"

        # Display label only (not used for learning)
        name = "Yuu?" if best_dist < RECOG_THRESHOLD else "Unknown"
        color = (0, 255, 0) if best_dist < RECOG_THRESHOLD else (0, 0, 255)

        # ===== LEARNING LOGIC =====
        if session_learned < MAX_SESSION_LEARN and best_dist < LEARN_THRESHOLD:
            if stable_start is None:
                stable_start = time.time()

            stable_time = time.time() - stable_start

            if stable_time >= TEMPORAL_TIME and ratio_ok:
                if time.time() - last_saved > SAVE_INTERVAL:
                    path = f"{SAVE_DIR}/auto_{int(time.time())}.jpg"
                    cv2.imwrite(path, frame)

                    known_encodings.append(face_encoding)
                    known_names.append(TARGET)

                    session_learned += 1
                    last_saved = time.time()
                    flash_time = time.time()
        else:
            stable_start = None

        # ===== DRAW FACE BOX =====
        label = f"{name}  dist={best_dist:.2f}"
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, label, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    # ===================== UI OVERLAY =====================
    cv2.putText(frame, f"Session learned: {session_learned}/{MAX_SESSION_LEARN}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.putText(frame, f"Total encodings: {len(known_encodings)}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)

    cv2.putText(frame, f"Mode: {consensus_state}",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180,180,180), 2)

    if time.time() - flash_time < 0.6:
        cv2.putText(frame, "+1 GOOD FRAME",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0,255,255), 2)

    if session_learned >= MAX_SESSION_LEARN:
        cv2.putText(frame, "LEARNING STOPPED",
                    (10, 160), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0,0,255), 2)

    cv2.imshow("Quick Controlled Learning (Bootstrap-Aware)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===================== CLEANUP =====================
cap.release()
cv2.destroyAllWindows()

np.save("encodings.npy", known_encodings)
np.save("names.npy", known_names)

print(f"Session learned frames: {session_learned}")
print(f"Total encodings now: {len(known_encodings)}")
