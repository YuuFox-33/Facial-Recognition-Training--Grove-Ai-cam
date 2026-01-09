import face_recognition
import cv2
import numpy as np
import os
import time

# Load known data
known_encodings = list(np.load("encodings.npy", allow_pickle=True))
known_names = list(np.load("names.npy", allow_pickle=True))

TARGET = "yuu"
SAVE_DIR = "faces/yuu"
os.makedirs(SAVE_DIR, exist_ok=True)

# Thresholds
RECOG_THRESHOLD = 0.40
LEARN_THRESHOLD = 0.36
TEMPORAL_TIME = 0.5
CONSENSUS_RATIO = 0.70
SAVE_INTERVAL = 2.0

cap = cv2.VideoCapture(0)

stable_start = None
last_saved = 0
session_learned = 0   # NEW
flash_time = 0        # NEW

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, locations)

    for (top, right, bottom, left), face_encoding in zip(locations, encodings):

        distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_idx = np.argmin(distances)
        best_dist = distances[best_idx]

        name = "Unknown"
        color = (0, 0, 255)

        if best_dist < RECOG_THRESHOLD:
            name = known_names[best_idx]
            color = (0, 255, 0)

            if name == TARGET:
                if stable_start is None:
                    stable_start = time.time()

                stable_time = time.time() - stable_start

                if stable_time >= TEMPORAL_TIME and best_dist < LEARN_THRESHOLD:
                    votes = np.sum(distances < LEARN_THRESHOLD)
                    ratio = votes / len(distances)

                    if ratio >= CONSENSUS_RATIO:
                        if time.time() - last_saved > SAVE_INTERVAL:
                            path = f"{SAVE_DIR}/auto_{int(time.time())}.jpg"
                            cv2.imwrite(path, frame)
                            known_encodings.append(face_encoding)
                            known_names.append(TARGET)
                            last_saved = time.time()

                            session_learned += 1      # NEW
                            flash_time = time.time()  # NEW
            else:
                stable_start = None
        else:
            stable_start = None

        label = f"{name} {best_dist:.2f}"
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, label, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    # ===== UI OVERLAY =====
    total_enc = len(known_encodings)
    cv2.putText(frame, f"Session learned: {session_learned}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.putText(frame, f"Total encodings: {total_enc}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)

    # Flash feedback when a frame is learned
    if time.time() - flash_time < 0.6:
        cv2.putText(frame, "+1 GOOD FRAME",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0,255,255), 2)

    cv2.imshow("Temporal Controlled Learning", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

np.save("encodings.npy", known_encodings)
np.save("names.npy", known_names)

print(f"Session learned frames: {session_learned}")
print(f"Total encodings now: {len(known_encodings)}")
