import face_recognition
import cv2
import numpy as np

known_encodings = list(np.load("encodings.npy", allow_pickle=True))
known_names = list(np.load("names.npy", allow_pickle=True))

RECOG_THRESHOLD = 0.45   # recognition only

cap = cv2.VideoCapture(0)

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

        if best_dist < RECOG_THRESHOLD:
            name = known_names[best_idx]
            color = (0, 255, 0)
        else:
            name = "Unknown"
            color = (0, 0, 255)

        label = f"{name}  dist={best_dist:.2f}"
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, label, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    cv2.imshow("Face Recognition – Test Only", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
