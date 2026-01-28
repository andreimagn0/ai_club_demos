import cv2
from deepface import DeepFace
import numpy as np

screen_width = 1920
screen_height = 1080

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Start webcam
cap = cv2.VideoCapture(0)

# Optional: improve performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Press 'q' to quit")

window_name = "AI Club Emotion Detection Demo"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5
    )

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]

        try:
            result = DeepFace.analyze(
                face_img,
                actions=['emotion'],
                enforce_detection=False
            )
            emotion = result[0]['dominant_emotion']
        except:
            emotion = "unknown"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            emotion,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    h, w = frame.shape[:2]
    scale = max(screen_width / w, screen_height / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    frame = cv2.resize(frame, (new_w, new_h))

    x_start = (new_w - screen_width) // 2
    y_start = (new_h - screen_height) // 2

    frame = frame[
        y_start:y_start + screen_height,
        x_start:x_start + screen_width
    ]


    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()