import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# Load model and label encoder
model = load_model("emotion_model.h5")
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

img_size = 48
cap = cv2.VideoCapture(0)


#url = 'http://192.168.140.251:8080/shot.jpg'  # Your IP webcam URL

import urllib.request

while True:
    try:
        img_resp = urllib.request.urlopen(url)
        img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_np, -1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(gray, (img_size, img_size))
        face = face.reshape(1, img_size, img_size, 1) / 255.0

        prediction = model.predict(face)[0]
        emotion = le.inverse_transform([np.argmax(prediction)])[0]
        confidence = np.max(prediction)

        label = f"{emotion} ({confidence*100:.1f}%)"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Emotion Detection", frame)

    except Exception as e:
        print("Camera error:", e)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cv2.destroyAllWindows()
