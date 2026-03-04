from flask import Flask, render_template, Response
import cv2
from deepface import DeepFace
import time
from collections import deque, Counter

app = Flask(__name__)

camera = cv2.VideoCapture(0)

FRAME_SKIP = 20
ANALYZE_SIZE = (320, 240)
SMOOTHING_WINDOW = 5

emotion_history = deque(maxlen=SMOOTHING_WINDOW)

current_emotion = "Detecting..."
current_confidence = 0
face_width = 0
face_height = 0
face_shape = ""
glasses_recommendation = ""

frame_count = 0


def smooth_emotion(new_emotion):
    emotion_history.append(new_emotion)
    return Counter(emotion_history).most_common(1)[0][0]


def classify_face_shape(width, height):
    ratio = width / height if height != 0 else 0
    if 0.95 <= ratio <= 1.05:
        return "Round"
    elif ratio > 1.1:
        return "Square"
    elif ratio < 0.9:
        return "Long"
    else:
        return "Oval"


def recommend_glasses(shape):
    if shape == "Round":
        return "Rectangular or geometric frames"
    elif shape == "Square":
        return "Round or oval frames"
    elif shape == "Long":
        return "Oversized or tall frames"
    elif shape == "Oval":
        return "Most frame styles suit you"
    else:
        return "Standard frames"


def generate_frames():
    global frame_count
    global current_emotion, current_confidence
    global face_width, face_height, face_shape, glasses_recommendation

    while True:
        success, frame = camera.read()
        if not success:
            break

        frame_count += 1

        if frame_count % FRAME_SKIP == 0:
            small_frame = cv2.resize(frame, ANALYZE_SIZE)
            try:
                result = DeepFace.analyze(
                    small_frame,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='opencv'
                )

                emotions = result[0]['emotion']
                dominant = result[0]['dominant_emotion']
                confidence = round(emotions[dominant], 2)

                region = result[0]['region']
                face_width = region['w']
                face_height = region['h']

                face_shape = classify_face_shape(face_width, face_height)
                glasses_recommendation = recommend_glasses(face_shape)

                current_emotion = smooth_emotion(dominant)
                current_confidence = confidence

            except:
                current_emotion = "No face detected"

        cv2.putText(frame, f"Emotion: {current_emotion}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(frame, f"Confidence: {current_confidence}%",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.putText(frame, f"Face Shape: {face_shape}",
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(frame, f"Glasses: {glasses_recommendation}",
                    (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    # Safe run with debug=False, no debugger PIN
    app.run(host="127.0.0.1", port=5000, debug=False)