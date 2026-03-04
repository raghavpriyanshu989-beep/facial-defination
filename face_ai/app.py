from flask import Flask, render_template, Response
import cv2
from deepface import DeepFace
from collections import deque, Counter
import threading
import speech_recognition as sr
import pyttsx3
import webbrowser
import time
import atexit

app = Flask(__name__)

# ---------------- CAMERA ----------------
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

FRAME_SKIP = 20
ANALYZE_SIZE = (320, 240)
SMOOTHING_WINDOW = 5

emotion_history = deque(maxlen=SMOOTHING_WINDOW)

current_emotion = "Detecting..."
current_confidence = 0
face_shape = ""
glasses_recommendation = ""

frame_count = 0

# ---------------- VOICE ENGINE ----------------
engine = pyttsx3.init()
engine.setProperty('rate', 170)

def speak(text):
    try:
        engine.say(text)
        engine.runAndWait()
    except:
        pass

# ---------------- MOOD PLAYLIST ----------------
def open_playlist_by_mood(mood):
    playlists = {
        "happy": "https://www.youtube.com/results?search_query=happy+bollywood+songs",
        "sad": "https://www.youtube.com/results?search_query=sad+lofi+songs",
        "angry": "https://www.youtube.com/results?search_query=motivational+gym+songs",
        "neutral": "https://www.youtube.com/results?search_query=chill+vibes+playlist",
        "surprise": "https://www.youtube.com/results?search_query=party+songs",
        "fear": "https://www.youtube.com/results?search_query=calm+relaxing+music",
        "disgust": "https://www.youtube.com/results?search_query=focus+music"
    }

    url = playlists.get(mood, playlists["neutral"])
    webbrowser.open(url)
    speak("Opening a playlist that matches your mood")

# ---------------- FACE SHAPE ----------------
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
    recommendations = {
        "Round": "Rectangular frames",
        "Square": "Round frames",
        "Long": "Oversized frames",
        "Oval": "Most styles suit you"
    }
    return recommendations.get(shape, "Standard frames")

# ---------------- EMOTION SMOOTHING ----------------
def smooth_emotion(new_emotion):
    emotion_history.append(new_emotion)
    return Counter(emotion_history).most_common(1)[0][0]

# ---------------- VOICE LISTENER ----------------
def listen_command():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=5)
        command = recognizer.recognize_google(audio).lower()
        print("You said:", command)
        return command
    except:
        return ""

# ---------------- ASSISTANT LOOP ----------------
def assistant_loop():
    global current_emotion
    speak("AI Assistant Activated")

    while True:
        command = listen_command()

        if "open youtube" in command:
            speak("Opening YouTube")
            webbrowser.open("https://www.youtube.com")

        elif "how is my mood" in command or "how's my mood" in command:
            speak(f"Your current mood is {current_emotion}")
            open_playlist_by_mood(current_emotion)

        elif "exit assistant" in command:
            speak("Goodbye")
            break

# ---------------- VIDEO GENERATOR ----------------
def generate_frames():
    global frame_count, current_emotion, current_confidence
    global face_shape, glasses_recommendation

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
                    enforce_detection=False
                )

                if isinstance(result, list):
                    result = result[0]

                emotions = result.get('emotion', {})
                dominant = result.get('dominant_emotion', "Unknown")
                confidence = round(emotions.get(dominant, 0), 2)

                if 'region' in result:
                    region = result['region']
                    x, y, w, h = region['x'], region['y'], region['w'], region['h']

                    face_shape = classify_face_shape(w, h)
                    glasses_recommendation = recommend_glasses(face_shape)

                    current_emotion = smooth_emotion(dominant)
                    current_confidence = confidence

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

            except Exception as e:
                print("DeepFace error:", e)
                current_emotion = "No face detected"

        cv2.putText(frame, f"Emotion: {current_emotion}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.putText(frame, f"Confidence: {current_confidence}%",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        cv2.putText(frame, f"Face Shape: {face_shape}",
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.putText(frame, f"Glasses: {glasses_recommendation}",
                    (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,200,0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ---------------- ROUTES ----------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ---------------- CLEAN EXIT ----------------
@atexit.register
def release_camera():
    camera.release()

# ---------------- START EVERYTHING ----------------
if __name__ == "__main__":
    assistant_thread = threading.Thread(target=assistant_loop)
    assistant_thread.daemon = True
    assistant_thread.start()

    app.run(host="127.0.0.1", port=5000, debug=False)
    