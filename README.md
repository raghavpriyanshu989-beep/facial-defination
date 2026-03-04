AI Emotion Detection and Glasses Recommendation System
Project Overview

This project is a real-time AI-based web application that detects a user’s facial emotion through a webcam feed and recommends suitable glasses based on face shape analysis.

The goal of this project was to combine computer vision, emotion recognition, and a rule-based recommendation system into an interactive web application. The system captures live video, analyzes facial expressions using artificial intelligence, estimates face proportions, and provides practical style suggestions — all in real time within a browser.

How the System Works

When the application runs:

The webcam captures live video frames.

Selected frames are processed using an AI model.

The system detects:

The dominant facial emotion

The confidence score of that emotion

The dimensions of the detected face region

The face shape is estimated using a width-to-height ratio.

Based on the calculated face shape, the system recommends appropriate glasses styles.

The results are displayed directly on the live video stream.

All processing happens continuously, creating a smooth and interactive experience.

Technologies Used

Python

Flask (Web framework)

OpenCV (Camera and image processing)

DeepFace (Emotion detection model)

HTML templates

The emotion recognition functionality is powered by DeepFace, while real-time image capture and processing are handled using OpenCV. The web application backend is built using Flask.

Core Features

Real-time facial emotion detection

Confidence score display

Face shape classification

Rule-based glasses recommendation

Emotion smoothing to prevent unstable predictions

Live browser-based video streaming

Performance optimization through frame skipping

Detailed System Explanation
Emotion Detection

The system uses DeepFace to analyze selected video frames and determine the dominant facial emotion. To prevent rapid changes in emotion output caused by minor variations between frames, a smoothing mechanism is implemented.

A fixed-size sliding window stores recent emotion results. The most frequently occurring emotion in that window is displayed. This improves stability and creates a more natural output.

Face Shape Classification

Face shape is estimated using a simple geometric approach:

Face Ratio = Face Width / Face Height

Based on the calculated ratio, the face is classified into one of the following categories:

Round

Square

Long

Oval

This method uses rule-based logic rather than complex landmark detection, making it computationally efficient while still effective for demonstration purposes.

Glasses Recommendation Logic

Once the face shape is determined, the system suggests glasses styles that typically complement that shape:

Round face → Rectangular or geometric frames

Square face → Round or oval frames

Long face → Oversized or tall frames

Oval face → Most frame styles

This recommendation system is intentionally simple and demonstrates how AI output can be connected to practical decision-making logic.

Performance Optimization

To ensure smooth real-time processing:

The system analyzes every 20th frame instead of every frame.

Frames are resized before being passed to the AI model.

Face detection errors are safely handled to prevent application crashes.

Debug mode is disabled for safer execution.

These optimizations help maintain responsiveness and reduce computational load.

Application Flow

The user opens the web application.

Flask loads the main HTML template.

The browser requests the video stream endpoint.

The server continuously captures and processes frames.

Emotion, confidence level, face shape, and glasses recommendation are overlaid on the video.

The processed frames are streamed back to the browser in real time.

Learning Outcomes

Through this project, I gained practical experience in:

Integrating AI models into web applications

Real-time video processing using OpenCV

Building streaming responses in Flask

Implementing performance optimization techniques

Combining machine learning outputs with rule-based systems

This project demonstrates how artificial intelligence can be integrated into interactive applications that provide meaningful and personalized outputs.

Future Improvements

Improve face shape detection using facial landmarks

Add age and gender prediction

Enhance the frontend interface and design

Deploy the application to a cloud platform

Store analysis results in a database

If you would like, I can now:

Add a professional “Installation & Setup” section

Create a “How to Run Locally” guide

Help you prepare a resume-ready project description

Help you deploy this project online
