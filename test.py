#!/Users/yliu3y/Passwords/venv/bin/python3
# -*- coding: utf-8 -*-
"""
Face Mesh Visualization using MediaPipe
Press 'q' to quit
"""

import cv2
import mediapipe as mp

# 1. Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# This object handles the detection logic
# refine_landmarks=True adds points for the eyes/iris specifically
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Define how the lines and dots look
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Flip the image horizontally for a selfie-view display
    image = cv2.flip(image, 1)
    
    # MediaPipe needs RGB, but OpenCV uses BGR
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    # 2. Draw the Points
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            
            # Draw the 'Tesselation' (the web of lines connecting the points)
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
            )
            
            # Draw the 'Contours' (the outlines of eyes, lips, and face shape)
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)
            )

    cv2.imshow('Face Landmark Points', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()