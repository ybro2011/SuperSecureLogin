#!/Users/yliu3y/Passwords/venv/bin/python3
# -*- coding: utf-8 -*-
"""
Face Recognition with MediaPipe Mesh Overlay
Press 's' to train, 'q' to quit
"""

import cv2
import face_recognition
import mediapipe as mp
import numpy as np

# --- INITIALIZE MEDIAPIPE (Pipeframe) ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

# --- DATABASE ---
known_face_encodings = []
known_face_names = []

def train_face(frame, name="User"):
    """Encodes a face from a live frame and adds it to memory."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_frame)
    if encodings:
        known_face_encodings.append(encodings[0])
        known_face_names.append(name)
        return True
    return False

# --- MAIN LOOP ---
video_capture = cv2.VideoCapture(0)

print("--- COMMANDS ---")
print("Press 's' to capture your face and train the model.")
print("Press 'q' to quit.")

while True:
    ret, frame = video_capture.read()
    if not ret: break

    # 1. Create the Pipeframe (MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    # Draw the mesh/pipeframe
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_spec
            )

    # 2. Recognition Logic
    # We only process recognition every few frames or at a lower res to keep it fast
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"
        color = (0, 0, 255) # Red for unknown

        if known_face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                color = (0, 255, 0) # Green for recognized

        # Draw Label
        top *= 4; right *= 4; bottom *= 4; left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, f"ID: {name}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 3. Handle Key Presses
    cv2.imshow('Face ID + Mesh', frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('s'):
        print("Capturing face...")
        if train_face(frame, name="Admin"):
            print("Successfully Trained!")
        else:
            print("Error: No face detected. Try again.")
            
    elif key == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()