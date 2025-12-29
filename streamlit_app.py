#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Celebrity Look-Alike Challenge - Simple Streamlit Web Application
Uses st.camera_input for less laggy webcam capture (on-demand processing).
"""

import streamlit as st
import cv2
import mediapipe as mp
import face_recognition
import numpy as np
import os
import time
import pickle
from PIL import Image

class CelebrityMatcher:
    """Encapsulated matching logic for celebrity recognition."""
    
    def __init__(self, celebs_dir="celebs", cache_file="encodings.pickle"):
        self.celebs_dir = celebs_dir
        self.cache_file = cache_file
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True, 
            refine_landmarks=True, 
            max_num_faces=1
        )
        self.connections = self.mp_face_mesh.FACEMESH_TESSELATION
        self.celeb_data = []
    
    def get_norm_lms(self, img_bgr):
        """Extract and normalize landmarks from image."""
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        if res.multi_face_landmarks:
            coords = np.array([(l.x, l.y) for l in res.multi_face_landmarks[0].landmark])
            return coords - coords.mean(axis=0)
        return None
    
    def load_database(self):
        """Load celebrity database from cache or build from images."""
        if not os.path.exists(self.celebs_dir):
            os.makedirs(self.celebs_dir, exist_ok=True)
        
        existing_cache = []
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "rb") as f:
                    existing_cache = pickle.load(f)
            except:
                existing_cache = []
        
        cached_filenames = {c.get('filename'): c for c in existing_cache if 'filename' in c}
        current_files = [f for f in os.listdir(self.celebs_dir) 
                        if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        
        updated_data = []
        new_encodes = 0
        
        for filename in current_files:
            # Skip yibo files
            if 'yibo' in filename.lower():
                continue
                
            if filename in cached_filenames:
                # Skip yibo entries from cache
                if 'yibo' not in cached_filenames[filename].get('name', '').lower():
                    updated_data.append(cached_filenames[filename])
            else:
                path = os.path.join(self.celebs_dir, filename)
                img = cv2.imread(path)
                if img is None:
                    continue
                
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encs = face_recognition.face_encodings(rgb)
                
                if encs:
                    lms = self.get_norm_lms(img)
                    if lms is not None:
                        name_part = filename.rsplit('_', 1)[0]
                        name = name_part.replace('_', ' ').title()
                        updated_data.append({
                            "name": name,
                            "enc": encs[0],
                            "lms": lms,
                            "img": cv2.resize(img, (200, 200)),
                            "filename": filename
                        })
                        new_encodes += 1
                else:
                    os.remove(path)
        
        self.celeb_data = updated_data
        
        if new_encodes > 0 or len(updated_data) != len(existing_cache):
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.celeb_data, f)
        
        return self.celeb_data
    
    def find_match(self, frame_rgb):
        """Find the best matching celebrity for a frame."""
        if not self.celeb_data:
            return None, None, None
        
        # Downsample for speed
        small_rgb = cv2.resize(frame_rgb, (0, 0), fx=0.25, fy=0.25)
        face_encs = face_recognition.face_encodings(small_rgb)
        
        if not face_encs:
            return None, None, None
        
        dists = face_recognition.face_distance(
            [c['enc'] for c in self.celeb_data], 
            face_encs[0]
        )
        idx = np.argmin(dists)
        distance = dists[idx]
        similarity = max(0, (1 - distance) * 100)
        
        return self.celeb_data[idx], distance, similarity


@st.cache_resource
def load_celebrity_data():
    """Load celebrity database with caching."""
    matcher = CelebrityMatcher()
    celeb_data = matcher.load_database()
    return matcher, celeb_data


def process_frame(camera_file, matcher):
    """Process a single frame and return annotated image with match info."""
    # Open the uploaded file as PIL Image
    img_pil = Image.open(camera_file)
    # Convert PIL Image to numpy array (RGB format)
    img_array = np.array(img_pil.convert('RGB'))
    # Convert RGB to BGR for OpenCV
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    
    # Convert to RGB for processing
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Initialize MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        refine_landmarks=True, 
        max_num_faces=1
    )
    
    # Process with MediaPipe
    results = face_mesh.process(rgb_img)
    
    match = None
    similarity = 0.0
    
    if results.multi_face_landmarks and matcher.celeb_data:
        # Find best match
        match, distance, similarity = matcher.find_match(rgb_img)
        
        if match:
            # Draw heatmap wireframe
            lm_list = results.multi_face_landmarks[0].landmark
            curr_lms = np.array([(l.x, l.y) for l in lm_list])
            curr_norm = curr_lms - curr_lms.mean(axis=0)
            errors = np.linalg.norm(curr_norm - match['lms'], axis=1)
            
            # Draw connections with color based on error
            connections = mp_face_mesh.FACEMESH_TESSELATION
            for conn in connections:
                avg_err = (errors[conn[0]] + errors[conn[1]]) / 2
                g = max(0, 255 - int(avg_err * 8500))
                r = min(255, int(avg_err * 8500))
                
                pt1 = (int(lm_list[conn[0]].x * w), int(lm_list[conn[0]].y * h))
                pt2 = (int(lm_list[conn[1]].x * w), int(lm_list[conn[1]].y * h))
                cv2.line(img, pt1, pt2, (0, g, r), 1)
    
    # Convert back to RGB for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb), match, similarity


def main():
    st.set_page_config(
        page_title="Celebrity Look-Alike Challenge",
        layout="wide"
    )
    
    st.title("How much do you look like a celebrity?")
    
    # Load celebrity data
    try:
        with st.spinner("Loading celebrity database..."):
            matcher, celeb_data = load_celebrity_data()
        if len(celeb_data) == 0:
            st.warning("No celebrities found in the database. Please add celebrity images to the 'celebs/' directory.")
    except Exception as e:
        st.error(f"Error loading celebrity database: {str(e)}")
        st.stop()
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Camera input (captures photo on-demand, less laggy)
        camera_image = st.camera_input("Take a photo", key="camera")
        
        match = None
        similarity = 0.0
        
        if camera_image:
            # Process the captured image
            with st.spinner("Processing..."):
                processed_img, match, similarity = process_frame(camera_image, matcher)
            
            st.image(processed_img, use_container_width=True, caption="Your photo with analysis")
    
    with col2:
        st.subheader("Match Information")
        
        if camera_image and match:
            # Display celebrity image
            st.image(match['img'], use_container_width=True, caption=match['name'])
            
            # Display name and similarity
            st.markdown(f"### {match['name']}")
            st.markdown(f"**Similarity: {similarity:.1f}%**")
            
            # Progress bar for similarity
            st.progress(similarity / 100)
        elif camera_image:
            st.info("No match found. Try adjusting your position or lighting.")
        else:
            st.info("Take a photo to see your celebrity match!")
        
        st.markdown("---")
        st.subheader("Register Your Face?")
        
        # Registration form
        with st.form("register_form"):
            name_input = st.text_input("", placeholder="your name here")
            submitted = st.form_submit_button("Register Face", use_container_width=True)
            
            if submitted and camera_image:
                try:
                    # Open the uploaded file as PIL Image
                    img_pil = Image.open(camera_image)
                    # Convert PIL Image to numpy array, then to OpenCV BGR format
                    img_array = np.array(img_pil.convert('RGB'))
                    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    else:
                        st.warning("Invalid image format. Please take a new photo.")
                        img_bgr = None
                    
                    # Check if face is detected
                    if img_bgr is not None:
                        rgb_array = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                        face_encs = face_recognition.face_encodings(rgb_array)
                    else:
                        face_encs = []
                    
                    if face_encs and img_bgr is not None:
                        # Save image
                        celebs_dir = "celebs"
                        if not os.path.exists(celebs_dir):
                            os.makedirs(celebs_dir, exist_ok=True)
                        
                        filename = f"{name_input.lower().replace(' ', '_')}_{int(time.time())}.jpg"
                        filepath = os.path.join(celebs_dir, filename)
                        cv2.imwrite(filepath, img_bgr)
                        
                        # Clear cache to reload
                        load_celebrity_data.clear()
                        
                        st.success(f"Successfully registered: {name_input}")
                        st.rerun()
                    else:
                        st.warning("No face detected. Please take a new photo with a clear face.")
                except Exception as e:
                    st.error(f"Error registering face: {str(e)}")
            elif submitted:
                st.warning("Please take a photo first.")


if __name__ == "__main__":
    main()
