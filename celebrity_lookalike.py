import cv2
import mediapipe as mp
import face_recognition
import numpy as np
import os
import time
import pickle

class CelebrityChallengeUltimate:
    def __init__(self):
        self.celebs_dir = "celebs"
        self.cache_file = "encodings.pickle"
        if not os.path.exists(self.celebs_dir): os.makedirs(self.celebs_dir)
        
        # Initialize AI Models
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True, 
            refine_landmarks=True, 
            max_num_faces=1
        )
        self.connections = self.mp_face_mesh.FACEMESH_TESSELATION
        
        self.celeb_data = []
        self.current_match = None
        
        # Load the database
        self.load_database()

    def get_norm_lms(self, img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        if res.multi_face_landmarks:
            coords = np.array([(l.x, l.y) for l in res.multi_face_landmarks[0].landmark])
            return coords - coords.mean(axis=0)
        return None

    def load_database(self):
        """Loads individual .jpg files from the celebs folder and saves them to a pickle cache."""
        if os.path.exists(self.cache_file):
            print("⚡ Loading face data from cache...")
            with open(self.cache_file, "rb") as f:
                self.celeb_data = pickle.load(f)
            print(f"✅ Loaded {len(self.celeb_data)} celebs from cache.")
            return

        print("⏳ First-time setup: Building database from images...")
        files = [f for f in os.listdir(self.celebs_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        
        for filename in files:
            path = os.path.join(self.celebs_dir, filename)
            img = cv2.imread(path)
            if img is None: continue
            
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encs = face_recognition.face_encodings(rgb)
            
            if encs:
                lms = self.get_norm_lms(img)
                if lms is not None:
                    # Logic to clean name: 'taylor_swift_12345.jpg' -> 'Taylor Swift'
                    name_part = filename.rsplit('_', 1)[0]
                    name = name_part.replace('_', ' ').title()
                    
                    self.celeb_data.append({
                        "name": name, 
                        "enc": encs[0], 
                        "lms": lms, 
                        "img": cv2.resize(img, (200, 200))
                    })
                    print(f"✅ Processed: {name}")
                else:
                    print(f"⚠️  Landmarks failed for {filename}")

        # Save to pickle so next time is instant
        with open(self.cache_file, "wb") as f:
            pickle.dump(self.celeb_data, f)
        print(f"🚀 Database Ready: {len(self.celeb_data)} Celebrities.")

    def run(self):
        cap = cv2.VideoCapture(0)
        live_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
        window_name = 'Celebrity Lookalike'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            cam_h, cam_w = frame.shape[:2]
            
            # --- Responsive Window Math ---
            rect = cv2.getWindowImageRect(window_name)
            win_w, win_h = rect[2], rect[3]
            if win_w < 100: win_w, win_h = 1280, 720 

            sidebar_w = int(win_w * 0.22)
            main_w = win_w - sidebar_w
            canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)

            # --- Aspect Ratio Correction (Letterboxing) ---
            scale = min(main_w/cam_w, win_h/cam_h)
            new_w, new_h = int(cam_w * scale), int(cam_h * scale)
            feed_resized = cv2.resize(frame, (new_w, new_h))
            
            x_off = (main_w - new_w) // 2
            y_off = (win_h - new_h) // 2
            canvas[y_off:y_off+new_h, x_off:x_off+new_w] = feed_resized
            canvas[:, main_w:] = (25, 25, 25) # Dark sidebar

            # --- Facial Recognition ---
            rgb_live = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = live_mesh.process(rgb_live)

            if res.multi_face_landmarks and self.celeb_data:
                # Resize for speed
                small_rgb = cv2.resize(rgb_live, (0,0), fx=0.25, fy=0.25)
                face_encs = face_recognition.face_encodings(small_rgb)
                
                if face_encs:
                    dists = face_recognition.face_distance([c['enc'] for c in self.celeb_data], face_encs[0])
                    idx = np.argmin(dists)
                    self.current_match = self.celeb_data[idx]
                    
                    # Sidebar Image & Text
                    side_img_size = sidebar_w - 40
                    f_scale = sidebar_w / 350
                    if side_img_size > 20:
                        side_img = cv2.resize(self.current_match['img'], (side_img_size, side_img_size))
                        canvas[20:20+side_img_size, main_w+20:main_w+20+side_img_size] = side_img
                    
                    cv2.putText(canvas, self.current_match['name'], (main_w+20, side_img_size+60), 1, f_scale, (255,255,255), 2)
                    sim = max(0, (1 - dists[idx]) * 100)
                    cv2.putText(canvas, f"{sim:.1f}%", (main_w+20, side_img_size+110), 1, f_scale*2, (0, 255, 0), 3)
                    
                    # Heatmap Mapping
                    lm_list = res.multi_face_landmarks[0].landmark
                    curr_lms = np.array([(l.x, l.y) for l in lm_list])
                    curr_norm = curr_lms - curr_lms.mean(axis=0)
                    errors = np.linalg.norm(curr_norm - self.current_match['lms'], axis=1)

                    for conn in self.connections:
                        avg_err = (errors[conn[0]] + errors[conn[1]]) / 2
                        # Color logic: Green (good) to Red (bad)
                        g = max(0, 255 - int(avg_err * 8500))
                        r = min(255, int(avg_err * 8500))
                        
                        pt1 = (int(lm_list[conn[0]].x * new_w) + x_off, int(lm_list[conn[0]].y * new_h) + y_off)
                        pt2 = (int(lm_list[conn[1]].x * new_w) + x_off, int(lm_list[conn[1]].y * new_h) + y_off)
                        cv2.line(canvas, pt1, pt2, (0, g, r), 1)

            cv2.imshow(window_name, canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    CelebrityChallengeUltimate().run()
