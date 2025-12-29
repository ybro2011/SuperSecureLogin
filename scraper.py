import os
import requests
import cv2
import mediapipe as mp
import numpy as np
import time
import random
import sys
from duckduckgo_search import DDGS

# --- CONFIGURATION ---
# Your provided TMDB API Key
TMDB_API_KEY = "928d45ccb3dae4dcce45dbba02d64ca2" 

class UltimateScraper:
    def __init__(self):
        self.base_dir = "celebs"
        if not os.path.exists(self.base_dir): os.makedirs(self.base_dir)
        
        # Initialize Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(
            model_selection=1, 
            min_detection_confidence=0.6
        )
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }

    def is_face_present(self, image_bytes):
        """Verifies if a face is clearly visible in the downloaded image."""
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None: return False
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.face_detector.process(rgb_img)
            return results.detections is not None
        except: return False

    def get_tmdb_image(self, name):
        """Source 1: TMDB (Professional Database). Reliable and high-res."""
        try:
            url = f"https://api.themoviedb.org/3/search/person"
            params = {"api_key": TMDB_API_KEY, "query": name}
            data = requests.get(url, params=params, timeout=5).json()
            if data.get('results'):
                path = data['results'][0].get('profile_path')
                if path:
                    img_url = f"https://image.tmdb.org/t/p/h632{path}"
                    resp = requests.get(img_url, timeout=10)
                    if resp.status_code == 200: return resp.content
            return None
        except: return None

    def get_wikipedia_image(self, name):
        """Source 2: Wikipedia API. Great for non-Hollywood icons."""
        try:
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query", "titles": name, "prop": "pageimages",
                "format": "json", "pithumbsize": 1000, "redirects": 1
            }
            resp = requests.get(url, params=params, timeout=5).json()
            pages = resp.get("query", {}).get("pages", {})
            for p in pages.values():
                if "thumbnail" in p:
                    img_resp = requests.get(p["thumbnail"]["source"], timeout=10)
                    if img_resp.status_code == 200: return img_resp.content
            return None
        except: return None

    def draw_progress_bar(self, current, total, status_text=""):
        fraction = current / total
        bar = ('=' * int(fraction * 30)).ljust(30)
        sys.stdout.write(f'\rProgress: [{bar}] {current}/{total} ({int(fraction*100)}%) | {status_text[:30].ljust(30)}')
        sys.stdout.flush()

    def scrape(self, celeb_list):
        total = len(celeb_list)
        print(f"🚀 Starting Tri-Source Stealth Scrape for {total} celebrities...")
        
        with DDGS() as ddgs:
            for i, name in enumerate(celeb_list):
                clean_name = name.replace(" ", "_").lower()
                
                # Skip if already exists
                if any(f.startswith(clean_name) for f in os.listdir(self.base_dir)):
                    self.draw_progress_bar(i + 1, total, f"Skipped: {name}")
                    continue

                self.draw_progress_bar(i + 1, total, f"Searching: {name}")
                img_data = None

                # 1. TRY TMDB FIRST (Cleanest data)
                img_data = self.get_tmdb_image(name)
                
                # 2. TRY WIKIPEDIA SECOND (Solid backup)
                if not img_data or not self.is_face_present(img_data):
                    img_data = self.get_wikipedia_image(name)

                # 3. TRY DUCKDUCKGO LAST (Use only if others fail)
                if not img_data or not self.is_face_present(img_data):
                    try:
                        results = list(ddgs.images(keywords=f"{name} headshot portrait", max_results=2))
                        for r in results:
                            resp = requests.get(r['image'], timeout=5, headers=self.headers)
                            if resp.status_code == 200 and self.is_face_present(resp.content):
                                img_data = resp.content
                                break
                    except: pass

                # SAVE IF FOUND
                if img_data and self.is_face_present(img_data):
                    filename = f"{clean_name}_{int(time.time())}.jpg"
                    with open(os.path.join(self.base_dir, filename), "wb") as f:
                        f.write(img_data)
                else:
                    print(f"\n❌ Failed to find face for {name}")

                # Random rest between 2 to 4 seconds to look human
                time.sleep(random.uniform(2, 4))
        print("\n✨ Scrape complete!")

if __name__ == "__main__":
    # Your full list of 100+ celebrities
    top_celebs = ["Taylor Swift", "Ariana Grande", "Billie Eilish", "Justin Bieber", "Dua Lipa", "Rihanna", "Beyonce", "Lady Gaga", "Shakira", "Drake", "The Weeknd", "Harry Styles", "Olivia Rodrigo", "Selena Gomez", "Kanye West", "Ed Sheeran", "Adele", "Bruno Mars", "Nicki Minaj", "Cardi B", "Bad Bunny", "Rosé", "J Balvin", "Travis Scott", "Zayn Malik", "Doja Cat", "Hozier", "Myles Smith", "Zendaya", "Tom Holland", "Robert Downey Jr", "Scarlett Johansson", "Dwayne Johnson", "Tom Cruise", "Brad Pitt", "Leonardo DiCaprio", "Margot Robbie", "Jennifer Lawrence", "Ryan Reynolds", "Blake Lively", "Timothee Chalamet", "Jenna Ortega", "Cillian Murphy", "Pedro Pascal", "Keanu Reeves", "Will Smith", "Angelina Jolie", "Jennifer Lopez", "Gal Gadot", "Chris Evans", "Chris Hemsworth", "Johnny Depp", "Emma Watson", "Sydney Sweeney", "Millie Bobby Brown", "Ryan Gosling", "Ben Affleck", "Jennifer Aniston", "Adam Sandler", "Caitlin Clark", "Austin Butler", "Jacob Elordi", "Cristiano Ronaldo", "Lionel Messi", "LeBron James", "Stephen Curry", "Lewis Hamilton", "Max Verstappen", "Novak Djokovic", "Serena Williams", "Tiger Woods", "Virat Kohli", "Neymar Jr", "Kylian Mbappe", "Shohei Ohtani", "Patrick Mahomes", "Simone Biles", "Charles Leclerc", "Roger Federer", "Kevin Durant", "Giannis Antetokounmpo", "Elon Musk", "Jeff Bezos", "Bill Gates", "Mark Zuckerberg", "Sam Altman", "Dario Amodei", "Demis Hassabis", "Lisa Su", "Jensen Huang", "Oprah Winfrey", "Kim Kardashian", "Kylie Jenner", "Kendall Jenner", "Joe Rogan", "Jimmy Donaldson", "Khloe Kardashian", "Kris Jenner", "Gigi Hadid", "Bella Hadid", "Donald Trump", "Kamala Harris", "Kate Middleton", "Sabrina Carpenter"]
    
    UltimateScraper().scrape(top_celebs)