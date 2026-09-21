import os
import json
import random
from PIL import Image
import numpy as np


BASE_DATA_PATH = "/content/drive/MyDrive/undertone-ai-project/undertone-ai/data"

JSON_PATH = os.path.join(BASE_DATA_PATH, "makeup", "makeup_json")

MAKEUP_DATABASE = []

def load_makeup_data():
    global MAKEUP_DATABASE
    MAKEUP_DATABASE = []
    json_files = ["hur_blush.json", "revlon_lipstick.json", "tirtir_cushion.json"]

    JSON_PATH = os.path.join(BASE_DATA_PATH, "makeup", "makeup_json")
    
    for file_name in json_files:
        full_path = os.path.join(JSON_PATH, file_name)
        if os.path.exists(full_path):
            with open(full_path, 'r') as f:
                try:
                    data = json.load(f)
                   
                    if isinstance(data, list):
                        MAKEUP_DATABASE.extend(data)
                    else:
                        MAKEUP_DATABASE.append(data)
                except Exception as e:
                    print(f"❌ Error loading {file_name}: {e}")
    

    print(f"✅ Database Loaded: {len(MAKEUP_DATABASE)} products ready.")
    

load_makeup_data()


def get_makeup_recommendation(undertone, mood, depth):
    return f"✨ Analysis Complete: You have a **{undertone}** undertone with **{depth}** skin depth. We recommend a **{mood}** makeup style!"


def select_products(undertone, mood, depth):

    categories = {"cushion": [], "lipstick": [], "blush": []}
    
    u_search = str(undertone).upper().strip() 
    m_search = str(mood).lower().strip()      
    d_search = str(depth).upper().strip()     

    print(f"🔍 Searching: {u_search} | {m_search} | {d_search}")

    for product in MAKEUP_DATABASE:
        brand = product.get("brand", "").lower()
        

        if "tirtir" in brand: prod_type = "cushion"
        elif "revlon" in brand: prod_type = "lipstick"
        else: prod_type = "blush"
        
        shades = product.get("shades", [])
        for shade in shades:

            shade_u_list = [str(u).upper().strip() for u in shade.get("undertone", [])]
            u_match = (u_search in shade_u_list) or ("NEUTRAL" in shade_u_list)
            
            shade_moods = [str(m).lower().strip() for m in shade.get("mood", [])]
            m_match = (m_search in shade_moods) if shade_moods else True
            
            shade_d = str(shade.get("depth", "")).upper().strip()
 
            if prod_type == "cushion":
                d_match = (d_search == shade_d)
            else:
                d_match = True 

            if u_match and m_match and d_match:
                img_rel_path = shade.get("image", "")
                
                
                full_img_path = os.path.join(BASE_DATA_PATH, "makeup", img_rel_path)
                
                if os.path.exists(full_img_path):
                    categories[prod_type].append((full_img_path, f"{product.get('brand')}: {shade.get('name')}"))
                else:
                   
                    alt_path = os.path.join(BASE_DATA_PATH, "makeup", img_rel_path.lower())
                    if os.path.exists(alt_path):
                        categories[prod_type].append((alt_path, f"{product.get('brand')}: {shade.get('name')}"))
                    else:
                        print(f"❌ Still Not Found: {full_img_path}")


    final_results = []

    for cat in ["cushion", "lipstick", "blush"]:
        if categories[cat]:
            pick = random.choice(categories[cat])
            final_results.append(pick)
            categories[cat].remove(pick)


    pool = categories["cushion"] + categories["lipstick"] + categories["blush"]
    random.shuffle(pool)
    
    for item in pool:
        if len(final_results) < 6:
            final_results.append(item)

    print(f"✨ Success! Found {len(final_results)} items.")
    return final_results