import cv2
import numpy as np
import gradio as gr
import time
import mediapipe as mp

from undertone_detector import get_undertone
from mood_detector import get_mood
from shade_detector import get_shade_depth
from makeup_recommender import get_makeup_recommendation, select_products

last_process_time = 0.0
MIN_INTERVAL = 0.8 

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=2,          
    refine_landmarks=True,
    min_detection_confidence=0.4, 
    min_tracking_confidence=0.4
)

def process_analysis_frame(frame):
    global last_process_time
    
    if frame is None:
        return None, "No input detected", []


    now = time.time()
    if now - last_process_time < MIN_INTERVAL:
        return gr.skip(), gr.skip(), gr.skip()
    last_process_time = now

    try:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = face_mesh.process(frame) 

        if not results.multi_face_landmarks:
            return frame, "⚠️ **Face not found.**", []

        landmarks = results.multi_face_landmarks[0].landmark

        undertone = get_undertone(frame_bgr, draw_points=True)
        mood = get_mood(frame_bgr, landmarks)
        shade_result = get_shade_depth(frame_bgr, landmarks, draw=True)

        depth = shade_result["depth"] if shade_result else "UNKNOWN"

        rec_text = get_makeup_recommendation(undertone, mood, depth)
        products = select_products(undertone, mood, depth)

        color = (0, 140, 255) if undertone == "WARM" else (255, 0, 0) if undertone == "COOL" else (200, 200, 200)
        cv2.putText(frame_bgr, f"Undertone: {undertone}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        preview_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        md_output = f"### Analysis Results\n**Undertone:** {undertone} | **Depth:** {depth} | **Mood:** {mood.upper()}\n\n**Tip:** {rec_text}"
        
        return preview_rgb, md_output, products

    except Exception as e:
        print(f"Error in processing: {e}")
        return frame, f"Error: {str(e)}", []


css = """
.glow-effect {
    box-shadow: 0 0 30px 15px rgba(255, 255, 255, 0.7) !important;
    border: 3px solid white !important;
    border-radius: 15px !important;
}
/* Global font */
.gradio-container {
  font-family: 'Tinos', 'Segoe UI', serif;
}

h1, h2, h3 {
  font-family: 'Playfair Display', serif;
}
body { background-color: #1a1a2e !important; }
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("<h1 style='text-align: center; color: #ff69b4;'>UNDERTONE CHECKER</h1>")
    
    with gr.Row():
        input_choice = gr.Radio(["Webcam", "Upload"], value="Webcam", label="Choose Source")
        enable_glow = gr.Checkbox(label="Enable Ring Light", value=True)

    with gr.Row():
        with gr.Column():
        
            webcam_in = gr.Image(label="Live", sources=["webcam"], streaming=True, elem_classes=["glow-effect"])
            upload_in = gr.Image(label="Upload", sources=["upload"], visible=False)
            
        with gr.Column():
            analysis_preview = gr.Image(label="Analyzed Frame", interactive=False)
            output_text = gr.Markdown("Waiting for face...")

    product_gallery = gr.Gallery(label="Suggested Products", columns=3, height="auto")


    def swap_inputs(choice):
        if choice == "Webcam":
            return gr.update(visible=True), gr.update(visible=False)
        return gr.update(visible=False), gr.update(visible=True)

    input_choice.change(swap_inputs, input_choice, [webcam_in, upload_in])

 
    enable_glow.change(
        fn=lambda x: gr.update(elem_classes=["glow-effect"] if x else []),
        inputs=enable_glow,
        outputs=webcam_in
    )


    webcam_in.stream(process_analysis_frame, [webcam_in], [analysis_preview, output_text, product_gallery])
    upload_in.change(process_analysis_frame, [upload_in], [analysis_preview, output_text, product_gallery])

demo.launch(
    theme=gr.themes.Soft(primary_hue="pink", secondary_hue="pink"),
    share=True,
    server_name="0.0.0.0",
    server_port=7860,
    debug=True,
    allowed_paths=["/content/drive/MyDrive/undertone-ai-project/undertone-ai/data"]
)
