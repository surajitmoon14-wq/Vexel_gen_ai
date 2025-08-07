import os
import requests
import base64
import io  # Required to handle image data in memory
from flask import Flask, render_template, request
from PIL import Image  # Import the Pillow library for image manipulation

# --- Setup ---
app = Flask(__name__)

# Get the API key from Replit Secrets
api_key = os.environ.get("API_KEY")
api_host = 'https://api.stability.ai'
engine_id = "stable-diffusion-v1-6"

# --- Web Routes ---

# This route shows the main page
@app.route('/')
def home():
    return render_template('index.html')

# This route is called when the user clicks "Generate Image"
@app.route('/generate', methods=['POST'])
def generate():
    # --- 1. Get User Input from the Form ---
    user_prompt = request.form.get('prompt')
    overlay_file = request.files.get('overlay_image') # Get the uploaded file

    # Add style details to the prompt for better results
    full_prompt = f"anime artwork, anime style, key visual, vibrant, studio anime, highly detailed, {user_prompt}"

    # --- 2. Call the Stability AI API to Generate the Base Image ---
    response = requests.post(
        f"{api_host}/v1/generation/{engine_id}/text-to-image",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        json={
            "text_prompts": [{"text": full_prompt}],
            "cfg_scale": 7, "height": 512, "width": 512,
            "samples": 1, "steps": 30,
        },
    )

    # Handle API errors
    if response.status_code != 200:
        return render_template('index.html', error="API call failed. Check your key or the API server status.")

    data = response.json()
    base_image_data = None
    for image in data["artifacts"]:
        base_image_data = base64.b64decode(image["base64"])

    if not base_image_data:
        return render_template('index.html', error="Failed to generate the base image.")

    # --- 3. Combine the Images using Pillow ---
    # Open the generated image from its raw data
    base_image = Image.open(io.BytesIO(base_image_data)).convert("RGBA")

    # Check if the user actually uploaded a file
    if overlay_file and overlay_file.filename != '':
        # Open the uploaded image
        overlay_image = Image.open(overlay_file.stream).convert("RGBA")

        # Resize the overlay to be smaller (e.g., 150x150 pixels)
        overlay_image = overlay_image.resize((150, 150))

        # Define where to paste it (top-right corner with 10px padding)
        position = (base_image.width - overlay_image.width - 10, 10)

        # Paste the overlay onto the base image. The 'overlay_image' is used
        # as a mask to handle transparent backgrounds correctly.
        base_image.paste(overlay_image, position, overlay_image)

    # --- 4. Save and Display the Final Image ---
    final_image_path = "static/generated/final_output.png"
    base_image.save(final_image_path)

    # Send the path to the final image back to the HTML
    return render_template('index.html', image_url="/" + final_image_path, prompt=user_prompt)

# --- Run the App ---
if __name__ == '__main__':
    # This makes the app accessible in Replit's webview
    app.run(host='0.0.0.0', port=81)