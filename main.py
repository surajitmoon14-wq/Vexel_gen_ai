# --- NEW IMPORTS FOR LOGIN SYSTEM ---
from flask import redirect, url_for, session, flash
from flask_bcrypt import Bcrypt
from replit import db
# ------------------------------------

import os
import base64
import requests
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# --- App Initialization & Configuration ---
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# --- NEW CONFIGURATION FOR LOGIN SYSTEM ---
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY')
bcrypt = Bcrypt(app)
# ----------------------------------------

# --- Configure API Keys from Environment Secrets (Your code, unchanged) ---
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
STABILITY_API_KEY = os.environ.get('STABILITY_API_KEY')

# --- Initialize Gemini for Chat (Your code, unchanged) ---
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("Gemini API configured successfully for Chat.")
else:
    print(
        "WARNING: GEMINI_API_KEY not set. Chat/Summarize features will not work."
    )

# --- Check for Stability AI Key (Your code, unchanged) ---
if not STABILITY_API_KEY:
    print("WARNING: STABILITY_API_KEY not set. Image Generator will not work.")


# --- MODIFIED Frontend Serving Route (This is the necessary change to protect your app) ---
@app.route('/')
def index():
    """Checks if user is logged in before showing the main Vexel AI app."""
    if 'username' in session:
        # If logged in, show the app and pass the username to the template
        return render_template('index.html', username=session['username'])

    # If not logged in, redirect them to the login page
    return redirect(url_for('login'))


# --- Image Generation Route (Your code, unchanged) ---
@app.route('/generate', methods=['POST'])
def generate_image():
    """Handles image generation requests using the Stability AI API."""
    if not STABILITY_API_KEY:
        return jsonify(
            {'error':
             'Server is not configured with a Stability AI API key.'}), 500

    prompt = request.form.get('prompt')
    if not prompt:
        return jsonify({'error': 'Prompt is required.'}), 400

    try:
        api_host = "https://api.stability.ai/v2beta/stable-image/generate/core"
        response = requests.post(api_host,
                                 headers={
                                     "Authorization":
                                     f"Bearer {STABILITY_API_KEY}",
                                     "Accept": "application/json"
                                 },
                                 files={"none": ''},
                                 data={
                                     "prompt": prompt,
                                     "output_format": "png"
                                 },
                                 timeout=45)
        response.raise_for_status()
        response_data = response.json()
        image_b64 = response_data.get("image")
        if not image_b64:
            raise ValueError("No image data received from Stability AI.")
        image_data_url = f'data:image/png;base64,{image_b64}'
        return jsonify({'image_url': image_data_url})

    except requests.exceptions.RequestException as e:
        error_details = e.response.text if e.response else str(e)
        print(f"Error calling Stability AI API: {error_details}")
        return jsonify(
            {'error':
             f"Failed to generate image. Details: {error_details}"}), 503
    except Exception as e:
        print(f"Error in /generate: {e}")
        return jsonify({'error':
                        f"An unexpected error occurred: {str(e)}"}), 500


# --- Text Summarization Route (Your code, unchanged) ---
@app.route('/summarize', methods=['POST'])
def summarize_text():
    """Handles text summarization requests."""
    if not GEMINI_API_KEY:
        return jsonify(
            {'error': 'Server is not configured for summarization.'}), 500

    data = request.get_json()
    text_to_summarize = data.get('text')
    if not text_to_summarize:
        return jsonify({'error': 'Text to summarize is required.'}), 400

    try:
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        prompt = f"Please provide a concise summary of the following conversation or text:\n\n---\n\n{text_to_summarize}"
        response = model.generate_content(prompt)
        return jsonify({'summary': response.text})
    except Exception as e:
        print(f"Error during summarization: {e}")
        return jsonify({'error': "Failed to summarize the text."}), 500


# --- UPGRADED: Chat Assistant Route (Your code, unchanged) ---
@app.route('/chat', methods=['POST'])
def handle_chat():
    """Handles text-based chat with different tones and file context."""
    if not GEMINI_API_KEY:
        return jsonify({'error': 'Server is not configured for chat.'}), 500

    data = request.get_json()
    user_prompt = data.get('prompt')
    tone = data.get('tone', 'default')  # 'default', 'formal', 'fun', or custom
    file_content = data.get('file_content', None)

    if not user_prompt:
        return jsonify({'error': 'A prompt is required.'}), 400

    # --- IMPROVED PERSONA ---
    # Define system prompts based on the selected tone
    system_prompts = {
        'formal':
        "You are a professional, formal, and highly articulate assistant. Provide precise, well-structured, and serious responses.",
        'fun':
        "You are a witty, fun-loving, and creative assistant. Use humor, emojis, and a lighthearted tone in your responses.",
        'default':
        "You are Vexel AI, a helpful and friendly assistant. Your tone should be conversational and informative, but not overly formal or casual. Provide clear and direct answers."
    }

    # If the tone is custom, use it directly. Otherwise, look it up.
    if tone in system_prompts:
        system_prompt = system_prompts[tone]
    else:
        system_prompt = tone  # A custom persona prompt provided by the user

    # Construct the final prompt, including file context if it exists
    final_prompt = f"{system_prompt}\n\n"
    if file_content:
        final_prompt += f"Based on the content of the attached file below, please answer the user's question.\n\n[File Content]:\n{file_content}\n\n---\n\n"
    final_prompt += f"User: {user_prompt}"

    try:
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        response = model.generate_content(final_prompt)
        return jsonify({'solution': response.text})

    except Exception as e:
        print(f"Error during chat: {e}")
        error_html = f"<p>Could not process your request. The AI model failed to respond.</p>"
        return jsonify({'error': error_html}), 500


# --- NEW ROUTES FOR SIGNUP, LOGIN, AND LOGOUT ---
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Handles user registration."""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in db.keys():
            flash("Username already exists! Please choose another.", "danger")
            return redirect(url_for('signup'))

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        db[username] = hashed_password

        flash("Account created successfully! Please log in.", "success")
        return redirect(url_for('login'))

    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handles user login."""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in db.keys() and bcrypt.check_password_hash(db[username], password):
            session['username'] = username # This line logs the user in
            return redirect(url_for('index'))
        else:
            flash("Invalid username or password.", "danger")
            return redirect(url_for('login'))

    return render_template('login.html')


@app.route('/logout')
def logout():
    """Logs the user out."""
    session.pop('username', None)
    return redirect(url_for('login'))
# ---------------------------------------------


# --- Main Execution (Your code, unchanged) ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)