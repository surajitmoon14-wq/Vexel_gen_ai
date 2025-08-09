import os
import base64
import json
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from functools import wraps
from PIL import Image

# --- Firebase Admin SDK Setup ---
import firebase_admin
from firebase_admin import credentials, firestore, auth

# --- App Initialization & Configuration ---
app = Flask(__name__, template_folder='templates')
CORS(app)

# --- Configure API Keys & Firebase from Environment Secrets ---
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'a-very-secret-key-for-dev')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

# Initialize Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("Gemini API configured successfully.")
else:
    print("CRITICAL: GEMINI_API_KEY environment variable not set. Core features will fail.")

# Initialize Firebase Admin
try:
    service_account_json_str = os.environ.get('FIREBASE_SERVICE_ACCOUNT_JSON')
    if not service_account_json_str:
        raise ValueError("FIREBASE_SERVICE_ACCOUNT_JSON secret not found.")

    service_account_info = json.loads(service_account_json_str)
    cred = credentials.Certificate(service_account_info)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Firebase Admin SDK initialized successfully.")
except Exception as e:
    print(f"CRITICAL: Firebase Admin SDK failed to initialize. Error: {e}")
    db = None

# --- Authentication Decorator ---
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split(" ")[1]
        if not token:
            return f(None, *args, **kwargs)
        try:
            decoded_token = auth.verify_id_token(token)
            current_user = {'uid': decoded_token['uid']}
        except Exception as e:
            print(f"Token verification failed: {e}")
            return f(None, *args, **kwargs)
        return f(current_user, *args, **kwargs)
    return decorated

# --- Frontend Serving Route ---
@app.route('/')
def index():
    return render_template('index.html')

# --- Core API Routes (Powered by Gemini) ---

@app.route('/generate', methods=['POST'])
@token_required
def generate_anime(current_user):
    if not GEMINI_API_KEY:
        return jsonify({'error': 'Server is not configured for image generation.'}), 500

    prompt = request.form.get('prompt')
    if not prompt:
        return jsonify({'error': 'Prompt is required.'}), 400

    try:
        image_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        generation_prompt = f"Generate a high-quality, artistic anime-style image of: {prompt}. Style: digital painting, masterpiece, best quality, vibrant colors."
        response = image_model.generate_content(generation_prompt)

        if not response.parts:
            if response.prompt_feedback.block_reason:
                raise ValueError(f"Image generation blocked due to: {response.prompt_feedback.block_reason.name}")
            else:
                raise ValueError("Image generation failed for an unknown reason.")

        image_part = response.parts[0]
        if not hasattr(image_part, 'inline_data'):
             raise ValueError("Could not extract image data from the AI response.")

        img_bytes = image_part.inline_data.data
        img_str = base64.b64encode(img_bytes).decode('utf-8')
        mime_type = image_part.inline_data.mime_type
        image_data_url = f'data:{mime_type};base64,{img_str}'

        # --- FIXED: Only save the AI's response to Firestore ---
        if current_user and db:
            chat_id = request.form.get('chat_id')
            if chat_id:
                chat_ref = db.collection('users').document(current_user['uid']).collection('chats').document(chat_id)
                chat_ref.update({
                    'messages': firestore.ArrayUnion([
                        {'sender': 'ai', 'content': image_data_url, 'type': 'image', 'timestamp': firestore.SERVER_TIMESTAMP}
                    ]),
                    'lastUpdated': firestore.SERVER_TIMESTAMP
                })

        return jsonify({'image_url': image_data_url})

    except Exception as e:
        print(f"Error in /generate: {e}")
        return jsonify({'error': f"An error occurred with the AI model: {e}"}), 500

@app.route('/chat', methods=['POST'])
@token_required
def handle_chat(current_user):
    if not GEMINI_API_KEY:
        return jsonify({'error': 'Server is not configured for chat.'}), 500

    data = request.get_json()
    user_prompt = data.get('prompt')
    if not user_prompt:
        return jsonify({'error': 'A prompt is required.'}), 400

    try:
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        system_prompt = (
            "You are a helpful and friendly chat assistant. "
            "If the user asks a math or equation-related question, solve it and provide the solution in a simple, clean HTML format. Use <p> for text and a <pre> tag with a dark background for the final equation or result. "
            "For all other questions, provide a conversational and helpful response in plain text or simple HTML."
        )

        response = model.generate_content(f"{system_prompt}\n\nUser: {user_prompt}")
        solution_html = response.text

        # --- FIXED: Only save the AI's response to Firestore ---
        if current_user and db:
            chat_id = data.get('chat_id')
            if chat_id:
                chat_ref = db.collection('users').document(current_user['uid']).collection('chats').document(chat_id)
                chat_ref.update({
                    'messages': firestore.ArrayUnion([
                        {'sender': 'ai', 'content': solution_html, 'type': 'text', 'timestamp': firestore.SERVER_TIMESTAMP}
                    ]),
                    'lastUpdated': firestore.SERVER_TIMESTAMP
                })

        return jsonify({'solution': solution_html})

    except Exception as e:
        print(f"Error during chat: {e}")
        error_html = f"<p>Could not process your request for: <b>{user_prompt}</b></p><p class='mt-2 text-red-400'>Error: The AI model failed to respond.</p>"
        return jsonify({'error': error_html}), 500

# --- Main Execution ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
