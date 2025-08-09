import os
import json
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from functools import wraps

# --- Firebase Admin SDK Setup ---
import firebase_admin
from firebase_admin import credentials, firestore, auth
# FIX: Explicitly import ArrayUnion and SERVER_TIMESTAMP
from firebase_admin.firestore import ArrayUnion, SERVER_TIMESTAMP

# --- App Initialization & Configuration ---
app = Flask(__name__) 
CORS(app)

# --- Configure API Keys & Firebase from Environment Secrets ---
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'a-very-secret-key-for-dev')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

# --- Initialize Gemini and AI Model (ONCE) ---
ai_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        ai_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        print("✅ Gemini API configured and model created successfully.")
    except Exception as e:
        print(f"❌ CRITICAL: Failed to configure Gemini. Error: {e}")
else:
    print("❌ CRITICAL: GEMINI_API_KEY environment variable not set.")

# --- Initialize Firebase Admin ---
db = None
try:
    service_account_json_str = os.environ.get('FIREBASE_SERVICE_ACCOUNT_JSON')
    if not service_account_json_str:
        raise ValueError("FIREBASE_SERVICE_ACCOUNT_JSON secret not found.")
    
    # Strip whitespace and check for common issues
    service_account_json_str = service_account_json_str.strip()
    if not service_account_json_str.startswith('{'):
        raise ValueError("JSON string does not start with '{'")
    
    service_account_info = json.loads(service_account_json_str)
    cred = credentials.Certificate(service_account_info)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("✅ Firebase Admin SDK initialized successfully.")
except json.JSONDecodeError as e:
    print(f"⚠️ WARNING: Firebase JSON is malformed. Error at line {e.lineno}, column {e.colno}: {e.msg}")
    print("Please check that your FIREBASE_SERVICE_ACCOUNT_JSON secret contains valid JSON with proper double quotes around property names.")
    db = None
except Exception as e:
    print(f"⚠️ WARNING: Firebase Admin SDK failed to initialize. Chat history will not be saved. Error: {e}")
    db = None

# --- Authentication Decorator ---
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if auth_header:
            parts = auth_header.split()
            if len(parts) == 2 and parts[0].lower() == 'bearer':
                token = parts[1]
                try:
                    decoded_token = auth.verify_id_token(token)
                    current_user = {'uid': decoded_token['uid']}
                    return f(current_user, *args, **kwargs)
                except Exception as e:
                    print(f"⚠️ Token verification failed: {e}")
        return f(None, *args, **kwargs)
    return decorated

# --- Frontend Serving Route ---
@app.route('/')
def index():
    return render_template('index.html')

# --- Helper Function to Save Messages ---
def save_message(current_user, chat_id, sender, content, msg_type='text'):
    """Saves a message to Firestore if user is logged in and db is available."""
    if not (current_user and db and chat_id):
        return

    try:
        chat_ref = db.collection('users').document(current_user['uid']).collection('chats').document(chat_id)
        chat_ref.update({
            'messages': ArrayUnion([
                {'sender': sender, 'content': content, 'type': msg_type, 'timestamp': SERVER_TIMESTAMP}
            ]),
            'lastUpdated': SERVER_TIMESTAMP
        })
    except Exception as e:
        print(f"❌ Error saving message to Firestore for chat {chat_id}: {e}")

# --- Core API Routes (Powered by Gemini) ---

@app.route('/generate', methods=['POST'])
@token_required
def generate_anime(current_user):
    if not ai_model:
        return jsonify({'error': 'Server is not configured for AI generation.'}), 500

    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({'error': 'A "prompt" is required in the JSON body.'}), 400
    prompt = data['prompt']
    chat_id = data.get('chat_id')

    # Save the user's prompt message before generating the AI response.
    save_message(current_user, chat_id, 'user', prompt)

    try:
        generation_prompt = f"Create a detailed description for an anime-style artwork of: {prompt}. Include artistic details like color palette, composition, style, and mood."
        response = ai_model.generate_content(generation_prompt)
        description = response.text

        # Save the AI's response message
        save_message(current_user, chat_id, 'ai', description)

        return jsonify({'description': description})

    except Exception as e:
        print(f"❌ Error in /generate: {e}")
        error_msg = "An internal error occurred with the AI model."
        save_message(current_user, chat_id, 'ai', error_msg)
        return jsonify({'error': error_msg}), 500

@app.route('/chat', methods=['POST'])
@token_required
def handle_chat(current_user):
    if not ai_model:
        return jsonify({'error': 'Server is not configured for chat.'}), 500

    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({'error': 'A "prompt" is required in the JSON body.'}), 400
    user_prompt = data['prompt']
    chat_id = data.get('chat_id')

    # Save the user's prompt message.
    save_message(current_user, chat_id, 'user', user_prompt)

    try:
        system_prompt = (
            "You are a helpful and friendly chat assistant. "
            "Format math solutions in simple HTML using <p> and <pre> tags."
        )
        response = ai_model.generate_content(f"{system_prompt}\n\nUser: {user_prompt}")
        solution_html = response.text

        # Save the AI's response message
        save_message(current_user, chat_id, 'ai', solution_html)

        return jsonify({'solution': solution_html})

    except Exception as e:
        print(f"❌ Error during chat: {e}")
        error_msg = "An internal error occurred while processing the chat."
        save_message(current_user, chat_id, 'ai', error_msg)
        return jsonify({'error': error_msg}), 500

# --- Main Execution ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
