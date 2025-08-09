import os
import json
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from functools import wraps

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
    print("Gemini API key loaded successfully.")
else:
    print("CRITICAL: GEMINI_API_KEY environment variable not set. Core features will fail.")

# Initialize Firebase Admin
try:
    service_account_json_str = os.environ.get('FIREBASE_SERVICE_ACCOUNT_JSON')
    if not service_account_json_str:
        print("WARNING: FIREBASE_SERVICE_ACCOUNT_JSON secret not found. Firebase features will be disabled.")
        db = None
    else:
        # Clean up the JSON string - remove any extra whitespace or formatting issues
        service_account_json_str = service_account_json_str.strip()
        
        # Try to parse the JSON
        service_account_info = json.loads(service_account_json_str)
        
        # Validate required fields
        required_fields = ['type', 'project_id', 'private_key', 'client_email']
        for field in required_fields:
            if field not in service_account_info:
                raise ValueError(f"Missing required field '{field}' in service account JSON")
        
        cred = credentials.Certificate(service_account_info)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("Firebase Admin SDK initialized successfully.")
except json.JSONDecodeError as e:
    print(f"CRITICAL: Invalid JSON in FIREBASE_SERVICE_ACCOUNT_JSON. Error: {e}")
    print("Please check that your Firebase service account JSON is properly formatted.")
    db = None
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
        return jsonify({'error': 'Server is not configured for text generation.'}), 500

    prompt = request.form.get('prompt')
    if not prompt:
        return jsonify({'error': 'Prompt is required.'}), 400

    try:
        # Initialize the Gemini model with error handling
        try:
            # Try the standard import first
            model = genai.GenerativeModel('gemini-1.5-flash')
        except AttributeError:
            # Fallback for older versions
            try:
                model = genai.GenerativeModel(model_name='gemini-1.5-flash')
            except Exception as e:
                print(f"Error initializing GenerativeModel: {e}")
                return jsonify({'error': 'AI model initialization failed.'}), 500
        except Exception as e:
            print(f"Error initializing GenerativeModel: {e}")
            return jsonify({'error': 'AI model initialization failed.'}), 500

        generation_prompt = f"Create a detailed description for an anime-style artwork of: {prompt}. Include artistic details like color palette, composition, style, and mood."
        response = model.generate_content(generation_prompt)

        description = response.text

        # Save the AI's response to Firestore
        if current_user and db:
            chat_id = request.form.get('chat_id')
            if chat_id:
                chat_ref = db.collection('users').document(current_user['uid']).collection('chats').document(chat_id)
                chat_ref.update({
                    'messages': firestore.ArrayUnion([
                        {'sender': 'ai', 'content': description, 'type': 'text', 'timestamp': firestore.SERVER_TIMESTAMP}
                    ]),
                    'lastUpdated': firestore.SERVER_TIMESTAMP
                })

        return jsonify({'description': description})

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
        # Initialize the Gemini model with error handling
        try:
            # Try the standard import first
            model = genai.GenerativeModel('gemini-1.5-flash')
        except AttributeError:
            # Fallback for older versions
            try:
                model = genai.GenerativeModel(model_name='gemini-1.5-flash')
            except Exception as e:
                print(f"Error initializing GenerativeModel: {e}")
                return jsonify({'error': 'AI model initialization failed.'}), 500
        except Exception as e:
            print(f"Error initializing GenerativeModel: {e}")
            return jsonify({'error': 'AI model initialization failed.'}), 500

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