# Vexel AI - Complete Python Backend (with Live AI and File Handling)
# This script provides a fully functional Flask backend for the Vexel AI application.
# It handles user authentication, chat history, settings, file uploads, and live AI model interactions.
#
# To run this:
# 1. Make sure you have Python installed.
# 2. Install the required libraries: pip install Flask Flask-SQLAlchemy Flask-Login Werkzeug requests python-dotenv
# 3. Create a file named .env in the same directory as this script and add your Gemini API key:
#    GEMINI_API_KEY=your_actual_api_key_here
# 4. Create the following folder structure:
#    - static/
#      - models/  (for face-api.js models)
#      - bot.png
#    - templates/
#      - index.html
#      - login.html
#      - signup.html (or similar for registration)
#    - uploads/ (for user file attachments)
# 5. Run this script: python your_app_name.py

import os
import json
import @app.route('/login')
def login():
    return redirect('/signin')  # Redirect to the Google Sign-in# Add this route for login
@app.route('/login')
def login():
    return redirect('/signin')  # Redirect to the Google Sign-in@app.route('/login')
def login():
    # your login logic here
    passbase64
import requests
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- App Configuration ---
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///vexel_ai.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login_page'

# --- Gemini AI Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
IMAGE_GEN_MODEL = "imagen-3.0-generate-002"
CHAT_MODEL = "gemini-2.5-flash-preview-05-20"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# --- Database Models ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    settings = db.relationship('Settings', backref='user', uselist=False, cascade="all, delete-orphan")
    chats = db.relationship('Chat', backref='user', lazy=True, cascade="all, delete-orphan")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Settings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    settings_json = db.Column(db.Text, nullable=False, default='{}')

    def to_dict(self):
        default = {
            'font': 'Inter', 'fontSize': '16', 'displayName': 'User', 'avatarUrl': '',
            'userTextColor': '#ffffff', 'aiTextColor': '#e5e7eb', 'sidebarInputBg': '#1f2937',
            'userBubble': 'transparent', 'aiBubble': 'transparent', 'bg1': '#4a0e69',
            'bg2': '#d946ef', 'bg3': '#1a1a2e', 'bg4': '#ec4899', 'tone': 'default',
            'customTone': '', 'theme': 'dark'
        }
        user_settings = json.loads(self.settings_json)
        default.update(user_settings)
        return default

class Chat(db.Model):
    id = db.Column(db.String(100), primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(150), nullable=False, default="New Chat")
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    messages = db.relationship('Message', backref='chat', lazy=True, cascade="all, delete-orphan")

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    chat_id = db.Column(db.String(100), db.ForeignKey('chat.id'), nullable=False)
    sender = db.Column(db.String(20), nullable=False)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    message_type = db.Column(db.String(20), default='text')

# --- User Loader ---
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- AI Helper Function ---
def call_gemini_api(model, payload):
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set in the environment.")
    headers = {'Content-Type': 'application/json'}
    method = "predict" if "imagen" in model else "generateContent"
    api_url = f"{GEMINI_API_URL}/{model}:{method}?key={GEMINI_API_KEY}"
    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        try:
            error_details = e.response.json()
            message = error_details.get("error", {}).get("message", str(e))
            raise ConnectionError(f"Failed to connect to AI service: {message}")
        except (ValueError, AttributeError):
            raise ConnectionError(f"Failed to connect to AI service: {e}")

# --- HTML Rendering and Static File Routes ---
@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        data = request.get_json()
        user = User.query.filter_by(email=data.get('email')).first()
        if user and user.check_password(data.get('password')):
            login_user(user, remember=True)
            session.permanent = True
            return jsonify({'message': 'Login successful'}), 200
        return jsonify({'error': 'Invalid email or password'}), 401
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register_page():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        data = request.get_json()
        if User.query.filter_by(email=data.get('email')).first():
            return jsonify({'error': 'Email address already registered'}), 409
        new_user = User(email=data.get('email'))
        new_user.set_password(data.get('password'))
        db.session.add(new_user)
        db.session.add(Settings(user=new_user, settings_json='{}'))
        db.session.commit()
        login_user(new_user, remember=True)
        return jsonify({'message': 'Registration successful'}), 201
    # Serve the signup page for GET requests
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login_page'))

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

# --- API Routes ---
@app.route('/settings', methods=['GET', 'POST'])
@login_required
def user_settings():
    settings = Settings.query.filter_by(user_id=current_user.id).first_or_404()
    if request.method == 'POST':
        settings.settings_json = json.dumps(request.get_json())
        db.session.commit()
        return jsonify({'message': 'Settings saved'}), 200
    return jsonify(settings.to_dict())

@app.route('/history', methods=['GET'])
@login_required
def get_history():
    chats = Chat.query.filter_by(user_id=current_user.id).order_by(Chat.timestamp.desc()).all()
    grouped = {}
    today = datetime.utcnow().date()
    for chat in chats:
        group = "Today" if chat.timestamp.date() == today else "Yesterday" if chat.timestamp.date() == today - timedelta(days=1) else chat.timestamp.strftime('%B %d, %Y')
        if group not in grouped: grouped[group] = []
        grouped[group].append({'id': chat.id, 'title': chat.title})
    return jsonify(grouped)

@app.route('/history/clear', methods=['POST'])
@login_required
def clear_history():
    Chat.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()
    return jsonify({'message': 'History cleared'})

@app.route('/chat/<string:chat_id>', methods=['GET', 'DELETE'])
@login_required
def manage_chat(chat_id):
    chat = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first_or_404()
    if request.method == 'DELETE':
        db.session.delete(chat)
        db.session.commit()
        return jsonify({'message': 'Chat deleted'})
    messages = Message.query.filter_by(chat_id=chat_id).order_by(Message.timestamp.asc()).all()
    return jsonify([{'sender': m.sender, 'content': m.content, 'type': m.message_type} for m in messages])

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({'message': 'File uploaded successfully', 'filepath': filepath}), 200
    return jsonify({'error': 'File upload failed'}), 500

@app.route('/chat', methods=['POST'])
@login_required
def handle_chat_message():
    data = request.get_json()
    prompt = data.get('prompt', '')
    chat_id = data.get('chat_id')
    model_mode = data.get('model', 'chat')
    file_content_b64 = data.get('file_content') # Now passed from frontend

    # --- Handle File Content ---
    full_prompt = prompt
    if file_content_b64:
        try:
            # The frontend sends a Data URL, so we need to parse it
            header, encoded = file_content_b64.split(",", 1)
            file_data = base64.b64decode(encoded)
            # We assume it's a text file for now for simplicity
            file_text = file_data.decode('utf-8')
            full_prompt = f"Using the following document as context:\n\n---\n{file_text}\n---\n\nNow, please answer the following question: {prompt}"
        except Exception as e:
            print(f"Error decoding file content: {e}")
            # Don't halt execution, just use the original prompt
            full_prompt = prompt

    chat = Chat.query.filter_by(id=chat_id, user_id=current_user.id).first()
    if not chat:
        title = (full_prompt[:50] + '...') if full_prompt else "New Chat"
        chat = Chat(id=chat_id, user_id=current_user.id, title=title)
        db.session.add(chat)

    if full_prompt:
        db.session.add(Message(chat_id=chat_id, sender='user', content=prompt)) # Save original prompt

    try:
        if model_mode == 'image':
            payload = {"instances": [{"prompt": full_prompt}], "parameters": {"sampleCount": 1}}
            api_response = call_gemini_api(IMAGE_GEN_MODEL, payload)
            image_b64 = api_response.get("predictions", [{}])[0].get("bytesBase64Encoded")
            if not image_b64:
                raise ValueError("AI model did not return an image.")
            ai_content = f"data:image/png;base64,{image_b64}"
            msg_type = 'image'
        else:
            tone = data.get('tone', 'default')
            emotion = data.get('emotion')

            if tone and tone != 'default' and tone != 'custom':
                 full_prompt = f"Please respond in a {tone} tone. {full_prompt}"
            elif tone == 'custom' and data.get('customTone'):
                 custom_instruction = data.get('customTone')
                 full_prompt = f"{custom_instruction}\n\n{full_prompt}"

            if model_mode == 'emotion' and emotion:
                full_prompt = f"(System: User's expression is '{emotion}'. Comment briefly, then address their prompt.)\n{full_prompt}"

            payload = {"contents": [{"parts": [{"text": full_prompt}]}]}
            api_response = call_gemini_api(CHAT_MODEL, payload)
            ai_content = api_response.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "Sorry, I couldn't process that.")
            msg_type = 'text'

        db.session.add(Message(chat_id=chat_id, sender='ai', content=ai_content, message_type=msg_type))
        db.session.commit()
        return jsonify({'solution': ai_content, 'type': msg_type})

    except (ValueError, ConnectionError) as e:
        return jsonify({'error': str(e)}), 500

@app.route('/summarize', methods=['POST'])
@login_required
def summarize_conversation():
    text = request.get_json().get('text', '')
    if not text:
        return jsonify({'error': 'No text to summarize'}), 400
    try:
        summary_prompt = f"Please provide a concise summary of the following conversation:\n\n{text}"
        payload = {"contents": [{"parts": [{"text": summary_prompt}]}]}
        api_response = call_gemini_api(CHAT_MODEL, payload)
        summary = api_response.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "Could not generate summary.")
        return jsonify({'summary': summary})
    except (ValueError, ConnectionError) as e:
        return jsonify({'error': str(e)}), 500

# --- Main Execution ---
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5000)
