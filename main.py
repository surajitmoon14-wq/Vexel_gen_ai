# --- NEW IMPORTS FOR LOGIN SYSTEM, HISTORY & IMAGEN (now DALL-E) ---
from flask import redirect, url_for, session, flash
from flask_bcrypt import Bcrypt
from replit import db
import json
import time
from datetime import datetime
import uuid
import random
# Removed vertexai specific imports as we are switching to DALL-E
# import vertexai
# from vertexai.generative_models import GenerativeModel

# New import for OpenAI
import openai
import requests # For downloading images from DALL-E's URL response

import os
import base64
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# --- App Initialization & Configuration ---
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# --- NEW CONFIGURATION FOR LOGIN SYSTEM ---
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'a_super_secret_key_for_dev')
bcrypt = Bcrypt(app)
# ----------------------------------------

# --- NEW: CREATE UPLOAD FOLDER IF IT DOESN'T EXIST ---
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
# ----------------------------------------------------

# --- Configure API Keys from Environment Secrets ---
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
# GCP_PROJECT_ID is no longer needed for DALL-E, but keeping it if other Vertex AI features are used
GCP_PROJECT_ID = os.environ.get('GCP_PROJECT_ID') 
# GCP_SERVICE_ACCOUNT_KEY is no longer needed for DALL-E
# GCP_SERVICE_ACCOUNT_KEY = os.environ.get('GCP_SERVICE_ACCOUNT_KEY')

# New API Key for DALL-E
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# --- Initialize Gemini (for chat/summarize) ---
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("Gemini API configured successfully.")
else:
    print("WARNING: GEMINI_API_KEY not set. Text-based AI features will not work.")

# --- Initialize DALL-E (replacing Vertex AI Imagen) ---
dalle_client = None
if OPENAI_API_KEY:
    try:
        dalle_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        print("OpenAI DALL-E client configured successfully.")
    except Exception as e:
        print(f"WARNING: Failed to initialize OpenAI DALL-E client. Image generation will not work. Error: {e}")
else:
    print("WARNING: OPENAI_API_KEY not set. Image generation will not work.")


# This dictionary is no longer used for emotion responses but is kept for other potential uses.
EMOTION_QUOTES = {
    "happy": [
        "Keep shining, the world needs your light!",
        "Happiness looks gorgeous on you.",
        "Ride the wave of happiness you're on!"
    ],
    "sad": [
        "It's okay to feel sad. This feeling is just a visitor.",
        "After the rain, there's always a rainbow. Hang in there.",
        "Be gentle with yourself. You're doing the best you can."
    ],
    "angry": [
        "Take a deep breath. This feeling will pass.",
        "Channel that fire into something productive.",
        "Peace is the goal. Let go of what disturbs it."
    ],
    "surprised": [
        "Life is full of wonderful surprises, isn't it?",
        "Expect the unexpected! Keeps things interesting.",
        "A surprise is a little gift from the universe."
    ]
}

# --- Helper Functions for Database Keys ---
def get_user_key(username):
    return f"user_{username}"

def get_history_key(username):
    return f"history_{username}"

def get_profile_key(username):
    return f"profile_{username}"

# --- Main Frontend Serving Route ---
@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))

    # Fetch user profile to pass to template
    profile_key = get_profile_key(session['username'])
    user_profile = json.loads(db.get(profile_key, '{}'))
    display_name = user_profile.get('displayName', session['username'])
    avatar_url = user_profile.get('avatarUrl', '/static/bot.png')
    font_size = user_profile.get('fontSize', '16px') # Default font size
    font_style = user_profile.get('fontStyle', 'Inter') # Default font style

    return render_template(
        'index.html', 
        username=display_name, 
        avatar_url=avatar_url,
        font_size=font_size,
        font_style=font_style
    )


# --- Image Generation Route (NOW USES DALL-E) ---
@app.route('/generate', methods=['POST'])
def generate_image():
    if 'username' not in session:
        return jsonify({'error': 'Authentication required.'}), 401

    # Check if DALL-E client is initialized
    if not dalle_client:
        return jsonify({'error': 'Server is not configured for image generation (DALL-E client not initialized).'}), 500

    # Handle both form data and JSON data (frontend should send JSON)
    if request.content_type and 'application/json' in request.content_type:
        data = request.get_json()
        prompt = data.get('prompt')
        chat_id = data.get('chat_id')
    else:
        # Fallback for form data, though JSON is expected from frontend
        prompt = request.form.get('prompt')
        chat_id = request.form.get('chat_id')

    if not prompt:
        return jsonify({'error': 'Prompt is required.'}), 400

    try:
        print(f"Generating image with DALL-E for prompt: '{prompt}'")

        # Generate the image using DALL-E 3
        response = dalle_client.images.generate(
            model="dall-e-3", # Specify DALL-E 3 model
            prompt=prompt,
            size="1024x1024", # You can choose "1024x1024", "1792x1024", or "1024x1792"
            quality="standard", # or "hd"
            n=1, # Number of images to generate
        )

        # DALL-E returns a URL to the generated image
        image_url_from_dalle = response.data[0].url

        # Download the image from the URL and save it locally
        image_filename = f"{uuid.uuid4()}.png"
        image_path = os.path.join(UPLOAD_FOLDER, image_filename)

        # Use requests to download the image
        img_data = requests.get(image_url_from_dalle).content
        with open(image_path, 'wb') as handler:
            handler.write(img_data)

        # Create the public URL for the locally saved image
        final_image_url = f"/{image_path}" # Relative URL for the browser

        # Save the interaction to chat history
        user_message = {"sender": "user", "content": prompt, "type": "text"}
        ai_message = {"sender": "ai", "content": f"Here is the image you requested for: '{prompt}'", "type": "image", "url": final_image_url}

        save_message_to_history(session['username'], chat_id, user_message)
        save_message_to_history(session['username'], chat_id, ai_message)

        # Return the URL of the generated image
        return jsonify({'solution': ai_message['content'], 'image_url': final_image_url})

    except openai.APIError as e:
        print(f"OpenAI API Error in /generate: {e}")
        error_message = f"Sorry, the DALL-E model reported an API error: {str(e)}"
        return jsonify({'error': error_message}), 500
    except Exception as e:
        print(f"General Error in /generate: {e}")
        error_message = f"Sorry, I couldn't create the image. An unexpected error occurred: {str(e)}"
        return jsonify({'error': error_message}), 500

# --- Text Summarization Route ---
@app.route('/summarize', methods=['POST'])
def summarize_text():
    if 'username' not in session:
        return jsonify({'error': 'Authentication required.'}), 401
    if not GEMINI_API_KEY:
        return jsonify({'error': 'Server is not configured for summarization.'}), 500
    text_to_summarize = request.get_json().get('text')
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

# --- Main Chat Route (With Emotion Detector Logic) ---
@app.route('/chat', methods=['POST'])
def handle_chat():
    if 'username' not in session:
        return jsonify({'error': 'Authentication required.'}), 401
    if not GEMINI_API_KEY:
        return jsonify({'error': 'Server is not configured for chat.'}), 500

    data = request.get_json()
    user_prompt = data.get('prompt', '')
    emotion = data.get('emotion', 'neutral')

    # The previous logic here was blocking emotion prompts from reaching the AI.
    # It has been replaced with a dynamic system instruction below to handle it correctly.

    # Standard chat logic
    tone = data.get('tone', 'default')
    file_content = data.get('file_content', None)
    chat_id = data.get('chat_id')

    if not user_prompt and not file_content:
        return jsonify({'error': 'A prompt or file is required.'}), 400

    system_prompts = {
        'formal': "You are a professional, formal assistant.",
        'fun': "You are a witty, fun-loving assistant.",
        'default': "You are Vexel AI, a helpful assistant."
    }

    # Set the system instruction based on the context (emotion detection or regular chat)
    if user_prompt.startswith("(System: The user's expression just changed to"):
        system_instruction = "You are Vexel AI. You can see the user via their webcam. Briefly and naturally comment on the emotion they are showing, which is mentioned in the user's prompt. For example, if the prompt says they are happy, you could say 'I see you're smiling!' or 'You look happy right now!'"
    else:
        system_instruction = system_prompts.get(tone, system_prompts['default'])

    model_contents = []
    if file_content:
        # Handle file content (unchanged)
        try:
            header, encoded_data = file_content.split(",", 1)
            mime_type = header.split(":")[1].split(";")[0]
            if mime_type.startswith("image/"):
                model_contents.append(user_prompt)
                model_contents.append({"mime_type": mime_type, "data": encoded_data})
            else:
                text_content = base64.b64decode(encoded_data).decode('utf-8')
                model_contents.append(f"File content:\n{text_content}\n\nUser prompt: {user_prompt}")
        except Exception:
            model_contents.append(f"Text content: {file_content}\n\nUser prompt: {user_prompt}")
    else:
        model_contents.append(user_prompt)

    try:
        model = genai.GenerativeModel('gemini-1.5-pro-latest', system_instruction=system_instruction)
        response = model.generate_content(model_contents)
        solution_text = response.text

        # Do not save the automatic emotion-triggered messages to history to keep the log clean.
        if not user_prompt.startswith("(System:"):
            user_message = {"sender": "user", "content": user_prompt, "type": "text"}
            if file_content and 'mime_type' in locals() and mime_type.startswith("image/"):
                user_message['attachment'] = file_content 

            ai_message = {"sender": "ai", "content": solution_text, "type": "text"}

            save_message_to_history(session['username'], chat_id, user_message)
            save_message_to_history(session['username'], chat_id, ai_message)

        return jsonify({'solution': solution_text})
    except Exception as e:
        print(f"Error during chat: {e}")
        return jsonify({'error': "The AI model failed to respond."}), 500

# --- Chat History Management Functions ---
def save_message_to_history(username, chat_id, message):
    history_key = get_history_key(username)
    user_history = json.loads(db.get(history_key, '{}'))
    if chat_id not in user_history:
        # Create a title from the first message content
        title_content = message.get('content', 'New Chat')
        if isinstance(title_content, str):
                title = (title_content[:30] + "...") if len(title_content) > 30 else title_content
        else:
                title = "New Chat"
        user_history[chat_id] = {"title": title, "created_at": time.time(), "messages": []}

    # Ensure messages list exists
    if 'messages' not in user_history[chat_id]:
        user_history[chat_id]['messages'] = []

    user_history[chat_id]['messages'].append(message)
    db[history_key] = json.dumps(user_history)

@app.route('/history', methods=['GET'])
def get_history():
    if 'username' not in session: return jsonify({'error': 'Authentication required.'}), 401
    history_key = get_history_key(session['username'])
    user_history = json.loads(db.get(history_key, '{}'))
    # Grouping and sorting logic (unchanged)
    grouped_chats = {}
    today = datetime.now().date()
    sorted_chats = sorted(user_history.items(), key=lambda item: item[1].get('created_at', 0), reverse=True)
    for chat_id, data in sorted_chats:
        chat_date = datetime.fromtimestamp(data.get('created_at', 0)).date()
        delta = today - chat_date
        if delta.days == 0: group_name = "Today"
        elif delta.days == 1: group_name = "Yesterday"
        else: group_name = chat_date.strftime("%B %d, %Y")
        if group_name not in grouped_chats:
            grouped_chats[group_name] = []
        grouped_chats[group_name].append({"id": chat_id, "title": data["title"]})
    return jsonify(grouped_chats)

@app.route('/chat/<chat_id>', methods=['GET'])
def get_chat_messages(chat_id):
    if 'username' not in session: return jsonify({'error': 'Authentication required.'}), 401
    history_key = get_history_key(session['username'])
    user_history = json.loads(db.get(history_key, '{}'))
    chat_data = user_history.get(chat_id)
    return jsonify(chat_data['messages']) if chat_data else (jsonify({"error": "Chat not found."}), 404)

@app.route('/chat/<chat_id>', methods=['DELETE'])
def delete_chat(chat_id):
    if 'username' not in session: return jsonify({'error': 'Authentication required.'}), 401
    history_key = get_history_key(session['username'])
    user_history = json.loads(db.get(history_key, '{}'))
    if chat_id in user_history:
        del user_history[chat_id]
        db[history_key] = json.dumps(user_history)
        return jsonify({"success": True})
    return jsonify({"error": "Chat not found."}), 404

@app.route('/history/clear', methods=['POST'])
def clear_history():
    if 'username' not in session: return jsonify({'error': 'Authentication required.'}), 401
    history_key = get_history_key(session['username'])
    db[history_key] = json.dumps({})
    return jsonify({"success": True})

# --- NEW: Chat History Search Route ---
@app.route('/history/search', methods=['GET'])
def search_history():
    if 'username' not in session:
        return jsonify({'error': 'Authentication required.'}), 401

    query = request.args.get('q', '').lower()
    if not query:
        return jsonify({'error': 'Search query is required.'}), 400

    history_key = get_history_key(session['username'])
    user_history = json.loads(db.get(history_key, '{}'))

    search_results = []
    for chat_id, data in user_history.items():
        # Search in title
        if query in data.get('title', '').lower():
            search_results.append({"id": chat_id, "title": data["title"], "match_in": "title"})
            continue # Move to next chat to avoid duplicates
        # Search in messages
        for message in data.get('messages', []):
            if isinstance(message.get('content'), str) and query in message.get('content', '').lower():
                search_results.append({"id": chat_id, "title": data["title"], "match_in": "message"})
                break # Found a match in this chat, move to the next one

    return jsonify(search_results)

# --- NEW: User Profile Management Routes ---
@app.route('/profile', methods=['GET'])
def get_profile():
    if 'username' not in session:
        return jsonify({'error': 'Authentication required.'}), 401

    profile_key = get_profile_key(session['username'])
    profile_data = json.loads(db.get(profile_key, '{}'))

    # Ensure defaults if profile is empty
    profile_data.setdefault('displayName', session['username'])
    profile_data.setdefault('avatarUrl', '/static/bot.png')
    profile_data.setdefault('fontSize', '16px')
    profile_data.setdefault('fontStyle', 'Inter')


    return jsonify(profile_data)

@app.route('/profile', methods=['POST'])
def update_profile():
    if 'username' not in session:
        return jsonify({'error': 'Authentication required.'}), 401

    data = request.get_json()

    profile_key = get_profile_key(session['username'])
    profile_data = json.loads(db.get(profile_key, '{}'))

    # Update only the fields that are provided in the request
    if 'displayName' in data:
        profile_data['displayName'] = data['displayName']
    if 'avatarUrl' in data:
        profile_data['avatarUrl'] = data['avatarUrl']
    if 'fontSize' in data:
        profile_data['fontSize'] = data['fontSize']
    if 'fontStyle' in data:
        profile_data['fontStyle'] = data['fontStyle']

    db[profile_key] = json.dumps(profile_data)

    return jsonify({"success": True, "message": "Profile updated."})

# --- User Authentication Routes ---
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user_key = get_user_key(username)
        if db.get(user_key):
            flash("Username already exists! Please choose another.", "danger")
            return redirect(url_for('signup'))
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        db[user_key] = hashed_password
        # Create a default profile for the new user
        profile_key = get_profile_key(username)
        default_profile = {
            'displayName': username, 
            'avatarUrl': '/static/bot.png',
            'fontSize': '16px',
            'fontStyle': 'Inter'
        }
        db[profile_key] = json.dumps(default_profile)
        flash("Account created successfully! Please log in.", "success")
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user_key = get_user_key(username)
        hashed_password = db.get(user_key)
        if hashed_password and bcrypt.check_password_hash(hashed_password, password):
            session['username'] = username
            return redirect(url_for('index'))
        else:
            flash("Invalid username or password.", "danger")
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

# --- Main Execution ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
