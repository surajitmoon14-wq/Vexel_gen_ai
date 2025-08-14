# --- NEW IMPORTS FOR LOGIN SYSTEM, HISTORY & IMAGE GENERATION ---
from flask import redirect, url_for, session, flash
from flask_bcrypt import Bcrypt
from replit import db
import json
import time
from datetime import datetime
import uuid
import random

# Removed previous image generation specific imports (Vertex AI, OpenAI DALL-E)
# import vertexai
# from vertexai.generative_models import GenerativeModel
# import openai

import requests # For downloading images from URLs and making API calls to Stability AI

import os
import base64
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Import Replit Object Storage for file persistence
try:
    from replit.object_storage import Client
    object_storage_client = Client()
    print("Object Storage client initialized successfully.")
except ImportError:
    print("WARNING: Replit Object Storage not available. Falling back to local storage.")
    object_storage_client = None

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
# GCP_PROJECT_ID is not directly used for Stability AI or Gemini chat, but kept if needed for other features
GCP_PROJECT_ID = os.environ.get('GCP_PROJECT_ID') 

# New API Key for Stability AI
STABILITY_API_KEY = os.environ.get('STABILITY_API_KEY')
STABILITY_API_HOST = os.environ.get('STABILITY_API_HOST', 'https://api.stability.ai')
STABILITY_ENGINE_ID = "stable-diffusion-xl-1024-v1-0" # Updated to current available model

# OpenAI API Key (for GPT-4o-mini chat fallback)
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


# --- Initialize Gemini (for chat/summarize and prompt generation) ---
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("Gemini API configured successfully.")
else:
    print("WARNING: GEMINI_API_KEY not set. Text-based AI features will not work (will try OpenAI for chat).")

# --- Initialize OpenAI Chat Client (for chat fallback) ---
# The OpenAI client is only needed if you use GPT-4o-mini for chat.
# It's not used for Stability AI image generation directly.
openai_chat_client = None
if OPENAI_API_KEY:
    try:
        import openai # Ensure openai is imported if not already by other parts
        openai_chat_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        print("OpenAI Chat client configured successfully.")
    except Exception as e:
        print(f"WARNING: Failed to initialize OpenAI Chat client. OpenAI chat will not work. Error: {e}")
else:
    print("WARNING: OPENAI_API_KEY not set. OpenAI chat will not work.")


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

def get_email_key(email):
    return f"email_{email}"

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

# --- NEW: Image Prompt Generation Route (USES GEMINI) ---
@app.route('/generate_image_prompt', methods=['POST'])
def generate_image_prompt():
    if 'username' not in session:
        return jsonify({'error': 'Authentication required.'}), 401
    if not GEMINI_API_KEY:
        return jsonify({'error': 'Gemini API key is not configured for prompt generation.'}), 500

    data = request.get_json()
    user_idea = data.get('idea')
    if not user_idea:
        return jsonify({'error': 'An idea for the image prompt is required.'}), 400

    try:
        print(f"Generating detailed image prompt with Gemini for idea: '{user_idea}'")
        model = genai.GenerativeModel('gemini-1.5-pro-latest')

        system_instruction = (
            "You are an expert prompt engineer for AI image generation models like Stable Diffusion. "
            "Your task is to take a simple idea and expand it into a highly detailed, creative, "
            "and vivid prompt (max 150 words) that will produce an amazing image. "
            "Include details about style, lighting, composition, colors, and mood. "
            "Do NOT include any conversational text, just the prompt itself."
        )
        prompt_text = f"Generate a detailed image prompt based on this idea: '{user_idea}'"

        response = model.generate_content([system_instruction, prompt_text])
        generated_prompt = response.text.strip()

        return jsonify({'detailed_prompt': generated_prompt})

    except Exception as e:
        print(f"Error during Gemini image prompt generation: {e}")
        return jsonify({'error': f"Failed to generate detailed prompt: {str(e)}"}), 500


# --- Image Generation Route (NOW USES GEMINI API) ---
@app.route('/generate', methods=['POST'])
def generate_image():
    if 'username' not in session:
        return jsonify({'error': 'Authentication required.'}), 401

    # Check if Gemini API key is configured
    if not GEMINI_API_KEY:
        return jsonify({'error': 'Server is not configured for image generation (Gemini API key missing).'}), 500

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
        print(f"Generating image description with Gemini API for prompt: '{prompt}'")

        # Use Gemini to create a detailed image description
        model = genai.GenerativeModel('gemini-1.5-pro-latest')

        # Create a detailed prompt for image description
        enhanced_prompt = (
            f"Create a highly detailed, vivid description of an image based on this prompt: '{prompt}'. "
            "Describe it as if you're looking at an actual photograph or artwork. Include details about "
            "colors, lighting, composition, textures, mood, and atmosphere. Make it so descriptive that "
            "someone could visualize the image perfectly. Write it as a single paragraph, maximum 200 words."
        )

        response = model.generate_content(enhanced_prompt)
        image_description = response.text.strip()

        # Create a text file with the description instead of an actual image
        description_filename = f"image_description_{uuid.uuid4()}.txt"
        description_path = os.path.join(UPLOAD_FOLDER, description_filename)

        with open(description_path, "w", encoding='utf-8') as f:
            f.write(f"Image Description for: '{prompt}'\n\n{image_description}")

        # Since we're not generating actual images, we'll provide the description as text response
        final_description = f"Here's a detailed description of the image for '{prompt}':\n\n{image_description}"

        # Save the interaction to chat history
        user_message = {"sender": "user", "content": prompt, "type": "text"}
        ai_message = {"sender": "ai", "content": final_description, "type": "text"}

        save_message_to_history(session['username'], chat_id, user_message)
        save_message_to_history(session['username'], chat_id, ai_message)

        # Return the description instead of image URL
        return jsonify({'solution': ai_message['content']})

    except Exception as e:
        print(f"Gemini Image Description Error in /generate: {e}")
        error_message = f"Sorry, I couldn't create the image description. An unexpected error occurred: {str(e)}"
        return jsonify({'error': error_message}), 500

# --- Text Summarization Route ---
@app.route('/summarize', methods=['POST'])
def summarize_text():
    if 'username' not in session:
        return jsonify({'error': 'Authentication required.'}), 401
    if not GEMINI_API_KEY:
        return jsonify({'error': 'Server is not configured for summarization (Gemini API key missing).'}), 500 # Added more specific error
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

# --- Main Chat Route (With Emotion Detector Logic - Now supports Gemini or OpenAI fallback) ---
@app.route('/chat', methods=['POST'])
def handle_chat():
    if 'username' not in session:
        return jsonify({'error': 'Authentication required.'}), 401

    data = request.get_json()
    user_prompt = data.get('prompt', '')
    emotion = data.get('emotion', 'neutral')
    tone = data.get('tone', 'default')
    file_content = data.get('file_content', None) # Base64 encoded file content
    chat_id = data.get('chat_id')

    if not user_prompt and not file_content:
        return jsonify({'error': 'A prompt or file is required.'}), 400

    system_prompts = {
        'formal': "You are Vexel AI, a professional, formal assistant created by Vexel Studio Lab.",
        'fun': "You are Vexel AI, a witty, fun-loving assistant created by Vexel Studio Lab.",
        'default': "You are Vexel AI, a helpful assistant created by Vexel Studio Lab. When asked about your creator or who made you, always say you were created by Vexel Studio Lab."
    }

    # Determine system instruction based on emotion or general tone
    if user_prompt.startswith("(System: The user's expression just changed to"):
        system_instruction_text = "You are Vexel AI, created by Vexel Studio Lab. You can see the user via their webcam. Briefly and naturally comment on the emotion they are showing, which is mentioned in the user's prompt. For example, if the prompt says they are happy, you could say 'I see you're smiling!' or 'You look happy right now!'"
    else:
        system_instruction_text = system_prompts.get(tone, system_prompts['default'])

    solution_text = ""
    try:
        if GEMINI_API_KEY:
            # --- Use Gemini for chat if GEMINI_API_KEY is available ---
            print("Using Gemini for chat.")
            model_contents = []
            if file_content:
                try:
                    header, encoded_data = file_content.split(",", 1)
                    mime_type = header.split(":")[1].split(";")[0]
                    if mime_type.startswith("image/"):
                        model_contents.append({"mime_type": mime_type, "data": encoded_data})
                        model_contents.append(user_prompt) # Text part after image
                    else:
                        text_content = base64.b64decode(encoded_data).decode('utf-8')
                        model_contents.append(f"File content:\n{text_content}\n\nUser prompt: {user_prompt}")
                except Exception:
                    model_contents.append(f"Text content: {file_content}\n\nUser prompt: {user_prompt}")
            else:
                model_contents.append(user_prompt)

            model = genai.GenerativeModel('gemini-1.5-pro-latest', system_instruction=system_instruction_text)
            response = model.generate_content(model_contents)
            solution_text = response.text

        elif OPENAI_API_KEY and openai_chat_client: # Use openai_chat_client here
            # --- Fallback to OpenAI GPT-4o-mini for chat if Gemini is not configured ---
            print("Using OpenAI GPT-4o-mini for chat.")
            messages = [{"role": "system", "content": system_instruction_text}]

            # Handle file content for OpenAI GPT-4o-mini:
            # GPT-4o-mini is primarily text-based. If an image is attached,
            # we'll add a note about it, but the model won't process the image itself.
            # For text files, content is appended.
            if file_content:
                try:
                    header, encoded_data = file_content.split(",", 1)
                    mime_type = header.split(":")[1].split(";")[0]
                    if mime_type.startswith("image/"):
                        # GPT-4o-mini does not support image input directly in chat completions.
                        # We'll just add a note about the image for context.
                        messages.append({"role": "user", "content": f"{user_prompt}\n\n(Note: An image was attached but is not processed by this text-only model.)"})
                    else:
                        text_content = base64.b64decode(encoded_data).decode('utf-8')
                        messages.append({"role": "user", "content": f"File content:\n{text_content}\n\nUser prompt: {user_prompt}"})
                except Exception:
                    messages.append({"role": "user", "content": f"Text content: {file_content}\n\nUser prompt: {user_prompt}"})
            else:
                messages.append({"role": "user", "content": user_prompt})

            openai_response = openai_chat_client.chat.completions.create( # Use openai_chat_client here
                model="gpt-4o-mini", # Using the text model specified by the user
                messages=messages
            )
            solution_text = openai_response.choices[0].message.content

        else:
            return jsonify({'error': 'Server is not configured for chat (neither Gemini nor OpenAI API key found).'}), 500

        # Do not save the automatic emotion-triggered messages to history to keep the log clean.
        if not user_prompt.startswith("(System:"):
            user_message = {"sender": "user", "content": user_prompt, "type": "text"}
            if file_content: # Save attachment info for user message if present
                user_message['attachment'] = file_content 

            ai_message = {"sender": "ai", "content": solution_text, "type": "text"}

            save_message_to_history(session['username'], chat_id, user_message)
            save_message_to_history(session['username'], chat_id, ai_message)

        return jsonify({'solution': solution_text})
    except openai.APIError as e:
        print(f"OpenAI Chat API Error: {e}")
        # Improved error message extraction for OpenAI APIError
        error_message = f"Sorry, the OpenAI chat model reported an API error: {e.response.json().get('error', {}).get('message', str(e))}"
        return jsonify({'error': error_message}), 500
    except Exception as e:
        print(f"Error during chat: {e}")
        error_message = f"The AI model failed to respond: {str(e)}"
        return jsonify({'error': error_message}), 500

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

# --- NEW: Avatar Upload Route ---
@app.route('/upload_avatar', methods=['POST'])
def upload_avatar():
    if 'username' not in session:
        return jsonify({'error': 'Authentication required.'}), 401

    if 'avatar' not in request.files:
        return jsonify({'error': 'No avatar file provided.'}), 400

    file = request.files['avatar']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    # Validate file type
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
    file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
    if file_ext not in allowed_extensions:
        return jsonify({'error': 'Invalid file type. Please upload an image file.'}), 400

    try:
        # Generate unique filename
        unique_filename = f"avatar_{session['username']}_{uuid.uuid4().hex}.{file_ext}"

        # Read file data
        file_data = file.read()

        # Try to save to Object Storage first, fallback to local storage
        avatar_url = None
        if object_storage_client:
            try:
                object_storage_client.upload_from_bytes(f"avatars/{unique_filename}", file_data)
                avatar_url = f"/avatar/{unique_filename}"
                print(f"Avatar uploaded to Object Storage: {unique_filename}")
            except Exception as e:
                print(f"Object Storage upload failed, falling back to local: {e}")
                object_storage_client = None

        if not avatar_url:
            # Fallback to local storage
            local_path = os.path.join(UPLOAD_FOLDER, unique_filename)
            with open(local_path, 'wb') as f:
                f.write(file_data)
            avatar_url = f"/static/uploads/{unique_filename}"
            print(f"Avatar saved locally: {unique_filename}")

        return jsonify({'avatar_url': avatar_url})

    except Exception as e:
        print(f"Error uploading avatar: {e}")
        return jsonify({'error': 'Failed to upload avatar.'}), 500

# --- Route to serve avatars from Object Storage ---
@app.route('/avatar/<filename>')
def serve_avatar(filename):
    if not object_storage_client:
        return jsonify({'error': 'Object Storage not available.'}), 404

    try:
        # Download from Object Storage
        temp_file = f"/tmp/{filename}"
        object_storage_client.download_to_filename(f"avatars/{filename}", temp_file)

        # Serve the file
        from flask import send_file
        return send_file(temp_file, as_attachment=False)

    except Exception as e:
        print(f"Error serving avatar from Object Storage: {e}")
        return jsonify({'error': 'Avatar not found.'}), 404

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
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']

        user_key = get_user_key(username)
        email_key = get_email_key(email)

        # Check if username or email already exists
        if db.get(user_key):
            flash("Username already exists! Please choose another.", "danger")
            return redirect(url_for('signup'))
        if db.get(email_key):
            flash("Email already registered! Please use a different email.", "danger")
            return redirect(url_for('signup'))

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

        # Store user data with both username and email mapping
        user_data = {
            'password': hashed_password,
            'email': email,
            'username': username
        }

        # Store user data and email mapping
        try:
            db[user_key] = json.dumps(user_data)
            db[email_key] = username  # Map email to username for login lookup

            # Verify the data was stored
            stored_user_data = db.get(user_key)
            stored_email_mapping = db.get(email_key)

            print(f"Created user: {username} with email: {email}")
            print(f"User key: {user_key}, Email key: {email_key}")
            print(f"Stored user data: {stored_user_data is not None}")
            print(f"Stored email mapping: {stored_email_mapping}")

            if not stored_user_data or not stored_email_mapping:
                flash("Failed to create account. Please try again.", "danger")
                return redirect(url_for('signup'))
        except Exception as e:
            print(f"Error storing user data: {e}")
            flash("Failed to create account. Please try again.", "danger")
            return redirect(url_for('signup'))

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
        login_input = request.form['email']  # This field now accepts both email and username
        password = request.form['password']

        username = None

        # Check if input contains @ symbol (likely email)
        if '@' in login_input:
            # Try to find username by email
            email_key = get_email_key(login_input)
            username = db.get(email_key)
            print(f"Login attempt with email: {login_input}")
            print(f"Email key: {email_key}")
            print(f"Found username: {username}")
        else:
            # Input is likely a username
            username = login_input
            print(f"Login attempt with username: {