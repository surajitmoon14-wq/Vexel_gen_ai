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
STABILITY_ENGINE_ID = "stable-diffusion-xl-1024-v0-9" # Recommended for high quality. You can also try "stable-diffusion-v1-6"

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


# --- Image Generation Route (NOW USES STABILITY AI) ---
@app.route('/generate', methods=['POST'])
def generate_image():
    if 'username' not in session:
        return jsonify({'error': 'Authentication required.'}), 401

    # Check if Stability AI API key is configured
    if not STABILITY_API_KEY:
        return jsonify({'error': 'Server is not configured for image generation (Stability AI API key missing).'}), 500

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
        print(f"Generating image with Stability AI for prompt: '{prompt}'")

        # Make the request to Stability AI API
        response = requests.post(
            f"{STABILITY_API_HOST}/v1/generation/{STABILITY_ENGINE_ID}/text-to-image",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json", # Request JSON response
                "Authorization": f"Bearer {STABILITY_API_KEY}"
            },
            json={
                "text_prompts": [
                    {
                        "text": prompt
                    }
                ],
                "cfg_scale": 7, # Classifier-free guidance scale
                "clip_guidance_preset": "FAST_BLUE", # Recommended preset
                "height": 1024, # Recommended for stable-diffusion-xl-1024-v0-9
                "width": 1024,  # Recommended for stable-diffusion-xl-1024-v0-9
                "samples": 1, # Number of images to generate
                "steps": 30, # Number of diffusion steps
            }
        )

        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        response_data = response.json()

        if not response_data or not response_data.get('artifacts'):
            raise Exception("Stability AI did not return any image artifacts.")

        # Stability AI returns base64 encoded images in 'artifacts'
        image_base64 = response_data['artifacts'][0]['base64']

        # Save the generated image from base64 to a file
        image_filename = f"{uuid.uuid4()}.png"
        image_path = os.path.join(UPLOAD_FOLDER, image_filename)

        with open(image_path, "wb") as f:
            f.write(base64.b64decode(image_base64))

        # Create the public URL for the locally saved image
        final_image_url = f"/{image_path}" # Relative URL for the browser

        # Save the interaction to chat history
        user_message = {"sender": "user", "content": prompt, "type": "text"}
        ai_message = {"sender": "ai", "content": f"Here is the image you requested for: '{prompt}'", "type": "image", "url": final_image_url}

        save_message_to_history(session['username'], chat_id, user_message)
        save_message_to_history(session['username'], chat_id, ai_message)

        # Return the URL of the generated image
        return jsonify({'solution': ai_message['content'], 'image_url': final_image_url})

    except requests.exceptions.RequestException as e:
        print(f"Stability AI Request Error in /generate: {e}")
        error_detail = "Unknown request error."
        if e.response is not None:
            try:
                error_json = e.response.json()
                error_detail = error_json.get('message', error_json.get('errors', str(e)))
            except json.JSONDecodeError:
                error_detail = e.response.text # Fallback to raw text if not JSON
        error_message = f"Sorry, the Stability AI model reported an API error: {error_detail}"
        return jsonify({'error': error_message}), e.response.status_code if e.response is not None else 500
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
        'formal': "You are a professional, formal assistant.",
        'fun': "You are a witty, fun-loving assistant.",
        'default': "You are Vexel AI, a helpful assistant."
    }

    # Determine system instruction based on emotion or general tone
    if user_prompt.startswith("(System: The user's expression just changed to"):
        system_instruction_text = "You are Vexel AI. You can see the user via their webcam. Briefly and naturally comment on the emotion they are showing, which is mentioned in the user's prompt. For example, if the prompt says they are happy, you could say 'I see you're smiling!' or 'You look happy right now!'"
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
