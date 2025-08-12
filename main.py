### --- Vexel AI Backend --- ###
# This script runs a Flask web server that powers the Vexel AI application.
# It handles user authentication, chat history, AI interactions, and image generation.

# --- NEW IMPORTS FOR LOGIN SYSTEM & HISTORY ---
from flask import redirect, url_for, session, flash, render_template, request, jsonify
from flask_bcrypt import Bcrypt
from replit import db # Uses Replit's built-in key-value database for storage
import json # Used to store complex data (like chat history) as text in the database
import time # Used for timestamping chats
from datetime import datetime
import uuid # Used to generate unique IDs for filenames to prevent conflicts
# ------------------------------------

import os # Used to interact with the operating system (e.g., check if a folder exists)
import base64 # Used for encoding/decoding data, especially for file uploads
import requests # Used to make HTTP requests to external APIs (like Stability AI)
import google.generativeai as genai # The official Google library for the Gemini API
from flask_cors import CORS # Handles Cross-Origin Resource Sharing to allow the frontend to communicate with the backend
from flask import Flask

# --- App Initialization & Configuration ---
# This creates the main Flask application instance.
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# --- NEW CONFIGURATION FOR LOGIN SYSTEM ---
# Sets a secret key for the application, used to secure user sessions and cookies.
# It's crucial for security features like login.
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY')
# Initializes Bcrypt, which is used for hashing passwords securely.
bcrypt = Bcrypt(app)
# ----------------------------------------

# --- NEW: CREATE UPLOAD FOLDER IF IT DOESN'T EXIST ---
# Defines a folder where user-uploaded images will be stored.
UPLOAD_FOLDER = 'static/uploads'
# Checks if this folder exists when the app starts, and creates it if it doesn't.
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
# ----------------------------------------------------

# --- Configure API Keys from Environment Secrets ---
# Safely retrieves API keys from environment variables (Replit Secrets).
# This prevents hardcoding sensitive keys directly in the code.
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
STABILITY_API_KEY = os.environ.get('STABILITY_API_KEY')

# --- Initialize Gemini for Chat ---
# Configures the Gemini library with your API key if it's found.
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("Gemini API configured successfully for Chat.")
else:
    # Prints a warning if the key is missing, as chat features won't work.
    print("WARNING: GEMINI_API_KEY not set. Chat/Summarize features will not work.")

# --- Check for Stability AI Key ---
if not STABILITY_API_KEY:
    print("WARNING: STABILITY_API_KEY not set. Image Generator will not work.")


# --- Main Frontend Serving Route ---
@app.route('/')
def index():
    """
    This is the main homepage. It checks if a user is logged in by looking for 'username' in the session.
    If they are logged in, it shows them the main chat application (index.html).
    If not, it redirects them to the login page.
    """
    if 'username' in session:
        return render_template('index.html', username=session['username'])
    return redirect(url_for('login'))


# --- Image Generation Route ---
@app.route('/generate', methods=['POST'])
def generate_image():
    """
    Handles requests to generate an image. It calls the Stability AI API.
    It also saves the user's prompt and the resulting image to the chat history.
    """
    if 'username' not in session:
        return jsonify({'error': 'Authentication required.'}), 401

    if not STABILITY_API_KEY:
        return jsonify({'error': 'Server is not configured with a Stability AI API key.'}), 500

    prompt = request.form.get('prompt')
    if not prompt:
        return jsonify({'error': 'Prompt is required.'}), 400

    try:
        # Makes the API call to Stability AI
        api_host = "https://api.stability.ai/v2beta/stable-image/generate/core"
        response = requests.post(api_host,
            headers={
                "Authorization": f"Bearer {STABILITY_API_KEY}",
                "Accept": "application/json"
            },
            files={"none": ''},
            data={"prompt": prompt, "output_format": "png"},
            timeout=45)
        response.raise_for_status() # This will raise an error if the API call failed

        image_b64 = response.json().get("image")
        if not image_b64:
            raise ValueError("No image data received from Stability AI.")

        # The image is returned as a base64 string, which is formatted into a data URL
        # that can be displayed directly in an HTML `<img>` tag.
        image_data_url = f'data:image/png;base64,{image_b64}'

        # Save the prompt and the image URL to the user's chat history
        chat_id = request.form.get('chat_id')
        user_message = {"sender": "user", "content": prompt, "type": "text"}
        ai_message = {"sender": "ai", "content": image_data_url, "type": "image"}
        save_message_to_history(session['username'], chat_id, user_message)
        save_message_to_history(session['username'], chat_id, ai_message)

        return jsonify({'image_url': image_data_url})

    except Exception as e:
        print(f"Error in /generate: {e}")
        return jsonify({'error': f"An unexpected error occurred: {str(e)}"}), 500


# --- Text Summarization Route ---
@app.route('/summarize', methods=['POST'])
def summarize_text():
    """Handles requests to summarize a block of text using the Gemini API."""
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

# --- Main Chat Route ---
@app.route('/chat', methods=['POST'])
def handle_chat():
    """
    This is the core chat function. It handles text prompts and file uploads,
    sends them to the Gemini API, and saves the conversation to history.
    """
    if 'username' not in session:
        return jsonify({'error': 'Authentication required.'}), 401

    if not GEMINI_API_KEY:
        return jsonify({'error': 'Server is not configured for chat.'}), 500

    data = request.get_json()
    user_prompt = data.get('prompt', '')
    tone = data.get('tone', 'default')
    file_content = data.get('file_content', None)
    chat_id = data.get('chat_id')

    if not user_prompt and not file_content:
        return jsonify({'error': 'A prompt or file is required.'}), 400

    # Defines different personalities (system instructions) for the AI model
    system_prompts = {
        'formal': "You are a professional, formal, and highly articulate assistant...",
        'fun': "You are a witty, fun-loving, and creative assistant...",
        'default': "You are Vexel AI. Your tone is straightforward and helpful..."
    }
    system_instruction = system_prompts.get(tone, tone)

    model_contents = [] # This list will hold the parts of the prompt sent to Gemini

    # --- File Processing Logic ---
    if file_content:
        try:
            # The file is sent as a base64 data URL; this splits it into header and data
            header, encoded_data = file_content.split(",", 1)
            mime_type = header.split(":")[1].split(";")[0]

            if mime_type.startswith("image/"):
                # For images, the prompt and image data are sent as separate parts
                model_contents.append(user_prompt)
                model_contents.append({"mime_type": mime_type, "data": encoded_data})
            else:
                # For text files, the content is decoded and added to the prompt
                text_content = base64.b64decode(encoded_data).decode('utf-8')
                prompt_with_context = f"Based on the attached file, answer this:\n\n{text_content}\n\n---\n\n{user_prompt}"
                model_contents.append(prompt_with_context)
        except Exception:
            # Fallback for plain text content
            model_contents.append(f"Based on this text, answer the question: {file_content}\n\n---\n\n{user_prompt}")
    else:
        # If there's no file, just send the user's prompt
        model_contents.append(user_prompt)

    try:
        # --- Call the Gemini API ---
        model = genai.GenerativeModel('gemini-1.5-pro-latest', system_instruction=system_instruction)
        response = model.generate_content(model_contents)
        solution_text = response.text

        # --- Save Conversation to History ---
        user_message = {"sender": "user", "content": user_prompt, "type": "text"}

        # If an image was uploaded, save it to a file and store its path in the history
        if file_content and mime_type.startswith("image/"):
            try:
                img_data = base64.b64decode(encoded_data)
                extension = mime_type.split('/')[-1]
                filename = f"{uuid.uuid4()}.{extension}" # Create a unique filename
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                with open(filepath, "wb") as f:
                    f.write(img_data)
                user_message['attachment'] = f"/{filepath}" # Store the web-accessible path
            except Exception as e:
                print(f"Error saving uploaded image: {e}")

        ai_message = {"sender": "ai", "content": solution_text, "type": "text"}

        save_message_to_history(session['username'], chat_id, user_message)
        save_message_to_history(session['username'], chat_id, ai_message)

        return jsonify({'solution': solution_text})
    except Exception as e:
        print(f"Error during chat: {e}")
        return jsonify({'error': "<p>The AI model failed to respond.</p>"}), 500

# --- Chat History Management Functions ---
def get_user_history_key(username):
    """Creates a consistent database key for a user's history."""
    return f"history_{username}"

def save_message_to_history(username, chat_id, message):
    """Saves a single message to a specific chat in the user's history."""
    history_key = get_user_history_key(username)
    # Loads the user's entire history from the database (it's stored as a JSON string)
    user_history = json.loads(db.get(history_key, '{}'))

    # If this is the first message of a new chat, create the chat entry
    if chat_id not in user_history:
        # Creates a title for the new chat from the first few words of the prompt
        title = (message.get('content', 'New Chat')[:30] + "...")
        user_history[chat_id] = {
            "title": title,
            "created_at": time.time(), # Save the creation timestamp
            "messages": []
        }

    # Add the new message to the correct chat
    user_history[chat_id]['messages'].append(message)
    # Save the updated history back to the database
    db[history_key] = json.dumps(user_history)

@app.route('/history', methods=['GET'])
def get_history():
    """
    Retrieves a user's entire chat history and groups it by date for display in the sidebar.
    """
    if 'username' not in session: return jsonify({'error': 'Authentication required.'}), 401

    history_key = get_user_history_key(session['username'])
    user_history = json.loads(db.get(history_key, '{}'))

    grouped_chats = {}
    today = datetime.now().date()
    # Sort chats by creation date, newest first
    sorted_chats = sorted(user_history.items(), key=lambda item: item[1].get('created_at', 0), reverse=True)

    # Group chats into categories like "Today", "Yesterday", etc.
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
    """Retrieves all messages from a single, specific chat."""
    if 'username' not in session: return jsonify({'error': 'Authentication required.'}), 401
    history_key = get_user_history_key(session['username'])
    user_history = json.loads(db.get(history_key, '{}'))
    chat_data = user_history.get(chat_id)
    if not chat_data: return jsonify({"error": "Chat not found."}), 404
    return jsonify(chat_data['messages'])

@app.route('/chat/<chat_id>', methods=['DELETE'])
def delete_chat(chat_id):
    """Deletes a specific chat from the user's history."""
    if 'username' not in session: return jsonify({'error': 'Authentication required.'}), 401
    history_key = get_user_history_key(session['username'])
    user_history = json.loads(db.get(history_key, '{}'))
    if chat_id in user_history:
        del user_history[chat_id]
        db[history_key] = json.dumps(user_history)
        return jsonify({"success": True})
    return jsonify({"error": "Chat not found."}), 404

@app.route('/history/clear', methods=['POST'])
def clear_history():
    """Deletes all chats for the logged-in user."""
    if 'username' not in session: return jsonify({'error': 'Authentication required.'}), 401
    history_key = get_user_history_key(session['username'])
    db[history_key] = json.dumps({}) # Overwrites history with an empty object
    return jsonify({"success": True})


# --- User Authentication Routes ---
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Handles new user registration."""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Check if the username is already taken in the database
        if username in db.keys():
            flash("Username already exists! Please choose another.", "danger")
            return redirect(url_for('signup'))
        # Hash the password before storing it
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
        # Check if the user exists and if the provided password matches the stored hash
        if username in db.keys() and bcrypt.check_password_hash(db[username], password):
            session['username'] = username # Logs the user in by saving their name in the session
            return redirect(url_for('index'))
        else:
            flash("Invalid username or password.", "danger")
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Logs the user out by clearing their session."""
    session.pop('username', None)
    return redirect(url_for('login'))


# --- Main Execution ---
# This line makes the server run when the script is executed.
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
