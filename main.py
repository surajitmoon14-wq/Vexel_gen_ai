# --- NEW IMPORTS FOR LOGIN SYSTEM & HISTORY ---
from flask import redirect, url_for, session, flash
from flask_bcrypt import Bcrypt
from replit import db
import json
import time
from datetime import datetime
import uuid # <-- ADD THIS IMPORT FOR UNIQUE FILENAMES
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

# --- NEW: CREATE UPLOAD FOLDER IF IT DOESN'T EXIST ---
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
# ----------------------------------------------------

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


# --- MODIFIED Frontend Serving Route (Your code, unchanged) ---
@app.route('/')
def index():
    """Checks if user is logged in before showing the main Vexel AI app."""
    if 'username' in session:
        return render_template('index.html', username=session['username'])
    return redirect(url_for('login'))


# --- Image Generation Route (MODIFIED to save history) ---
@app.route('/generate', methods=['POST'])
def generate_image():
    """Handles image generation requests using the Stability AI API."""
    if 'username' not in session:
        return jsonify({'error': 'Authentication required.'}), 401

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

        chat_id = request.form.get('chat_id')
        user_message = {"sender": "user", "content": prompt, "type": "text"}
        ai_message = {"sender": "ai", "content": image_data_url, "type": "image"}
        save_message_to_history(session['username'], chat_id, user_message)
        save_message_to_history(session['username'], chat_id, ai_message)

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

#
# =========================================================================================
# === FINAL CORRECTED CHAT ROUTE ==========================================================
# =========================================================================================
#
@app.route('/chat', methods=['POST'])
def handle_chat():
    """Handles text-based chat with different tones and file context."""
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

    system_prompts = {
        'formal': "You are a professional, formal, and highly articulate assistant. Provide precise, well-structured, and serious responses.",
        'fun': "You are a witty, fun-loving, and creative assistant. Use humor, emojis, and a lighthearted tone in your responses.",
        'default': "You are Vexel AI. Your tone is straightforward and helpful. Provide clear, direct answers."
    }
    system_instruction = system_prompts.get(tone, tone)

    model_contents = []
    prompt_part = user_prompt
    mime_type = ""

    if file_content:
        try:
            header, encoded_data = file_content.split(",", 1)
            mime_type = header.split(":")[1].split(";")[0]

            if mime_type.startswith("image/"):
                image_part = {"mime_type": mime_type, "data": encoded_data}
                model_contents.append(prompt_part)
                model_contents.append(image_part)
            else:
                text_content = base64.b64decode(encoded_data).decode('utf-8')
                prompt_part = f"Based on the attached file, answer this:\n\n{text_content}\n\n---\n\n{prompt_part}"
                model_contents.append(prompt_part)
        except (ValueError, IndexError):
            prompt_part = f"Based on this text, answer the question: {file_content}\n\n---\n\n{prompt_part}"
            model_contents.append(prompt_part)
    else:
        model_contents.append(prompt_part)

    try:
        model = genai.GenerativeModel('gemini-1.5-pro-latest', system_instruction=system_instruction)
        response = model.generate_content(model_contents)
        solution_text = response.text

        # --- SAVE TO HISTORY (MODIFIED LOGIC) ---
        user_message = {"sender": "user", "content": user_prompt, "type": "text"}

        if file_content and mime_type.startswith("image/"):
            # Instead of saving base64, save the image file and store its URL
            try:
                img_data = base64.b64decode(encoded_data)
                extension = mime_type.split('/')[-1]
                filename = f"{uuid.uuid4()}.{extension}"
                filepath = os.path.join(UPLOAD_FOLDER, filename)

                with open(filepath, "wb") as f:
                    f.write(img_data)

                # Store the URL path for the frontend to use
                user_message['attachment'] = f"/{filepath}" 
            except Exception as e:
                print(f"Error saving uploaded image: {e}")

        ai_message = {"sender": "ai", "content": solution_text, "type": "text"}

        save_message_to_history(session['username'], chat_id, user_message)
        save_message_to_history(session['username'], chat_id, ai_message)

        return jsonify({'solution': solution_text})

    except Exception as e:
        print(f"Error during chat: {e}")
        error_html = "<p>Could not process your request. The AI model failed to respond.</p>"
        return jsonify({'error': error_html}), 500
#
# =========================================================================================
# === END OF CORRECTED SECTION ============================================================
# =========================================================================================
#

# --- MODIFIED AND NEW ROUTES FOR CHAT HISTORY ---
def get_user_history_key(username):
    return f"history_{username}"

def save_message_to_history(username, chat_id, message):
    history_key = get_user_history_key(username)
    user_history = json.loads(db.get(history_key, '{}'))

    if chat_id not in user_history:
        title = "Image Query" if 'attachment' in message and not message.get('content') else message.get('content', 'New Chat')[:30] + "..."
        user_history[chat_id] = {
            "title": title,
            "created_at": time.time(),
            "messages": []
        }

    user_history[chat_id]['messages'].append(message)
    db[history_key] = json.dumps(user_history)

@app.route('/history', methods=['GET'])
def get_history():
    if 'username' not in session:
        return jsonify({'error': 'Authentication required.'}), 401

    history_key = get_user_history_key(session['username'])
    user_history = json.loads(db.get(history_key, '{}'))

    grouped_chats = {}
    today = datetime.now().date()

    sorted_chats = sorted(user_history.items(), key=lambda item: item[1].get('created_at', 0), reverse=True)

    for chat_id, data in sorted_chats:
        chat_date = datetime.fromtimestamp(data.get('created_at', 0)).date()
        delta = today - chat_date
        if delta.days == 0:
            group_name = "Today"
        elif delta.days == 1:
            group_name = "Yesterday"
        else:
            group_name = chat_date.strftime("%B %d, %Y")

        if group_name not in grouped_chats:
            grouped_chats[group_name] = []

        grouped_chats[group_name].append({"id": chat_id, "title": data["title"]})

    return jsonify(grouped_chats)

@app.route('/chat/<chat_id>', methods=['GET'])
def get_chat_messages(chat_id):
    if 'username' not in session:
        return jsonify({'error': 'Authentication required.'}), 401
    history_key = get_user_history_key(session['username'])
    user_history = json.loads(db.get(history_key, '{}'))
    chat_data = user_history.get(chat_id)
    if not chat_data:
        return jsonify({"error": "Chat not found."}), 404
    return jsonify(chat_data['messages'])

@app.route('/chat/<chat_id>', methods=['DELETE'])
def delete_chat(chat_id):
    if 'username' not in session:
        return jsonify({'error': 'Authentication required.'}), 401
    history_key = get_user_history_key(session['username'])
    user_history = json.loads(db.get(history_key, '{}'))
    if chat_id in user_history:
        del user_history[chat_id]
        db[history_key] = json.dumps(user_history)
        return jsonify({"success": True})
    return jsonify({"error": "Chat not found."}), 404

@app.route('/history/clear', methods=['POST'])
def clear_history():
    if 'username' not in session:
        return jsonify({'error': 'Authentication required.'}), 401
    history_key = get_user_history_key(session['username'])
    db[history_key] = json.dumps({})
    return jsonify({"success": True})


# --- ROUTES FOR SIGNUP, LOGIN, AND LOGOUT (Your code, unchanged) ---
@app.route('/signup', methods=['GET', 'POST'])
def signup():
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
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in db.keys() and bcrypt.check_password_hash(db[username], password):
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


# --- Main Execution (Your code, unchanged) ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)