import os
import base64
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# --- App Initialization & Configuration ---
app = Flask(__name__, template_folder='templates')
CORS(app)

# --- Configure API Keys from Environment Secrets ---
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

# --- Initialize Gemini ---
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("Gemini API configured successfully.")
else:
    print("CRITICAL: GEMINI_API_KEY not set. Core features will fail.")


# --- Frontend Serving Route ---
@app.route('/')
def index():
    """Serves the main HTML file."""
    return render_template('index.html')

# --- Image Generation Route (NOW POWERED BY GEMINI) ---
@app.route('/generate', methods=['POST'])
def generate_image():
    """Handles image generation requests using the Gemini API."""
    if not GEMINI_API_KEY:
        return jsonify({'error': 'Server is not configured with a Gemini API key.'}), 500

    prompt = request.form.get('prompt')
    if not prompt:
        return jsonify({'error': 'Prompt is required.'}), 400

    try:
        # Use a Gemini model capable of image generation
        image_model = genai.GenerativeModel('gemini-1.5-flash')

        # IMPROVED: A more direct and explicit prompt for image generation
        generation_prompt = f"Create a high-quality, artistic, digital painting of: {prompt}. The image should be vibrant and of masterpiece quality."

        response = image_model.generate_content(generation_prompt)

        if not response.parts or not hasattr(response.parts[0], 'inline_data'):
            if response.prompt_feedback.block_reason:
                reason = response.prompt_feedback.block_reason.name
                error_message = f"Image generation was blocked for safety reasons: {reason}."
                raise ValueError(error_message)
            else:
                raise ValueError("The AI model did not return an image. Please try a more descriptive prompt.")

        image_part = response.parts[0]
        img_bytes = image_part.inline_data.data
        mime_type = image_part.inline_data.mime_type

        img_str = base64.b64encode(img_bytes).decode('utf-8')
        image_data_url = f'data:image/png;base64,{img_str}'

        return jsonify({'image_url': image_data_url})

    except Exception as e:
        print(f"Error in /generate: {e}")
        return jsonify({'error': f"An unexpected error occurred: {str(e)}"}), 500


# --- Text Summarization Route ---
@app.route('/summarize', methods=['POST'])
def summarize_text():
    """Handles text summarization requests."""
    if not GEMINI_API_KEY:
        return jsonify({'error': 'Server is not configured for summarization.'}), 500

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

# --- UPGRADED: Chat Assistant Route ---
@app.route('/chat', methods=['POST'])
def handle_chat():
    """Handles text-based chat with different tones and file context."""
    if not GEMINI_API_KEY:
        return jsonify({'error': 'Server is not configured for chat.'}), 500

    data = request.get_json()
    user_prompt = data.get('prompt')
    tone = data.get('tone', 'default')
    file_content = data.get('file_content', None)

    if not user_prompt:
        return jsonify({'error': 'A prompt is required.'}), 400

    system_prompts = {
        'formal': "You are a professional, formal, and highly articulate assistant. Provide precise, well-structured, and serious responses.",
        'fun': "You are a witty, fun-loving, and creative assistant. Use humor, emojis, and a lighthearted tone in your responses.",
        'default': "You are Vexel AI, a helpful and friendly assistant. Your tone should be conversational and informative, but not overly formal or casual. Provide clear and direct answers."
    }

    system_prompt = system_prompts.get(tone, tone)

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

# --- Main Execution ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
