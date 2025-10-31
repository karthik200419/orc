import os
import uuid
from flask import Flask, render_template, request, send_from_directory
import easyocr
from transformers import pipeline

# --- Disable TensorFlow backend and logs ---
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# --- Initialize Flask app ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# --- Ensure upload folder exists ---
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Initialize EasyOCR reader ---
reader = easyocr.Reader(['en'])  # you can add more languages like ['en', 'hi'] for Hindi+English

# --- Load summarization model (uses PyTorch backend) ---
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")

# --- Home page route ---
@app.route('/')
def index():
    return render_template('index.html')

# --- Upload & process route ---
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file uploaded"
    file = request.files['file']
    if file.filename == '':
        return "No file selected"

    # Generate unique filename
    unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

    # Save uploaded image
    file.save(filepath)

    # Perform OCR using EasyOCR
    result = reader.readtext(filepath, detail=0)
    extracted_text = " ".join(result)

    # Summarize extracted text
    if len(extracted_text.strip()) == 0:
        summary = "No readable text found in image."
    else:
        max_input = 500
        chunks = [extracted_text[i:i + max_input] for i in range(0, len(extracted_text), max_input)]
        summary_list = []
        for chunk in chunks:
            result = summarizer(chunk, max_length=100, min_length=30, do_sample=False)
            summary_list.append(result[0]['summary_text'])
        summary = " ".join(summary_list)

    return render_template(
        'index.html',
        extracted_text=extracted_text,
        summary=summary,
        filename=unique_filename
    )

# --- Serve uploaded files ---
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# --- Run app ---
if __name__ == '__main__':
    app.run(debug=True)
