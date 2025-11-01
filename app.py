from flask import Flask, render_template, request
import pytesseract
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, pipeline
import io
import re
from textblob import TextBlob

app = Flask(__name__)

# ---- Paths ----
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---- Load OCR Models ----
trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# ---- Load Summarizer ----
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# ---- OCR Functions ----
def clean_text(text):
    text = re.sub(r"[^A-Za-z0-9.,;:!?()'\-\n ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def correct_text(text):
    """Correct grammar and spelling using TextBlob"""
    try:
        blob = TextBlob(text)
        corrected = str(blob.correct())
        return corrected
    except Exception:
        return text

def ocr_tesseract(image):
    try:
        text = pytesseract.image_to_string(image, lang="eng")
        return clean_text(text)
    except Exception as e:
        return f"Tesseract Error: {e}"

def ocr_trocr(image):
    pixel_values = trocr_processor(images=image, return_tensors="pt").pixel_values
    with torch.no_grad():
        generated_ids = trocr_model.generate(pixel_values)
    generated_text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return clean_text(generated_text)

def summarize_text(text):
    if not text or len(text.split()) < 15:
        return "Text too short for summarization."

    # ✅ Correct English before summarizing
    corrected_text = correct_text(text)

    summary = summarizer(corrected_text, max_length=80, min_length=25, do_sample=False)
    return summary[0]["summary_text"]

# ---- Routes ----
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return render_template("index.html", text="No image uploaded")

    file = request.files["image"]
    image = Image.open(io.BytesIO(file.read())).convert("RGB")

    # Step 1 - Try Tesseract
    text = ocr_tesseract(image)

    # Step 2 - Fallback to TrOCR if unclear
    if len(text) < 25:
        text = ocr_trocr(image)

    # Step 3 - Summarize (auto-corrected English)
    summary = summarize_text(text)

    # Step 4 - Display both original and corrected text
    corrected_text = correct_text(text)

    return render_template("index.html", text=corrected_text, summary=summary)

if __name__ == "__main__":
    app.run(debug=True)
