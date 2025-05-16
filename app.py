import os
import uuid
import logging
from flask import Flask, request, render_template_string, flash, redirect, url_for
import torchaudio
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import difflib
from werkzeug.utils import secure_filename
from phonemizer import phonemize
from phonemizer.backend import EspeakBackend, FestivalBackend

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for flash messages
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg'}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB max file size
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Load Wav2Vec2 model and processor once
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
model.eval()

# Check available phonemizer backends
def get_phonemizer_backend():
    if EspeakBackend.is_available():
        logger.info("Using espeak backend")
        return EspeakBackend(language='en-us')
    elif FestivalBackend.is_available():
        logger.info("Using festival backend")
        return FestivalBackend(language='en-us')
    else:
        logger.warning("No phonemizer backend available (espeak or festival)")
        return None

phonemizer_backend = get_phonemizer_backend()
if not phonemizer_backend:
    logger.error("Phonemizer backend not found. Install espeak-ng or festival for phoneme support.")

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def compare_phonemes(ref, hyp):
    """Compare reference and hypothesis phonemes, return formatted comparison and accuracy."""
    ref_list = ref.split() if ref else []
    hyp_list = hyp.split() if hyp else []
    matcher = difflib.SequenceMatcher(None, ref_list, hyp_list)
    total, correct = 0, 0
    output = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        for r, h in zip(ref_list[i1:i2], hyp_list[j1:j2]):
            total += 1
            if r == h:
                output.append(f"{r} -> {h} ✔️")
                correct += 1
            else:
                output.append(f"{r} -> {h} ❌")
    accuracy = (correct / total) * 100 if total > 0 else 0
    return "\n".join(output), accuracy

def compare_text(ref, hyp):
    """Fallback comparison for raw text when phonemization is unavailable."""
    ref_words = ref.lower().split()
    hyp_words = hyp.lower().split()
    matcher = difflib.SequenceMatcher(None, ref_words, hyp_words)
    total, correct = 0, 0
    output = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        for r, h in zip(ref_words[i1:i2], hyp_words[j1:j2]):
            total += 1
            if r == h:
                output.append(f"{r} -> {h} ✔️")
                correct += 1
            else:
                output.append(f"{r} -> {h} ❌")
    accuracy = (correct / total) * 100 if total > 0 else 0
    return "\n".join(output), accuracy

@app.route("/", methods=["GET", "POST"])
def analyze():
    if request.method == "POST":
        reference_text = request.form.get("reference", "").strip()
        audio_file = request.files.get("audio")

        # Validate inputs
        if not reference_text:
            flash("Reference text is required.")
            return redirect(url_for('analyze'))
        if not audio_file or not audio_file.filename:
            flash("Audio file is required.")
            return redirect(url_for('analyze'))
        if not allowed_file(audio_file.filename):
            flash("Invalid audio file type. Allowed: wav, mp3, flac, ogg.")
            return redirect(url_for('analyze'))

        # Securely save the file with a unique filename
        filename = secure_filename(audio_file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        try:
            audio_file.save(filepath)

            # Load and process audio
            try:
                waveform, sample_rate = torchaudio.load(filepath)
            except Exception as e:
                flash(f"Error loading audio file: {str(e)}")
                logger.error(f"Audio loading failed: {str(e)}")
                return redirect(url_for('analyze'))

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Resample to 16000 Hz if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)
                sample_rate = 16000

            # Prepare input for model
            try:
                input_values = processor(waveform.squeeze(0), sampling_rate=sample_rate, return_tensors="pt").input_values
            except Exception as e:
                flash(f"Error processing audio: {str(e)}")
                logger.error(f"Audio processing failed: {str(e)}")
                return redirect(url_for('analyze'))

            # Run model inference
            try:
                with torch.no_grad():
                    logits = model(input_values).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = processor.decode(predicted_ids[0])
            except Exception as e:
                flash(f"Error during model inference: {str(e)}")
                logger.error(f"Model inference failed: {str(e)}")
                return redirect(url_for('analyze'))

            # Compare phonemes or text
            if phonemizer_backend:
                try:
                    reference_phonemes = phonemize(reference_text, language='en-us', backend=phonemizer_backend, strip=True)
                    spoken_phonemes = phonemize(transcription, language='en-us', backend=phonemizer_backend, strip=True)
                    comparison_text, accuracy = compare_phonemes(reference_phonemes, spoken_phonemes)
                    comparison_type = "Phoneme Comparison"
                except Exception as e:
                    flash(f"Phonemization failed: {str(e)}. Using text comparison instead. To enable phonemization, install espeak-ng (Linux: sudo apt-get install espeak-ng, macOS: brew install espeak, Windows: download from http://espeak.sourceforge.net/).")
                    logger.error(f"Phonemization failed: {str(e)}")
                    comparison_text, accuracy = compare_text(reference_text, transcription)
                    comparison_type = "Text Comparison (Phonemization Failed)"
            else:
                flash("No phonemization backend available. Install espeak-ng or festival for phoneme support (Linux: sudo apt-get install espeak-ng, macOS: brew install espeak, Windows: download from http://espeak.sourceforge.net/). Using text comparison.")
                comparison_text, accuracy = compare_text(reference_text, transcription)
                comparison_type = "Text Comparison (No Phonemization Backend)"

        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception as e:
                    logger.error(f"Failed to delete file {filepath}: {str(e)}")

        # Render results template
        template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Pronunciation Checker Results</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                pre { background: #f4f4f4; padding: 10px; border-radius: 5px; }
                .error { color: red; }
                a { text-decoration: none; color: #007bff; }
                a:hover { text-decoration: underline; }
            </style>
        </head>
        <body>
            <h1>Pronunciation Checker Results</h1>
            {% with messages = get_flashed_messages() %}
                {% if messages %}
                    {% for message in messages %}
                        <p class="error">{{ message }}</p>
                    {% endfor %}
                {% endif %}
            {% endwith %}
            <h2>Transcription:</h  <p>{{ transcription }}</p>
            <h2>{{ comparison_type }}:</h2>
            <pre>{{ comparison_text }}</pre>
            <h3>Accuracy: {{ accuracy|round(2) }}%</h3>
            <a href="{{ url_for('analyze') }}">Try another</a>
        </body>
        </html>
        """
        return render_template_string(template, transcription=transcription, 
                                   comparison_text=comparison_text, accuracy=accuracy, 
                                   comparison_type=comparison_type)

    # GET method: show upload form
    template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Pronunciation Checker</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .error { color: red; }
            input[type=text], input[type=file] { margin: 10px 0; }
            input[type=submit] { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            input[type=submit]:hover { background: #0056b3; }
        </style>
    </head>
    <body>
        <h1>Pronunciation Checker</h1>
        {% with messages = get_flashed_messages() %}
            {% if messages %}
                {% for message in messages %}
                    <p class="error">{{ message }}</p>
                {% endfor %}
            {% endif %}
        {% endwith %}
        <form method="post" enctype="multipart/form-data">
            <label for="reference">Reference Text:</label><br>
            <input type="text" id="reference" name="reference" size="60" required><br><br>
            <label for="audio">Audio File (wav, mp3, flac, ogg):</label><br>
            <input type="file" id="audio" name="audio" accept="audio/*" required><br><br>
            <input type="submit" value="Analyze">
        </form>
    </body>
    </html>
    """
    return render_template_string(template)

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)