from flask import Flask, render_template, request, jsonify, session, send_file
import os
import logging
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import secrets

# Import utility modules
from utils.genai_utils import call_genai
from utils.audio_utils import text_to_audio
from utils.code_executor import detect_dependencies, save_code_to_file
from utils.image_utils import generate_images, get_model_info

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', secrets.token_hex(16))

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('generated_audio', exist_ok=True)
os.makedirs('generated_code', exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS