import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

# Validate required environment variables
required_vars = ['GEMINI_API_KEY', 'GOOGLE_APPLICATION_CREDENTIALS']
for var in required_vars:
    if not os.getenv(var):
        print(f"Warning: {var} is not set in the environment variables.")

class Config:
    # Local model configuration
    USE_LOCAL_MODEL = False  # Using Gemini API by default for better performance
    LOCAL_MODEL_NAME = "facebook/bart-large-cnn"  # Fallback model, not used when USE_LOCAL_MODEL is False
    
    # Get API key from environment
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    if not GEMINI_API_KEY or GEMINI_API_KEY == 'your-gemini-api-key':
        print("Warning: GEMINI_API_KEY is not properly set in .env file")
        GEMINI_API_KEY = None
    
    # Google Cloud Vision Configuration
    # Note: Set GOOGLE_APPLICATION_CREDENTIALS in your environment variables
    # pointing to your service account key file
    
    # Model configuration
    MODEL_NAME = "gemini-1.5-pro-latest"  # Updated to the latest Gemini model
    
    # Generation configuration - optimized for larger text and better quality
    GENERATION_CONFIG = {
        "temperature": 0.5,      # Balanced between creativity and accuracy
        "top_p": 0.95,          # Slightly higher for better quality
        "top_k": 40,            # Increased for better quality
        "max_output_tokens": 4096,  # Increased to handle ~1000 words (4 tokens per word on average)
        "candidate_count": 1,    # Only generate one response for speed
        "response_mime_type": "text/plain"
    }
    
    # Text processing settings
    MAX_INPUT_TOKENS = 32000      # Maximum tokens for input (Gemini 1.5 Pro limit)
    CHUNK_SIZE = 15000            # Process text in chunks of ~3750 words
    CHUNK_OVERLAP = 500           # Overlap between chunks for context
    
    # Safety settings
    SAFETY_SETTINGS = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    
    # File upload settings
    ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp', 'webp'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    
    # Output settings
    SUMMARY_RATIO = 0.3  # Ratio of original text to include in summary
    
    # Temporary directory for file processing
    UPLOAD_FOLDER = 'temp_uploads'
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
