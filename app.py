import os
import tempfile
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from document_processor import DocumentProcessor
from config import Config

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH
app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize document processor
processor = None
try:
    print("Initializing DocumentProcessor...")
    processor = DocumentProcessor()
    print("DocumentProcessor initialized successfully")
except Exception as e:
    import traceback
    print(f"Error initializing DocumentProcessor: {str(e)}")
    print("Detailed traceback:")
    traceback.print_exc()
    processor = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze/document', methods=['POST'])
def analyze_document():
    """
    Analyze and simplify a legal document.
    
    Supported formats: PDF, DOCX, TXT, JPG, PNG, TIFF, BMP, WEBP
    Max file size: 16MB
    """
    if not processor:
        return jsonify({
            'success': False,
            'error': 'Document processor not initialized'
        }), 500

    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No file part in the request'
        }), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No selected file'
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': f'Unsupported file type. Allowed types: {Config.ALLOWED_EXTENSIONS}'
        }), 400

    try:
        # Save the uploaded file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)

        # Process the document
        result = processor.analyze_document(temp_path)

        # Clean up the temporary file
        try:
            os.remove(temp_path)
        except:
            pass

        return jsonify(result)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error processing file: {str(e)}'
        }), 500

@app.route('/api/analyze/text', methods=['POST'])
def analyze_text():
    """Analyze and simplify legal text directly."""
    if not processor:
        return jsonify({
            'success': False,
            'error': 'Document processor not initialized'
        }), 500

    data = request.get_json()
    if not data or 'text' not in data or not data['text'].strip():
        return jsonify({
            'success': False,
            'error': 'Text cannot be empty'
        }), 400

    try:
        # Save text to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write(data['text'])
            temp_path = temp_file.name

        # Process the text
        result = processor.analyze_document(temp_path)

        # Clean up the temporary file
        try:
            os.remove(temp_path)
        except:
            pass

        return jsonify(result)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error processing text: {str(e)}'
        }), 500

# Create necessary directories
os.makedirs('templates', exist_ok=True)
os.makedirs('uploads', exist_ok=True)

if __name__ == '__main__':
    try:
        print("Starting server...")
        print(f"Server will be available at http://localhost:5000")
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        print("Please check if port 5000 is available.")
        input("Press Enter to exit...")
