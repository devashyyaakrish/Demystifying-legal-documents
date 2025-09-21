# Legal Document Simplifier with Gemini AI and OCR

A web application that uses Google's Gemini AI and OCR technology to analyze and simplify complex legal documents, making them more understandable for non-lawyers.

## Features

- **Document Upload**: Support for PDF, DOCX, TXT, and image files (JPG, PNG, TIFF, BMP, WEBP)
- **OCR Processing**: Extract text from scanned documents and images using Google Cloud Vision
- **AI-Powered Simplification**: Convert complex legal jargon into plain, easy-to-understand language
- **Document Summarization**: Get concise summaries of lengthy legal documents
- **Responsive Web Interface**: Works on both desktop and mobile devices
- **Copy to Clipboard**: Easily copy simplified text to your clipboard

## Prerequisites

- Python 3.8 or higher
- Google Cloud account with Vision API enabled
- Google Gemini API key

## Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd legal-document-simplifier
   ```

2. **Create and activate a virtual environment**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Google Cloud credentials**
   - Create a service account in Google Cloud Console
   - Download the JSON key file
   - Set the environment variable to point to your key file:
     ```bash
     # Windows
     set GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account-key.json"
     
     # macOS/Linux
     export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account-key.json"
     ```

## Running the Application

1. **Start the Flask development server**
   ```bash
   python app.py
   ```

2. **Open your web browser** and navigate to:
   ```
   http://localhost:5000
   ```

3. **Upload a document** or paste text to get started

## API Endpoints

- `POST /api/analyze/document` - Process a document file
- `POST /api/analyze/text` - Process plain text

## Project Structure

- `app.py` - Main Flask application
- `document_processor.py` - Handles document processing and text simplification
- `config.py` - Configuration settings
- `templates/` - HTML templates
  - `index.html` - Web interface
- `static/` - Static files (CSS, JavaScript, images)
- `temp_uploads/` - Temporary storage for uploaded files (automatically created)

## Security Notes

- Never commit your API keys or service account files to version control
- The application is configured for development use
- For production deployment, consider:
  - Using a production WSGI server (e.g., Gunicorn, uWSGI)
  - Setting up HTTPS
  - Implementing proper authentication
  - Configuring appropriate CORS policies

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
