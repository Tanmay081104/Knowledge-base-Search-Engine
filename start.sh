#!/bin/bash

# Render deployment startup script for Knowledge Base Search Engine

echo "Starting Knowledge Base Search Engine..."

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Download required models (if needed)
echo "Setting up models..."
python -c "
try:
    import spacy
    spacy.download('en_core_web_sm')
    print('SpaCy model downloaded successfully')
except Exception as e:
    print(f'SpaCy model download failed: {e}')
"

# Start the application with Gunicorn
echo "Starting server with Gunicorn..."
exec gunicorn backend.app.main:app -c gunicorn.conf.py