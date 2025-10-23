#!/usr/bin/env python3
"""
Alternative startup script for Render deployment
Uses the simplified main application
"""
import os
import subprocess
import sys

def main():
    print("🚀 Starting Knowledge Base Search Engine on Render...")
    
    # Get port from environment (Render sets this)
    port = os.getenv('PORT', '8000')
    
    # Use simplified version for more reliable deployment
    cmd = [
        'gunicorn',
        '--bind', f'0.0.0.0:{port}',
        '--workers', '2',
        '--worker-class', 'uvicorn.workers.UvicornWorker',
        '--timeout', '120',
        '--access-logfile', '-',
        '--error-logfile', '-',
        'simple_main:app'  # Use the simplified version
    ]
    
    print(f"📡 Starting server on port {port}")
    print(f"🔧 Command: {' '.join(cmd)}")
    
    # Execute the command
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()