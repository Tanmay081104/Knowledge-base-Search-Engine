"""
Simple PDF Q&A System with Groq
Reads entire PDF content and answers questions using Groq API
"""

import os
import tempfile
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import PyPDF2
from groq import Groq

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.1-8b-instant"

# Initialize FastAPI app
app = FastAPI(title="PDF Q&A System with Groq")

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# Global variable to store PDF content
pdf_content: str = ""
pdf_filename: str = ""

# Pydantic models
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    success: bool
    question: str
    answer: str
    pdf_filename: str

# HTML Frontend
HTML_CONTENT = '''
<!DOCTYPE html>
<html>
<head>
    <title>PDF Q&A System with Groq</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            max-width: 800px; 
            margin: 0 auto; 
            padding: 20px;
            background-color: #f0f2f5;
        }
        .container { 
            background: white; 
            padding: 30px; 
            border-radius: 15px; 
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        h1 { 
            color: #2c3e50; 
            text-align: center;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 30px;
        }
        .section { 
            margin-bottom: 30px; 
            padding: 20px; 
            border: 2px solid #ecf0f1; 
            border-radius: 10px;
            background-color: #fdfdfd;
        }
        .upload-area { 
            border: 3px dashed #3498db; 
            padding: 30px; 
            text-align: center; 
            border-radius: 10px;
            background-color: #ebf3fd;
            transition: all 0.3s;
        }
        .upload-area:hover {
            border-color: #2980b9;
            background-color: #ddeeff;
        }
        input[type="file"] { 
            margin: 15px 0;
            padding: 8px;
        }
        button { 
            background: linear-gradient(45deg, #3498db, #2980b9);
            color: white; 
            padding: 12px 25px; 
            border: none; 
            border-radius: 8px; 
            cursor: pointer;
            font-size: 16px;
            margin: 8px;
            transition: all 0.3s;
        }
        button:hover { 
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(52, 152, 219, 0.3);
        }
        textarea { 
            width: 100%; 
            min-height: 80px; 
            border: 2px solid #bdc3c7; 
            border-radius: 8px; 
            padding: 15px;
            font-size: 16px;
            resize: vertical;
            box-sizing: border-box;
        }
        textarea:focus {
            border-color: #3498db;
            outline: none;
        }
        .answer { 
            background: linear-gradient(135deg, #d5f4e6, #c8f0df);
            padding: 20px; 
            border-radius: 10px; 
            margin-top: 15px;
            border-left: 5px solid #27ae60;
            box-shadow: 0 2px 10px rgba(39, 174, 96, 0.1);
        }
        .error { 
            background: linear-gradient(135deg, #ffeaa7, #fdcb6e);
            color: #d63031; 
            padding: 20px; 
            border-radius: 10px; 
            margin-top: 15px;
            border-left: 5px solid #e17055;
        }
        .success { 
            background: linear-gradient(135deg, #a8e6cf, #88d8a3);
            color: #00b894; 
            padding: 20px; 
            border-radius: 10px; 
            margin-top: 15px;
            border-left: 5px solid #00b894;
        }
        .loading {
            display: inline-block;
            width: 25px;
            height: 25px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #3498db;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .pdf-info {
            background: linear-gradient(135deg, #e8f4fd, #d1ecf1);
            padding: 15px;
            border-radius: 8px;
            margin-top: 10px;
            border-left: 4px solid #3498db;
        }
        .question-examples {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
        }
        .example-question {
            background: #e9ecef;
            padding: 8px 12px;
            margin: 5px 0;
            border-radius: 5px;
            cursor: pointer;
            transition: background 0.2s;
        }
        .example-question:hover {
            background: #dee2e6;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📄 PDF Q&A System</h1>
        <p class="subtitle">Upload a PDF and ask questions</p>
        
        <div class="section">
            <h2>📤 Upload PDF Document</h2>
            <div class="upload-area">
                <p><strong>📎 Choose your PDF file</strong></p>
                <p>The entire content will be read and made available for questions</p>
                <input type="file" id="fileInput" accept=".pdf" />
                <br>
                <button onclick="uploadPDF()">📄 Upload & Process PDF</button>
            </div>
            <div id="uploadResult"></div>
        </div>
        
        <div class="section">
            <h2>❓ Ask Questions</h2>
            <textarea id="questionInput" placeholder="Ask any question about your PDF content..."></textarea>
            <br>
            <button onclick="askQuestion()">🤖 Get Answer from Groq</button>
            
            <div class="question-examples">
                <strong>💡 Example Questions:</strong>
                <div class="example-question" onclick="setQuestion('What is the main topic of this document?')">
                    What is the main topic of this document?
                </div>
                <div class="example-question" onclick="setQuestion('Can you summarize the key points?')">
                    Can you summarize the key points?
                </div>
                <div class="example-question" onclick="setQuestion('What are the important conclusions?')">
                    What are the important conclusions?
                </div>
            </div>
            
            <div id="queryResult"></div>
        </div>
        
        <div class="section">
            <h2>📊 System Status</h2>
            <button onclick="checkStatus()">📈 Check System Status</button>
            <div id="statusResult"></div>
        </div>
    </div>

    <script>
        function setQuestion(question) {
            document.getElementById('questionInput').value = question;
        }
        
        async function uploadPDF() {
            const fileInput = document.getElementById('fileInput');
            const resultDiv = document.getElementById('uploadResult');
            
            if (!fileInput.files[0]) {
                resultDiv.innerHTML = '<div class="error">❌ Please select a PDF file to upload.</div>';
                return;
            }
            
            if (!fileInput.files[0].name.toLowerCase().endsWith('.pdf')) {
                resultDiv.innerHTML = '<div class="error">❌ Please select a PDF file only.</div>';
                return;
            }
            
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            
            resultDiv.innerHTML = '<div class="loading"></div> 📖 Reading entire PDF content...';
            
            try {
                const response = await fetch('/upload-pdf', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.success) {
                    resultDiv.innerHTML = `<div class="success">
                        ✅ <strong>PDF Successfully Loaded!</strong><br>
                        📄 File: ${result.filename}<br>
                        📏 Content Length: ${result.content_length.toLocaleString()} characters<br>
                        📖 Pages Read: ${result.pages_count}<br>
                        🎯 Ready to answer questions!
                    </div>`;
                } else {
                    resultDiv.innerHTML = `<div class="error">❌ Upload failed: ${result.message}</div>`;
                }
            } catch (error) {
                resultDiv.innerHTML = `<div class="error">❌ Error: ${error.message}</div>`;
            }
        }
        
        async function askQuestion() {
            const questionInput = document.getElementById('questionInput');
            const resultDiv = document.getElementById('queryResult');
            
            if (!questionInput.value.trim()) {
                resultDiv.innerHTML = '<div class="error">❌ Please enter a question.</div>';
                return;
            }
            
            resultDiv.innerHTML = '<div class="loading"></div> 🧠 Groq is analyzing the PDF content and generating answer...';
            
            try {
                const response = await fetch('/ask-question', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        question: questionInput.value
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    resultDiv.innerHTML = `<div class="answer">
                        <h3>🤖 Groq's Answer:</h3>
                        <p><strong>Question:</strong> ${result.question}</p>
                        <hr style="margin: 15px 0; border: none; border-top: 1px solid #bdc3c7;">
                        <p><strong>Answer:</strong></p>
                        <div style="font-size: 16px; line-height: 1.6;">${result.answer}</div>
                        <hr style="margin: 15px 0; border: none; border-top: 1px solid #bdc3c7;">
                        <small>📄 Source: ${result.pdf_filename}</small>
                    </div>`;
                } else {
                    resultDiv.innerHTML = `<div class="error">❌ ${result.answer}</div>`;
                }
            } catch (error) {
                resultDiv.innerHTML = `<div class="error">❌ Error: ${error.message}</div>`;
            }
        }
        
        async function checkStatus() {
            const resultDiv = document.getElementById('statusResult');
            
            try {
                const response = await fetch('/status');
                const result = await response.json();
                
                resultDiv.innerHTML = `<div class="pdf-info">
                    <strong>🚀 System Status:</strong> ${result.status}<br>
                    <strong>📄 PDF Loaded:</strong> ${result.pdf_loaded ? 'Yes ✅' : 'No ❌'}<br>
                    <strong>📁 Current File:</strong> ${result.current_pdf || 'None'}<br>
                    <strong>📏 Content Length:</strong> ${result.content_length.toLocaleString()} characters<br>
                    <strong>🤖 AI Model:</strong> ${result.groq_model}<br>
                    <strong>⏰ Timestamp:</strong> ${new Date(result.timestamp).toLocaleString()}
                </div>`;
            } catch (error) {
                resultDiv.innerHTML = `<div class="error">❌ Status check failed: ${error.message}</div>`;
            }
        }
        
        // Auto-check status on page load
        window.onload = function() {
            checkStatus();
        };
    </script>
</body>
</html>
'''

def extract_pdf_text(file_path: str) -> tuple[str, int]:
    """Extract all text from PDF file and return text content and page count"""
    text_content = ""
    page_count = 0
    
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            page_count = len(pdf_reader.pages)
            
            print(f"📖 Reading PDF with {page_count} pages...")
            
            for page_num in range(page_count):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                text_content += page_text + "\n"
                print(f"✅ Processed page {page_num + 1}/{page_count}")
            
        return text_content.strip(), page_count
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")

async def generate_answer_from_pdf(question: str, pdf_content: str, filename: str) -> str:
    """Generate answer using Groq based on PDF content"""
    if not pdf_content:
        return "❌ No PDF content available. Please upload a PDF file first."
    
    # Create a comprehensive prompt with the entire PDF content
    prompt = f"""You are an expert assistant analyzing a PDF document. Answer the user's question based ONLY on the content provided below.

PDF Document: {filename}

COMPLETE PDF CONTENT:
{pdf_content}

USER QUESTION: {question}

INSTRUCTIONS:
- Answer based ONLY on the PDF content provided above
- Be comprehensive and detailed in your response
- If the question cannot be answered from the PDF content, say so clearly
- Quote relevant sections from the PDF when helpful
- Provide a clear, well-structured answer

ANSWER:"""

    try:
        print(f"🤖 Sending to Groq: Question about {filename}")
        print(f"📄 PDF Content Length: {len(pdf_content)} characters")
        
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,  # Increased for detailed answers
            temperature=0.3,
        )
        
        answer = response.choices[0].message.content.strip()
        print(f"✅ Groq response generated successfully")
        return answer
        
    except Exception as e:
        error_msg = f"❌ Error generating answer with Groq: {str(e)}"
        print(error_msg)
        return error_msg

# Routes
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main frontend"""
    return HTMLResponse(content=HTML_CONTENT)

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process PDF file"""
    global pdf_content, pdf_filename
    
    # Validate file
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        print(f"📤 Received PDF file: {file.filename}")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        print(f"💾 Saved to temporary file: {tmp_file_path}")
        
        # Extract all text from PDF
        text_content, page_count = extract_pdf_text(tmp_file_path)
        
        # Store globally
        pdf_content = text_content
        pdf_filename = file.filename
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        print(f"🎉 PDF processing completed successfully!")
        print(f"📊 Stats: {len(text_content)} characters, {page_count} pages")
        
        return {
            "success": True,
            "message": f"PDF {file.filename} processed successfully!",
            "filename": file.filename,
            "content_length": len(text_content),
            "pages_count": page_count
        }
        
    except Exception as e:
        # Clean up temp file if it exists
        try:
            os.unlink(tmp_file_path)
        except:
            pass
        
        error_msg = f"Processing failed: {str(e)}"
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/ask-question", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """Process user question about the PDF content"""
    global pdf_content, pdf_filename
    
    print(f"❓ Question received: {request.question}")
    
    try:
        # Generate answer using Groq
        answer = await generate_answer_from_pdf(request.question, pdf_content, pdf_filename)
        
        return QueryResponse(
            success=True,
            question=request.question,
            answer=answer,
            pdf_filename=pdf_filename or "No PDF loaded"
        )
        
    except Exception as e:
        error_msg = f"Error processing question: {str(e)}"
        print(f"❌ {error_msg}")
        return QueryResponse(
            success=False,
            question=request.question,
            answer=error_msg,
            pdf_filename=pdf_filename or "No PDF loaded"
        )

@app.get("/status")
async def get_status():
    """Get system status"""
    return {
        "status": "",
        "pdf_loaded": bool(pdf_content),
        "current_pdf": pdf_filename or None,
        "content_length": len(pdf_content) if pdf_content else 0,
        "groq_model": GROQ_MODEL,
        "timestamp": "2024-01-01T00:00:00"
    }

if __name__ == "__main__":
    import uvicorn
    print("PDF Q and A")
    print("Upload any PDF and ask questions about its content!")
    print("🌐 Opening at: http://127.0.0.1:8000")
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )