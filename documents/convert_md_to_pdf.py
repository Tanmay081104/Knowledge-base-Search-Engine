#!/usr/bin/env python3
"""
Simple script to convert markdown files to PDF using weasyprint.
"""

import sys
import os
from markdown import markdown
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration

def convert_md_to_pdf(md_file, pdf_file):
    """Convert a markdown file to PDF."""
    
    # Read the markdown file
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown(md_content, extensions=['extra', 'codehilite'])
    
    # Basic CSS for better formatting
    css_content = """
    body { 
        font-family: Arial, sans-serif; 
        line-height: 1.6; 
        margin: 40px;
        font-size: 12pt;
    }
    h1, h2, h3, h4, h5, h6 { 
        color: #333; 
        margin-top: 20px;
        margin-bottom: 10px;
    }
    h1 { font-size: 24pt; }
    h2 { font-size: 20pt; }
    h3 { font-size: 16pt; }
    code { 
        background-color: #f4f4f4; 
        padding: 2px 4px; 
        border-radius: 3px;
        font-family: 'Courier New', monospace;
    }
    pre { 
        background-color: #f4f4f4; 
        padding: 10px; 
        border-radius: 5px;
        overflow-x: auto;
    }
    blockquote {
        border-left: 4px solid #ddd;
        margin: 0;
        padding-left: 20px;
        color: #666;
    }
    """
    
    # Create HTML with CSS
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>{css_content}</style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # Convert to PDF
    font_config = FontConfiguration()
    html_obj = HTML(string=full_html)
    css_obj = CSS(string=css_content, font_config=font_config)
    
    html_obj.write_pdf(pdf_file, stylesheets=[css_obj], font_config=font_config)
    print(f"✓ Converted {md_file} to {pdf_file}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python convert_md_to_pdf.py <input.md> <output.pdf>")
        sys.exit(1)
    
    md_file = sys.argv[1]
    pdf_file = sys.argv[2]
    
    if not os.path.exists(md_file):
        print(f"Error: File {md_file} does not exist")
        sys.exit(1)
    
    try:
        convert_md_to_pdf(md_file, pdf_file)
    except Exception as e:
        print(f"Error converting {md_file} to PDF: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()