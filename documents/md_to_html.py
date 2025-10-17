#!/usr/bin/env python3
"""
Convert markdown to HTML for manual PDF conversion.
"""

import sys
import os
from markdown import markdown

def convert_md_to_html(md_file, html_file):
    """Convert a markdown file to HTML."""
    
    # Read the markdown file
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown(md_content, extensions=['extra', 'codehilite'])
    
    # Basic CSS for better formatting
    css_content = """
    <style>
    body { 
        font-family: Arial, sans-serif; 
        line-height: 1.6; 
        margin: 40px;
        font-size: 12pt;
        max-width: 800px;
    }
    h1, h2, h3, h4, h5, h6 { 
        color: #333; 
        margin-top: 20px;
        margin-bottom: 10px;
    }
    h1 { font-size: 24pt; border-bottom: 2px solid #333; }
    h2 { font-size: 20pt; border-bottom: 1px solid #666; }
    h3 { font-size: 16pt; }
    code { 
        background-color: #f4f4f4; 
        padding: 2px 4px; 
        border-radius: 3px;
        font-family: 'Courier New', monospace;
        font-size: 11pt;
    }
    pre { 
        background-color: #f4f4f4; 
        padding: 15px; 
        border-radius: 5px;
        overflow-x: auto;
        border: 1px solid #ddd;
    }
    pre code {
        background: none;
        padding: 0;
    }
    blockquote {
        border-left: 4px solid #ddd;
        margin: 0;
        padding-left: 20px;
        color: #666;
        font-style: italic;
    }
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 15px 0;
    }
    th, td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
    }
    th {
        background-color: #f2f2f2;
    }
    ul, ol {
        margin: 10px 0;
        padding-left: 30px;
    }
    li {
        margin: 5px 0;
    }
    @media print {
        body { margin: 20px; }
        h1 { page-break-before: always; }
        pre, blockquote { page-break-inside: avoid; }
    }
    </style>
    """
    
    # Create full HTML document
    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{os.path.splitext(os.path.basename(md_file))[0]}</title>
    {css_content}
</head>
<body>
    {html_content}
</body>
</html>"""
    
    # Write HTML file
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(full_html)
    
    print(f"✓ Converted {md_file} to {html_file}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python md_to_html.py <input.md> <output.html>")
        sys.exit(1)
    
    md_file = sys.argv[1]
    html_file = sys.argv[2]
    
    if not os.path.exists(md_file):
        print(f"Error: File {md_file} does not exist")
        sys.exit(1)
    
    try:
        convert_md_to_html(md_file, html_file)
    except Exception as e:
        print(f"Error converting {md_file} to HTML: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()