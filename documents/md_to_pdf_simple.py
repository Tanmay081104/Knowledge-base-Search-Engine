#!/usr/bin/env python3
"""
Simple markdown to PDF converter using reportlab.
"""

import sys
import os
import re
from markdown import markdown
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
from reportlab.lib.colors import black, blue, gray
from html.parser import HTMLParser
from io import StringIO

class HTMLToReportLab(HTMLParser):
    def __init__(self):
        super().__init__()
        self.output = []
        self.styles = getSampleStyleSheet()
        self.current_style = 'Normal'
        self.text_buffer = StringIO()
        
        # Create custom styles (only if they don't exist)
        if 'CustomHeading1' not in self.styles:
            self.styles.add(ParagraphStyle(name='CustomHeading1', parent=self.styles['Heading1'], 
                                         fontSize=18, spaceAfter=12, spaceBefore=12))
        if 'CustomHeading2' not in self.styles:
            self.styles.add(ParagraphStyle(name='CustomHeading2', parent=self.styles['Heading2'], 
                                         fontSize=16, spaceAfter=10, spaceBefore=10))
        if 'CustomHeading3' not in self.styles:
            self.styles.add(ParagraphStyle(name='CustomHeading3', parent=self.styles['Heading3'], 
                                         fontSize=14, spaceAfter=8, spaceBefore=8))
        if 'CustomCode' not in self.styles:
            self.styles.add(ParagraphStyle(name='CustomCode', parent=self.styles['Code'], 
                                         fontSize=10, backColor='#f4f4f4'))
        
    def handle_starttag(self, tag, attrs):
        if tag == 'h1':
            self.current_style = 'CustomHeading1'
        elif tag == 'h2':
            self.current_style = 'CustomHeading2'  
        elif tag == 'h3':
            self.current_style = 'CustomHeading3'
        elif tag == 'p':
            self.current_style = 'Normal'
        elif tag == 'code':
            self.current_style = 'CustomCode'
        elif tag == 'pre':
            self.current_style = 'CustomCode'
        elif tag == 'strong' or tag == 'b':
            self.text_buffer.write('<b>')
        elif tag == 'em' or tag == 'i':
            self.text_buffer.write('<i>')
        elif tag == 'br':
            self.text_buffer.write('<br/>')
            
    def handle_endtag(self, tag):
        if tag in ['h1', 'h2', 'h3', 'p', 'pre']:
            text = self.text_buffer.getvalue().strip()
            if text:
                if tag == 'pre':
                    # For preformatted text, use Preformatted instead of Paragraph
                    self.output.append(Preformatted(text, self.styles['Code']))
                else:
                    self.output.append(Paragraph(text, self.styles[self.current_style]))
                self.output.append(Spacer(1, 6))
            self.text_buffer = StringIO()
        elif tag == 'strong' or tag == 'b':
            self.text_buffer.write('</b>')
        elif tag == 'em' or tag == 'i':
            self.text_buffer.write('</i>')
            
    def handle_data(self, data):
        # Clean up the text and add to buffer
        cleaned_data = data.replace('\n', ' ').strip()
        if cleaned_data:
            self.text_buffer.write(cleaned_data)
            
    def get_story(self):
        # Handle any remaining text
        text = self.text_buffer.getvalue().strip()
        if text:
            self.output.append(Paragraph(text, self.styles[self.current_style]))
        return self.output

def convert_md_to_pdf(md_file, pdf_file):
    """Convert a markdown file to PDF using reportlab."""
    
    # Read the markdown file
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown(md_content, extensions=['extra'])
    
    # Clean up HTML - remove some problematic tags
    html_content = re.sub(r'</?div[^>]*>', '', html_content)
    html_content = re.sub(r'</?span[^>]*>', '', html_content)
    
    # Parse HTML and convert to ReportLab elements
    parser = HTMLToReportLab()
    parser.feed(html_content)
    story = parser.get_story()
    
    if not story:
        # Fallback: create a simple text document
        styles = getSampleStyleSheet()
        story = [Paragraph("Document content could not be parsed properly.", styles['Normal']),
                Spacer(1, 12),
                Preformatted(md_content, styles['Code'])]
    
    # Create PDF
    doc = SimpleDocTemplate(pdf_file, pagesize=A4, 
                          rightMargin=72, leftMargin=72,
                          topMargin=72, bottomMargin=18)
    
    try:
        doc.build(story)
        print(f"✓ Converted {md_file} to {pdf_file}")
    except Exception as e:
        print(f"Error building PDF: {e}")
        # Create a simpler version
        styles = getSampleStyleSheet()
        simple_story = [
            Paragraph(f"Content from: {md_file}", styles['Title']),
            Spacer(1, 12),
            Preformatted(md_content, styles['Code'])
        ]
        doc.build(simple_story)
        print(f"✓ Created simplified PDF: {pdf_file}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python md_to_pdf_simple.py <input.md> <output.pdf>")
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