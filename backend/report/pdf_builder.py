import os
import re
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable

def create_pdf(report_payload: dict) -> BytesIO:
    """
    Generates a medical-grade PDF from the report payload.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=letter,
        rightMargin=50, leftMargin=50,
        topMargin=50, bottomMargin=50
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=HexColor('#0d1528'),
        spaceAfter=20,
    )
    
    heading_style = ParagraphStyle(
        'HeadingStyle',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=HexColor('#00d4ff'),
        spaceBefore=15,
        spaceAfter=10,
    )
    
    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['Normal'],
        fontSize=10,
        lineHeight=14,
        spaceAfter=10,
    )
    
    disclaimer_style = ParagraphStyle(
        'DisclaimerStyle',
        parent=styles['Italic'],
        fontSize=8,
        textColor=HexColor('#8899aa'),
        spaceBefore=30,
    )

    story = []

    # Header
    story.append(Paragraph("Clinical AI Copilot", title_style))
    story.append(Paragraph(f"<b>Report ID:</b> {report_payload.get('report_id')}", body_style))
    story.append(HRFlowable(width="100%", thickness=1, color=HexColor('#e8f4f8'), spaceBefore=10, spaceAfter=20))

    # We can use the full_markdown_report but it might have markdown characters.
    # We will do a basic strip of ** and # for clean PDF reading.
    raw_md = report_payload.get("full_markdown_report", "No report available.")
    
    # Very basic markdown conversion to Platypus Paragraph tags
    # Replace **text** with <b>text</b>
    raw_md = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', raw_md)
    # Replace *text* with <i>text</i>
    raw_md = re.sub(r'\*(.*?)\*', r'<i>\1</i>', raw_md)
    # Remove # headers
    raw_md = re.sub(r'#+\s+', '', raw_md)
    
    # Split by newlines
    lines = raw_md.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if it looks like a section header (1., 2., 3. or uppercase)
        if re.match(r'^\d+\.\s+<b>', line) or (line.isupper() and len(line) > 5):
            story.append(Spacer(1, 10))
            story.append(Paragraph(line, heading_style))
        else:
            story.append(Paragraph(line, body_style))

    # Footer Disclaimer
    story.append(HRFlowable(width="100%", thickness=1, color=HexColor('#e8f4f8'), spaceBefore=30, spaceAfter=10))
    story.append(Paragraph(report_payload.get("disclaimer", ""), disclaimer_style))

    doc.build(story)
    buffer.seek(0)
    return buffer
