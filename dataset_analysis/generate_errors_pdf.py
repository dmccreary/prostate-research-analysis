import os
import pandas as pd
import reportlab.rl_config
reportlab.rl_config.pageCompression = 0  # Ensure plain-text searchability in all PDF viewers

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

# Register TrueType fonts for full Unicode support
pdfmetrics.registerFont(TTFont('Arial', 'C:/Windows/Fonts/arial.ttf'))
pdfmetrics.registerFont(TTFont('Arial-Bold', 'C:/Windows/Fonts/arialbd.ttf'))
pdfmetrics.registerFont(TTFont('Arial-Italic', 'C:/Windows/Fonts/ariali.ttf'))
pdfmetrics.registerFont(TTFont('Arial-BoldItalic', 'C:/Windows/Fonts/arialbi.ttf'))

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont('Arial', 8)
        self.setFillColor(colors.HexColor('#666666'))
        
        # Running Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(40, letter[1] - 30, "PCRSG Criteria Evaluation — Potential Labeling Errors (N = 23)")
            self.setStrokeColor(colors.HexColor('#CBD5E0'))
            self.setLineWidth(0.5)
            self.line(40, letter[1] - 34, letter[0] - 40, letter[1] - 34)
            
        # Running Footer
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 40, 25, page_str)
        self.drawString(40, 25, "Source: labeled-dataset-v2-analysis.csv | Prostate Cancer Results Study Group (PCRSG) Audit")
        self.setStrokeColor(colors.HexColor('#CBD5E0'))
        self.setLineWidth(0.5)
        self.line(40, 35, letter[0] - 40, 35)
        self.restoreState()

def build_pdf():
    input_csv = r'D:\prostate-research-analysis\prostate-research-analysis\dataset_analysis\labeled-dataset-v2-analysis.csv'
    output_pdf = r'D:\prostate-research-analysis\prostate-research-analysis\dataset_analysis\potential_labeling_errors.pdf'
    
    df = pd.read_csv(input_csv, keep_default_na=False)
    errs = df[df['Labeling Discrepancy'].str.startswith('Potential Labeling Error:')].copy()
    
    doc = SimpleDocTemplate(
        output_pdf,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=45,
        bottomMargin=45
    )
    
    content_width = letter[0] - 80  # 612 - 80 = 532 pt
    
    # Styles
    doc_title_style = ParagraphStyle(
        'DocTitle',
        fontName='Arial-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#1B365D'),
        spaceAfter=4
    )
    
    doc_sub_style = ParagraphStyle(
        'DocSub',
        fontName='Arial',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#4A5568'),
        spaceAfter=12
    )
    
    item_header_style = ParagraphStyle(
        'ItemHeader',
        fontName='Arial-Bold',
        fontSize=10.5,
        leading=13,
        textColor=colors.white
    )
    
    title_style = ParagraphStyle(
        'ItemTitle',
        fontName='Arial-Bold',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor('#1A365D'),
        spaceBefore=1,
        spaceAfter=5
    )
    
    discrepancy_box_style = ParagraphStyle(
        'DiscrepancyText',
        fontName='Arial',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#742A2A')
    )
    
    abstract_heading_style = ParagraphStyle(
        'AbstractHeading',
        fontName='Arial-Bold',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor('#2D3748'),
        spaceBefore=4,
        spaceAfter=2
    )
    
    abstract_style = ParagraphStyle(
        'ItemAbstract',
        fontName='Arial',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#2D3748'),
        alignment=4  # Justified
    )
    
    story = []
    
    # Document Header
    story.append(Paragraph("PCRSG Article Inclusion Criteria Audit", doc_title_style))
    story.append(Paragraph(
        "<b>Catalog of Potential Labeling Errors (N = 23)</b> &nbsp;|&nbsp; Source: <code>labeled-dataset-v2-analysis.csv</code><br/>"
        "This document lists all 23 articles identified as potential labeling discrepancies against the Prostate Cancer Results Study Group (PCRSG) criteria: "
        "<b>21 False Positives</b> (Ground Truth = 1 in dataset, but abstract contains explicit criteria violations) and "
        "<b>2 False Negatives</b> (Ground Truth = 0 in dataset, but abstract explicitly meets all criteria).",
        doc_sub_style
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#1B365D'), spaceBefore=0, spaceAfter=12))
    
    def escape_xml(text):
        return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    
    for i, (_, r) in enumerate(errs.iterrows(), 1):
        pmid = str(r['PMID']).strip()
        gt = str(r['Ground Truth Label']).strip()
        title = str(r['Title']).strip()
        abstract = str(r['Abstract']).strip()
        disc = str(r['Labeling Discrepancy']).strip()
        
        title_escaped = escape_xml(title)
        abstract_escaped = escape_xml(abstract) if abstract else "<i>[No abstract available]</i>"
        disc_escaped = escape_xml(disc)
        
        # Color coding: Crimson/Maroon if GT=1, Navy if GT=0
        banner_color = colors.HexColor('#8C1D40') if gt == '1' else colors.HexColor('#005A9C')
        gt_label_text = "1 (Positive in dataset)" if gt == '1' else "0 (Negative in dataset)"
        
        # Header banner table
        header_text = f"<b>Article #{i} &nbsp;|&nbsp; PMID: {pmid} &nbsp;|&nbsp; Ground Truth in Dataset: {gt_label_text}</b>"
        header_para = Paragraph(header_text, item_header_style)
        header_table = Table([[header_para]], colWidths=[content_width])
        header_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), banner_color),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ]))
        
        # Title paragraph
        title_para = Paragraph(f"<b>Title:</b> {title_escaped}", title_style)
        
        # Specific discrepancy callout box
        disc_text = f"<b>Specific Discrepancy:</b><br/>{disc_escaped}"
        disc_para = Paragraph(disc_text, discrepancy_box_style)
        disc_table = Table([[disc_para]], colWidths=[content_width])
        disc_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#FFF5F5')),
            ('BOX', (0,0), (-1,-1), 0.75, colors.HexColor('#FEB2B2')),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ]))
        
        # Abstract heading & paragraph
        abs_heading = Paragraph("<b>Abstract:</b>", abstract_heading_style)
        abs_para = Paragraph(abstract_escaped, abstract_style)
        
        # Keep header, title, and discrepancy box together to prevent awkward page splits
        lead_block = KeepTogether([
            header_table,
            Spacer(1, 4),
            title_para,
            disc_table,
            Spacer(1, 4),
            abs_heading
        ])
        
        story.append(lead_block)
        story.append(abs_para)
        story.append(Spacer(1, 8))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#CBD5E0'), spaceBefore=2, spaceAfter=10))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF successfully generated at: {output_pdf}")

if __name__ == '__main__':
    build_pdf()
