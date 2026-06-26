import os
import datetime
import logging
import tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import config

logger = logging.getLogger("SHM_PDFGenerator")

def generate_pdf_report(results, source_file_path):
    """Generates a professional PDF report summarizing the structural health analysis."""
    logger.info("Initializing PDF report generation...")
    
    # 1. Create unique report filename
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.basename(source_file_path)
    clean_base_name = os.path.splitext(base_name)[0].replace(" ", "_")
    report_filename = f"SHM_Report_{clean_base_name}_{timestamp_str}.pdf"
    report_path = os.path.join(config.REPORTS_DIR, report_filename)
    
    try:
        # 2. Generate Matplotlib chart for the report
        temp_img_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        temp_img_path = temp_img_file.name
        temp_img_file.close() # Close handle so matplotlib can write to it
        
        # Plot pie chart
        labels = ['Healthy', 'Minor Damage', 'Severe Damage']
        counts = [results['healthy_count'], results['minor_count'], results['severe_count']]
        colors_list = [config.COLOR_HEALTHY, config.COLOR_MINOR, config.COLOR_SEVERE]
        
        # Filter classes with 0 count to avoid warnings or overlapping slices in pie chart
        filtered_labels = []
        filtered_counts = []
        filtered_colors = []
        for l, c, col in zip(labels, counts, colors_list):
            if c > 0:
                filtered_labels.append(f"{l} ({c})")
                filtered_counts.append(c)
                filtered_colors.append(col)
                
        plt.figure(figsize=(6, 4))
        if len(filtered_counts) > 0:
            plt.pie(
                filtered_counts, 
                labels=filtered_labels, 
                colors=filtered_colors, 
                autopct='%1.1f%%', 
                startangle=140,
                textprops={'fontsize': 10, 'weight': 'bold'}
            )
        else:
            # Fallback if no samples predicted
            plt.text(0.5, 0.5, "No Data Predicted", ha='center', va='center')
            
        plt.title("Structural Condition Distribution", fontsize=12, fontweight='bold', pad=15)
        plt.tight_layout()
        plt.savefig(temp_img_path, dpi=150)
        plt.close()
        logger.info(f"Temporary pie chart saved to: {temp_img_path}")
        
        # 3. Setup PDF document template
        doc = SimpleDocTemplate(
            report_path,
            pagesize=letter,
            rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54
        )
        
        styles = getSampleStyleSheet()
        
        # Define Custom Styles
        title_style = ParagraphStyle(
            'ReportTitle',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=22,
            leading=26,
            textColor=colors.HexColor(config.COLOR_TEXT),
            spaceAfter=6
        )
        
        subtitle_style = ParagraphStyle(
            'ReportSubtitle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=11,
            leading=14,
            textColor=colors.HexColor(config.COLOR_MUTED),
            spaceAfter=20
        )
        
        h1_style = ParagraphStyle(
            'SectionH1',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=14,
            leading=18,
            textColor=colors.HexColor(config.COLOR_TEXT),
            spaceBefore=15,
            spaceAfter=8,
            keepWithNext=True
        )
        
        body_style = ParagraphStyle(
            'ReportBody',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            leading=14,
            textColor=colors.HexColor(config.COLOR_TEXT)
        )
        
        bold_body_style = ParagraphStyle(
            'ReportBodyBold',
            parent=body_style,
            fontName='Helvetica-Bold'
        )

        table_header_style = ParagraphStyle(
            'TableHeader',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=10,
            leading=12,
            textColor=colors.white
        )

        # Build Document Story
        story = []
        
        # Header Block
        story.append(Paragraph("Structural Health Monitoring Analysis Report", title_style))
        story.append(Paragraph(f"Generated via Artificial Neural Network (ANN) Diagnostic Engine &bull; {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", subtitle_style))
        
        # Separator line
        line_data = [['']]
        line_table = Table(line_data, colWidths=[doc.width])
        line_table.setStyle(TableStyle([
            ('LINEABOVE', (0,0), (-1,-1), 1.5, colors.HexColor(config.COLOR_PRIMARY)),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(line_table)
        story.append(Spacer(1, 15))
        
        # File & Dataset Metadata Table
        meta_data = [
            [
                Paragraph("<b>Uploaded File Name:</b>", body_style), 
                Paragraph(base_name, body_style),
                Paragraph("<b>Date of Analysis:</b>", body_style),
                Paragraph(datetime.datetime.now().strftime('%Y-%m-%d'), body_style)
            ],
            [
                Paragraph("<b>Total Records Analyzed:</b>", body_style),
                Paragraph(f"{results['total_samples']} sensor readings", body_style),
                Paragraph("<b>Monitoring System:</b>", body_style),
                Paragraph("SHM-ANN v1.0", body_style)
            ]
        ]
        meta_table = Table(meta_data, colWidths=[1.5*inch, 2.0*inch, 1.3*inch, 2.2*inch])
        meta_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('TOPPADDING', (0,0), (-1,-1), 4),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 20))
        
        # 4. Results callout card (Overall Health Status)
        health_status = results['overall_health']
        if health_status == "Healthy":
            status_color = colors.HexColor(config.COLOR_HEALTHY)
            text_color = colors.white
        elif health_status == "Minor Damage":
            status_color = colors.HexColor(config.COLOR_MINOR)
            text_color = colors.white
        else:
            status_color = colors.HexColor(config.COLOR_SEVERE)
            text_color = colors.white
            
        status_card_style = ParagraphStyle(
            'StatusCardText',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=16,
            leading=20,
            textColor=text_color,
            alignment=1 # Centered
        )
        
        card_data = [
            [Paragraph(f"OVERALL STRUCTURAL HEALTH: {health_status.upper()}", status_card_style)],
            [Paragraph(f"Prediction Confidence: {results['overall_confidence']:.2f}%", ParagraphStyle('CardConf', parent=status_card_style, fontSize=11, leading=14))]
        ]
        card_table = Table(card_data, colWidths=[doc.width])
        card_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), status_color),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,0), 12),
            ('BOTTOMPADDING', (0,-1), (-1,-1), 12),
            ('INNERGRID', (0,0), (-1,-1), 0, colors.transparent),
            ('BOX', (0,0), (-1,-1), 1, status_color),
        ]))
        story.append(card_table)
        story.append(Spacer(1, 20))
        
        # 5. Two-column analysis layout (Statistics Table + Pie Chart)
        story.append(Paragraph("Sensor Reading Classifications Summary", h1_style))
        
        # Stats Table
        stats_headers = [
            Paragraph("Health Condition", table_header_style), 
            Paragraph("Sample Count", table_header_style), 
            Paragraph("Distribution (%)", table_header_style)
        ]
        
        stats_rows = [
            [
                Paragraph("<b>Healthy</b>", body_style), 
                Paragraph(str(results['healthy_count']), body_style), 
                Paragraph(f"{results['healthy_pct']:.2f}%", body_style)
            ],
            [
                Paragraph("<b>Minor Damage</b>", body_style), 
                Paragraph(str(results['minor_count']), body_style), 
                Paragraph(f"{results['minor_pct']:.2f}%", body_style)
            ],
            [
                Paragraph("<b>Severe Damage</b>", body_style), 
                Paragraph(str(results['severe_count']), body_style), 
                Paragraph(f"{results['severe_pct']:.2f}%", body_style)
            ],
            [
                Paragraph("<b>Total Readings</b>", bold_body_style), 
                Paragraph(str(results['total_samples']), bold_body_style), 
                Paragraph("100.00%", bold_body_style)
            ]
        ]
        
        table_content = [stats_headers] + stats_rows
        stats_table = Table(table_content, colWidths=[1.8*inch, 1.2*inch, 1.2*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor(config.COLOR_TEXT)),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.lightgrey),
            ('LINEBELOW', (0,-1), (-1,-1), 1.5, colors.HexColor(config.COLOR_TEXT)),
        ]))
        
        # Embed chart image
        chart_image = Image(temp_img_path, width=3.0*inch, height=2.0*inch)
        
        # Combine Stats table and Chart into a single side-by-side Table layout
        layout_data = [[stats_table, chart_image]]
        layout_table = Table(layout_data, colWidths=[4.2*inch, 3.2*inch])
        layout_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING', (1,0), (1,0), 10),
            ('RIGHTPADDING', (0,0), (0,0), 0),
        ]))
        story.append(layout_table)
        story.append(Spacer(1, 15))
        
        # 6. Engineering Recommendations
        story.append(Paragraph("Engineering Action Plan & Recommendations", h1_style))
        
        recommendations_texts = config.RECOMMENDATIONS.get(health_status, [])
        recommendation_bullet_style = ParagraphStyle(
            'RecBullet',
            parent=body_style,
            leftIndent=20,
            firstLineIndent=-10,
            spaceAfter=6
        )
        
        rec_story_elements = []
        for text in recommendations_texts:
            rec_story_elements.append(Paragraph(f"&bull;&nbsp;&nbsp;{text}", recommendation_bullet_style))
            
        rec_card = Table([[rec_story_elements]], colWidths=[doc.width])
        rec_card.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8F9FA")),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#BDC3C7")),
            ('TOPPADDING', (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 10),
            ('LEFTPADDING', (0,0), (-1,-1), 12),
            ('RIGHTPADDING', (0,0), (-1,-1), 12),
        ]))
        story.append(rec_card)
        story.append(Spacer(1, 20))
        
        # Disclaimer block
        disclaimer_style = ParagraphStyle(
            'DisclaimerText',
            parent=styles['Normal'],
            fontName='Helvetica-Oblique',
            fontSize=8,
            leading=10,
            textColor=colors.HexColor(config.COLOR_MUTED),
            alignment=1 # Centered
        )
        story.append(Paragraph("Disclaimer: This report was automatically compiled by an Artificial Neural Network model trained on sensor metrics. Results should be verified by a licensed structural engineer before implementing critical safety changes.", disclaimer_style))
        
        # Build PDF
        doc.build(story)
        logger.info(f"PDF report successfully generated at: {report_path}")
        
        # Cleanup temporary image file
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
            logger.info("Temporary chart image cleaned up.")
            
        return report_path
    except Exception as e:
        logger.error(f"Failed to generate PDF report: {e}", exc_info=True)
        raise e
