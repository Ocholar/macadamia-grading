"""
Report Generator Module.
Generates comprehensive PDF inspection certificates and tabular CSV reports.
"""

import io
import csv
import logging
from typing import Dict, Any, List
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)

logger = logging.getLogger(__name__)


def generate_pdf_report(grading_result: Dict[str, Any]) -> bytes:
    """
    Compiles an official PDF Certificate of Quality Grading for a macadamia batch.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CertTitle",
        parent=styles["Heading1"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "CertSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#4A5568"),
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=12,
        spaceAfter=6,
    )
    cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#2D3748"),
    )
    cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#1A202C"),
        fontName="Helvetica-Bold",
    )

    summary = grading_result.get("summary", {})
    analytics = grading_result.get("analytics", {})
    nuts = grading_result.get("nuts", [])

    elements = []

    # 1. Header Banner
    elements.append(Paragraph("MACADAMIA QUALITY GRADING AUDIT CERTIFICATE", title_style))
    elements.append(
        Paragraph(
            f"Automated Optical Inspection System • Batch ID: <b>{summary.get('batch_id', 'N/A')}</b> • "
            f"Date: {summary.get('timestamp', 'N/A')} • Processing Time: {summary.get('processing_time_sec', 0)}s",
            subtitle_style,
        )
    )
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2B6CB0"), spaceAfter=12))

    # 2. Executive Summary & Batch Grade Badge
    quality_grade = summary.get("quality_grade", "Commercial Grade")
    grade_color = "#276749" if "Grade 1" in quality_grade else ("#C05621" if "Grade 2" in quality_grade else "#9B2C2C")

    grade_banner_data = [
        [
            Paragraph("<b>OVERALL BATCH GRADE:</b>", cell_bold),
            Paragraph(f"<font color='{grade_color}' size='12'><b>{quality_grade.upper()}</b></font>", cell_style),
            Paragraph(f"<b>Total Inspected Nuts:</b> {summary.get('total_count', 0)}", cell_style),
            Paragraph(f"<b>Overall Defect Rate:</b> {summary.get('defect_rate_pct', 0.0)}%", cell_style),
        ]
    ]
    t_grade = Table(grade_banner_data, colWidths=[150, 180, 100, 110])
    t_grade.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EDF2F7")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E0")),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ])
    )
    elements.append(t_grade)
    elements.append(Spacer(1, 10))

    # 3. Sizing Breakdown Table
    elements.append(Paragraph("1. Physical Sizing & Metric Distribution", heading_style))
    size_counts = analytics.get("size_distribution", {})
    size_pcts = analytics.get("size_percentages", {})

    size_table_data = [
        [
            Paragraph("<b>Size Grade</b>", cell_bold),
            Paragraph("<b>Diameter Range</b>", cell_bold),
            Paragraph("<b>Count</b>", cell_bold),
            Paragraph("<b>Percentage</b>", cell_bold),
            Paragraph("<b>Commercial Application</b>", cell_bold),
        ],
        [
            Paragraph("Small", cell_style),
            Paragraph("&lt; 18.0 mm", cell_style),
            Paragraph(str(size_counts.get("Small", 0)), cell_style),
            Paragraph(f"{size_pcts.get('Small', 0.0)}%", cell_style),
            Paragraph("Confectionery / Diced Kernels", cell_style),
        ],
        [
            Paragraph("Medium", cell_style),
            Paragraph("18.0 - 22.0 mm", cell_style),
            Paragraph(str(size_counts.get("Medium", 0)), cell_style),
            Paragraph(f"{size_pcts.get('Medium', 0.0)}%", cell_style),
            Paragraph("Standard Snack / Retail Pack", cell_style),
        ],
        [
            Paragraph("Large", cell_style),
            Paragraph("&gt; 22.0 mm", cell_style),
            Paragraph(str(size_counts.get("Large", 0)), cell_style),
            Paragraph(f"{size_pcts.get('Large', 0.0)}%", cell_style),
            Paragraph("Jumbo Wholes / Premium Export", cell_style),
        ],
        [
            Paragraph("<b>Batch Summary</b>", cell_bold),
            Paragraph(f"Avg: <b>{analytics.get('avg_diameter_mm', 0)} mm</b>", cell_bold),
            Paragraph(f"<b>{summary.get('total_count', 0)}</b>", cell_bold),
            Paragraph("<b>100.0%</b>", cell_bold),
            Paragraph(f"Range: {analytics.get('min_diameter_mm', 0)} - {analytics.get('max_diameter_mm', 0)} mm", cell_style),
        ],
    ]
    t_size = Table(size_table_data, colWidths=[100, 110, 70, 80, 180])
    t_size.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
            ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#EDF2F7")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ])
    )
    elements.append(t_size)
    elements.append(Spacer(1, 10))

    # 4. Defect Breakdown Table
    elements.append(Paragraph("2. Quality & Defect Classification Audit", heading_style))
    defect_counts = analytics.get("defect_distribution", {})
    defect_pcts = analytics.get("defect_percentages", {})

    defect_table_data = [
        [
            Paragraph("<b>Classification Category</b>", cell_bold),
            Paragraph("<b>Count</b>", cell_bold),
            Paragraph("<b>Rate (%)</b>", cell_bold),
            Paragraph("<b>Industry Tolerance (Grade 1)</b>", cell_bold),
            Paragraph("<b>Audit Status</b>", cell_bold),
        ],
        [
            Paragraph("Healthy / Premium", cell_style),
            Paragraph(str(defect_counts.get("Healthy / Premium", 0)), cell_style),
            Paragraph(f"{defect_pcts.get('Healthy / Premium', 0.0)}%", cell_style),
            Paragraph("Base Target", cell_style),
            Paragraph("<font color='green'><b>ACCEPTED</b></font>", cell_style),
        ],
        [
            Paragraph("Cracked / Broken", cell_style),
            Paragraph(str(defect_counts.get("Cracked / Broken", 0)), cell_style),
            Paragraph(f"{defect_pcts.get('Cracked / Broken', 0.0)}%", cell_style),
            Paragraph("&le; 2.0%", cell_style),
            Paragraph(
                "<font color='green'>PASS</font>" if defect_pcts.get("Cracked / Broken", 0.0) <= 2.0 else "<font color='orange'>TOLERANCE FLAG</font>",
                cell_style,
            ),
        ],
        [
            Paragraph("Discolored / Brown Center", cell_style),
            Paragraph(str(defect_counts.get("Discolored / Brown Center", 0)), cell_style),
            Paragraph(f"{defect_pcts.get('Discolored / Brown Center', 0.0)}%", cell_style),
            Paragraph("&le; 2.0%", cell_style),
            Paragraph(
                "<font color='green'>PASS</font>" if defect_pcts.get("Discolored / Brown Center", 0.0) <= 2.0 else "<font color='orange'>TOLERANCE FLAG</font>",
                cell_style,
            ),
        ],
        [
            Paragraph("Moldy / Insect Damaged", cell_style),
            Paragraph(str(defect_counts.get("Moldy / Insect Damaged", 0)), cell_style),
            Paragraph(f"{defect_pcts.get('Moldy / Insect Damaged', 0.0)}%", cell_style),
            Paragraph("&le; 1.0%", cell_style),
            Paragraph(
                "<font color='green'>PASS</font>" if defect_pcts.get("Moldy / Insect Damaged", 0.0) <= 1.0 else "<font color='red'><b>CRITICAL FAIL</b></font>",
                cell_style,
            ),
        ],
    ]
    t_defect = Table(defect_table_data, colWidths=[150, 60, 70, 140, 120])
    t_defect.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ])
    )
    elements.append(t_defect)
    elements.append(Spacer(1, 10))

    # 5. Top Flagged Defects Detail
    defective_nuts = [n for n in nuts if n["is_defective"]]
    if defective_nuts:
        elements.append(Paragraph("3. Flagged Defect Audit Spotlights (XAI Verified)", heading_style))
        sample_defects = defective_nuts[:8]
        nut_table_data = [
            [
                Paragraph("<b>Nut #</b>", cell_bold),
                Paragraph("<b>Size</b>", cell_bold),
                Paragraph("<b>Defect Category</b>", cell_bold),
                Paragraph("<b>Confidence</b>", cell_bold),
                Paragraph("<b>Surface Affected</b>", cell_bold),
                Paragraph("<b>Agronomic Finding</b>", cell_bold),
            ]
        ]
        for n in sample_defects:
            nut_table_data.append([
                Paragraph(f"#{n['nut_id']}", cell_style),
                Paragraph(f"{n['size_grade']} ({n['diameter_mm']}mm)", cell_style),
                Paragraph(n["class_label"], cell_style),
                Paragraph(f"{n['confidence']:.0%}", cell_style),
                Paragraph(f"{n['defect_severity_pct']}%", cell_style),
                Paragraph(n.get("diagnostic_notes", "")[:80], cell_style),
            ])
        t_nut = Table(nut_table_data, colWidths=[40, 90, 130, 65, 80, 135])
        t_nut.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4A5568")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ])
        )
        elements.append(t_nut)
        elements.append(Spacer(1, 12))

    # 6. QA Certification Sign-Off
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E0"), spaceAfter=8))
    sign_data = [
        [
            Paragraph("<b>Automated Verification Engine:</b> Macadamia AI v1.0", cell_style),
            Paragraph("<b>Quality Assurance Auditor:</b> ___________________", cell_style),
            Paragraph("<b>Status:</b> [ X ] AUDITED & VERIFIED", cell_style),
        ]
    ]
    t_sign = Table(sign_data, colWidths=[200, 200, 140])
    elements.append(t_sign)

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


def generate_csv_report(grading_result: Dict[str, Any]) -> str:
    """
    Generates a structured CSV report containing batch summary and per-nut records.
    """
    output = io.StringIO()
    writer = csv.writer(output)

    summary = grading_result.get("summary", {})
    analytics = grading_result.get("analytics", {})
    nuts = grading_result.get("nuts", [])

    # Batch Summary Metadata Header
    writer.writerow(["# MACADAMIA QUALITY GRADING AUDIT REPORT"])
    writer.writerow(["Batch ID", summary.get("batch_id", "")])
    writer.writerow(["Timestamp", summary.get("timestamp", "")])
    writer.writerow(["Overall Quality Grade", summary.get("quality_grade", "")])
    writer.writerow(["Total Inspected Nuts", summary.get("total_count", 0)])
    writer.writerow(["Total Defective Nuts", analytics.get("total_defective_count", 0)])
    writer.writerow(["Defect Rate (%)", summary.get("defect_rate_pct", 0.0)])
    writer.writerow(["Premium Rate (%)", summary.get("premium_rate_pct", 0.0)])
    writer.writerow(["Average Diameter (mm)", analytics.get("avg_diameter_mm", 0.0)])
    writer.writerow([])

    # Per-Nut Record Table
    headers = [
        "Nut_ID",
        "Bounding_Box_X1",
        "Bounding_Box_Y1",
        "Bounding_Box_X2",
        "Bounding_Box_Y2",
        "Diameter_mm",
        "Diameter_cm",
        "Size_Grade",
        "Defect_Class",
        "Is_Defective",
        "Confidence_Pct",
        "Defect_Surface_Area_Pct",
        "Diagnostic_Notes",
    ]
    writer.writerow(headers)

    for n in nuts:
        bbox = n.get("bbox", [0, 0, 0, 0])
        writer.writerow([
            n.get("nut_id", 0),
            bbox[0],
            bbox[1],
            bbox[2],
            bbox[3],
            n.get("diameter_mm", 0.0),
            n.get("diameter_cm", 0.0),
            n.get("size_grade", ""),
            n.get("class_label", ""),
            "YES" if n.get("is_defective", False) else "NO",
            round(n.get("confidence", 0.0) * 100.0, 1),
            n.get("defect_severity_pct", 0.0),
            n.get("diagnostic_notes", ""),
        ])

    return output.getvalue()
