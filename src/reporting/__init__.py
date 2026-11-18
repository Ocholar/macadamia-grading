"""
Reporting Module for Macadamia Quality Grading System.
Generates downloadable PDF audit certificates and CSV reports.
"""

from .report_generator import generate_pdf_report, generate_csv_report

__all__ = ["generate_pdf_report", "generate_csv_report"]
