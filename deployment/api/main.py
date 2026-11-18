"""
FastAPI Application for Macadamia Quality Grading AI System.
Provides RESTful APIs for batch inspection, metric sizing, defect detection,
XAI heatmaps, and downloadable PDF/CSV reports.
"""

import os
import sys
import logging
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, File, UploadFile, Query, Form, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline import MacadamiaGradingPipeline
from src.ingestion.handler import ImageIngestionHandler
from src.reporting.report_generator import generate_pdf_report, generate_csv_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("macadamia_api")

app = FastAPI(
    title="Macadamia Quality Grading AI API",
    version="1.0.0",
    description="Automated optical grading, metric sizing, defect classification, and XAI auditing for macadamia nuts.",
)

# Enable CORS for frontend applications (React, Flutter, Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global singleton pipeline instance
pipeline: Optional[MacadamiaGradingPipeline] = None


@app.on_event("startup")
def startup_event():
    """Initializes the AI grading pipeline on application startup."""
    global pipeline
    logger.info("Initializing Macadamia Grading Pipeline...")
    config_path = os.path.join(PROJECT_ROOT, "config.yaml")
    pipeline = MacadamiaGradingPipeline(config_path=config_path)
    logger.info("Macadamia Grading Pipeline successfully initialized.")


# Pydantic Request & Response Models
class Base64GradeRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded JPEG or PNG image")
    pixels_per_mm: Optional[float] = Field(None, description="Optional manual calibration factor (pixels/mm)")
    mat_type: Optional[str] = Field("auto", description="Inspection mat background: 'auto', 'blue_mat', 'green_mat', 'dark_mat', 'light_mat'")


class BatchSummaryResponse(BaseModel):
    batch_id: str
    timestamp: str
    total_count: int
    quality_grade: str
    defect_rate_pct: float
    premium_rate_pct: float
    avg_diameter_mm: float
    processing_time_sec: float


class NutGradeResponse(BaseModel):
    nut_id: int
    bbox: List[int]
    diameter_mm: float
    diameter_cm: float
    size_grade: str
    class_id: int
    class_label: str
    confidence: float
    is_defective: bool
    defect_severity_pct: float
    diagnostic_notes: str


class GradeReportResponse(BaseModel):
    summary: BatchSummaryResponse
    analytics: Dict[str, Any]
    nuts: List[NutGradeResponse]
    annotated_image_base64: str
    defective_count: int


@app.get("/api/v1/health")
def health_check():
    """Returns service health and model status."""
    return {
        "status": "online",
        "service": "Macadamia Quality Grading AI",
        "version": "1.0.0",
        "detector_loaded": pipeline.detector is not None if pipeline else False,
        "classifier_loaded": pipeline.classifier is not None if pipeline else False,
        "calibrator_pixels_per_mm": pipeline.calibrator.pixels_per_mm if pipeline else 6.5,
    }


@app.get("/api/v1/config")
def get_configuration():
    """Retrieves active system configuration and commercial standards."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return pipeline.config


@app.post("/api/v1/grade", response_model=GradeReportResponse)
async def grade_image_upload(
    file: UploadFile = File(..., description="Image file (JPG, PNG)"),
    pixels_per_mm: Optional[float] = Query(None, description="Optional pixel-to-mm ratio"),
    mat_type: Optional[str] = Query("auto", description="Mat type: 'auto', 'blue_mat', 'green_mat', 'dark_mat', 'light_mat'"),
):
    """
    Analyzes an uploaded batch photo of macadamias.
    Returns counts, size distributions, defect classifications, and annotated image.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        result = pipeline.process_image(
            image_bytes,
            pixels_per_mm_override=pixels_per_mm,
            mat_type_override=mat_type,
        )

        annotated_b64 = ImageIngestionHandler.to_base64(result["images"]["annotated"])

        serialized_nuts = [
            NutGradeResponse(
                nut_id=n["nut_id"],
                bbox=n["bbox"],
                diameter_mm=n["diameter_mm"],
                diameter_cm=n["diameter_cm"],
                size_grade=n["size_grade"],
                class_id=n["class_id"],
                class_label=n["class_label"],
                confidence=n["confidence"],
                is_defective=n["is_defective"],
                defect_severity_pct=n["defect_severity_pct"],
                diagnostic_notes=n["diagnostic_notes"],
            )
            for n in result["nuts"]
        ]

        return GradeReportResponse(
            summary=BatchSummaryResponse(**result["summary"]),
            analytics=result["analytics"],
            nuts=serialized_nuts,
            annotated_image_base64=annotated_b64,
            defective_count=result["analytics"]["total_defective_count"],
        )
    except Exception as e:
        logger.exception("Error processing batch grading request: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/grade/base64", response_model=GradeReportResponse)
def grade_image_base64(request: Base64GradeRequest):
    """Analyzes a base64 encoded photo (ideal for mobile and embedded devices)."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        ingestion = ImageIngestionHandler()
        image_bgr = ingestion.load_from_base64(request.image_base64)

        result = pipeline.process_image(
            image_bgr,
            pixels_per_mm_override=request.pixels_per_mm,
            mat_type_override=request.mat_type,
        )

        annotated_b64 = ImageIngestionHandler.to_base64(result["images"]["annotated"])

        serialized_nuts = [
            NutGradeResponse(
                nut_id=n["nut_id"],
                bbox=n["bbox"],
                diameter_mm=n["diameter_mm"],
                diameter_cm=n["diameter_cm"],
                size_grade=n["size_grade"],
                class_id=n["class_id"],
                class_label=n["class_label"],
                confidence=n["confidence"],
                is_defective=n["is_defective"],
                defect_severity_pct=n["defect_severity_pct"],
                diagnostic_notes=n["diagnostic_notes"],
            )
            for n in result["nuts"]
        ]

        return GradeReportResponse(
            summary=BatchSummaryResponse(**result["summary"]),
            analytics=result["analytics"],
            nuts=serialized_nuts,
            annotated_image_base64=annotated_b64,
            defective_count=result["analytics"]["total_defective_count"],
        )
    except Exception as e:
        logger.exception("Error processing base64 image: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/report/pdf")
async def export_pdf_report(
    file: UploadFile = File(..., description="Batch photo"),
    pixels_per_mm: Optional[float] = Query(None),
    mat_type: Optional[str] = Query("auto"),
):
    """Generates and downloads an official PDF Certificate of Quality Grading."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    image_bytes = await file.read()
    result = pipeline.process_image(
        image_bytes,
        pixels_per_mm_override=pixels_per_mm,
        mat_type_override=mat_type,
    )

    pdf_bytes = generate_pdf_report(result)
    batch_id = result["summary"].get("batch_id", "batch")
    filename = f"macadamia_quality_certificate_{batch_id}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.post("/api/v1/report/csv")
async def export_csv_report(
    file: UploadFile = File(..., description="Batch photo"),
    pixels_per_mm: Optional[float] = Query(None),
    mat_type: Optional[str] = Query("auto"),
):
    """Generates and downloads a detailed CSV data export."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    image_bytes = await file.read()
    result = pipeline.process_image(
        image_bytes,
        pixels_per_mm_override=pixels_per_mm,
        mat_type_override=mat_type,
    )

    csv_text = generate_csv_report(result)
    batch_id = result["summary"].get("batch_id", "batch")
    filename = f"macadamia_quality_report_{batch_id}.csv"

    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
