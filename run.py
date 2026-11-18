#!/usr/bin/env python3
"""
run.py — Local development launcher (no Docker required).
Starts FastAPI backend and Streamlit dashboard in parallel.
Usage:
    python run.py           # Start both API + UI
    python run.py --api     # API only
    python run.py --ui      # UI only
    python run.py --test    # Run test suite
    python run.py --demo    # Process all sample batches and save results
"""

import argparse
import os
import subprocess
import sys
import time
import threading

PROJECT_ROOT = os.path.dirname(__file__)
sys.path.insert(0, PROJECT_ROOT)


def banner():
    print("""
╔══════════════════════════════════════════════════════╗
║       🌰 Macadamia Quality Grading AI  v1.0          ║
╠══════════════════════════════════════════════════════╣
║  FastAPI REST:  http://localhost:8000                 ║
║  Swagger Docs:  http://localhost:8000/docs            ║
║  Streamlit UI:  http://localhost:8501                 ║
╚══════════════════════════════════════════════════════╝
""")


def start_api():
    """Launches the FastAPI backend server."""
    print("[API] Starting FastAPI on :8000 ...")
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "deployment.api.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload",
        "--log-level", "info",
    ], cwd=PROJECT_ROOT)


def start_ui():
    """Launches the Streamlit dashboard."""
    print("[UI] Starting Streamlit on :8501 ...")
    time.sleep(1.5)  # Let API warm up first
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        os.path.join(PROJECT_ROOT, "deployment", "ui", "app.py"),
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
        "--server.headless", "false",
        "--browser.gatherUsageStats", "false",
        "--theme.base", "dark",
        "--theme.primaryColor", "#58A6FF",
        "--theme.backgroundColor", "#0F1117",
        "--theme.secondaryBackgroundColor", "#161B22",
        "--theme.textColor", "#E8ECF0",
    ], cwd=PROJECT_ROOT)


def run_tests():
    """Executes the pytest test suite."""
    print("[TEST] Running full test suite ...")
    result = subprocess.run([
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "--no-header",
    ], cwd=PROJECT_ROOT)
    return result.returncode


def run_demo():
    """Processes all sample batch images and saves annotated output + PDF/CSV reports."""
    import cv2
    from src.pipeline import MacadamiaGradingPipeline
    from src.reporting.report_generator import generate_pdf_report, generate_csv_report

    print("[DEMO] Running batch inference on all sample images ...")
    pipeline = MacadamiaGradingPipeline(
        config_path=os.path.join(PROJECT_ROOT, "config.yaml")
    )
    raw_dir     = os.path.join(PROJECT_ROOT, "data", "raw")
    output_dir  = os.path.join(PROJECT_ROOT, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)

    for fname in sorted(os.listdir(raw_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(raw_dir, fname)
        stem = os.path.splitext(fname)[0]
        print(f"\n  📷 Processing: {fname}")

        t0 = time.time()
        result = pipeline.process_image(img_path)
        elapsed = round(time.time() - t0, 3)

        # Save annotated image
        ann_path = os.path.join(output_dir, f"{stem}_annotated.jpg")
        cv2.imwrite(ann_path, result["images"]["annotated"])

        # Save PDF report
        pdf_path = os.path.join(output_dir, f"{stem}_report.pdf")
        with open(pdf_path, "wb") as f:
            f.write(generate_pdf_report(result))

        # Save CSV report
        csv_path = os.path.join(output_dir, f"{stem}_report.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(generate_csv_report(result))

        s = result["summary"]
        print(f"     ✅ {s['total_count']} nuts | {s['quality_grade']} | "
              f"Defects: {s['defect_rate_pct']}% | {elapsed}s")
        print(f"        → {ann_path}")
        print(f"        → {pdf_path}")
        print(f"        → {csv_path}")

    print("\n[DEMO] Done. All outputs saved to data/processed/")


def main():
    parser = argparse.ArgumentParser(description="Macadamia Quality Grading AI Launcher")
    parser.add_argument("--api",  action="store_true", help="Start FastAPI backend only")
    parser.add_argument("--ui",   action="store_true", help="Start Streamlit UI only")
    parser.add_argument("--test", action="store_true", help="Run test suite")
    parser.add_argument("--demo", action="store_true", help="Run demo inference on sample images")
    args = parser.parse_args()

    banner()

    if args.test:
        sys.exit(run_tests())

    if args.demo:
        run_demo()
        return

    if args.api and not args.ui:
        start_api()
    elif args.ui and not args.api:
        start_ui()
    else:
        # Start both: API in a background thread, UI in foreground
        api_thread = threading.Thread(target=start_api, daemon=True)
        api_thread.start()
        start_ui()


if __name__ == "__main__":
    main()
