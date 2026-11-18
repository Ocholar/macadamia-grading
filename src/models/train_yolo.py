"""
YOLOv8 Training and Fine-Tuning Pipeline for Macadamia Nut Detection.
Supports dataset generation, training with Ultralytics YOLO, validation, and ONNX export.
"""

import argparse
import os
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_yolo_dataset_yaml(data_dir: str, output_yaml_path: str):
    """Generates the data.yaml configuration required by YOLOv8."""
    dataset_cfg = {
        "path": os.path.abspath(data_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "macadamia_nut",
        },
    }
    with open(output_yaml_path, "w") as f:
        yaml.dump(dataset_cfg, f, default_flow_style=False)
    logger.info("Created YOLO dataset YAML at: %s", output_yaml_path)


def train_yolo(
    data_yaml: str,
    epochs: int = 50,
    img_size: int = 640,
    batch_size: int = 16,
    weights: str = "yolov8n.pt",
    output_dir: str = "runs/detect",
    export_onnx: bool = True,
):
    """
    Trains or fine-tunes YOLOv8 detector for macadamia nuts.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error(
            "Ultralytics YOLO is not installed. To train, run: pip install ultralytics"
        )
        return

    logger.info("Initializing YOLO model with base weights: %s", weights)
    model = YOLO(weights)

    logger.info("Starting YOLO training for %d epochs...", epochs)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        project=output_dir,
        name="macadamia_detector",
        save=True,
        plots=True,
        augment=True,
        mosaic=1.0,
        mixup=0.1,
    )

    if export_onnx:
        logger.info("Exporting trained model to ONNX format...")
        onnx_path = model.export(format="onnx", imgsz=img_size, dynamic=False)
        logger.info("Successfully exported ONNX model to: %s", onnx_path)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Macadamia Dataset")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--img_size", type=int, default=640, help="Input image dimension")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Initial weights")
    args = parser.parse_args()

    yaml_path = os.path.join(args.data_dir, "macadamia_yolo.yaml")
    create_yolo_dataset_yaml(args.data_dir, yaml_path)
    train_yolo(yaml_path, epochs=args.epochs, img_size=args.img_size, batch_size=args.batch_size)
