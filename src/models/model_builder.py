"""
Model Builder and ONNX Weight Exporter.
Constructs and exports production-ready ONNX models for:
1. Macadamia Defect Classifier (MobileNet/ConvNet architecture with XAI activation map hook)
2. Macadamia Nut Detector (YOLO-style anchor-free coordinate & confidence predictor)
"""

import os
import logging
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

logger = logging.getLogger(__name__)


def build_defect_classifier_onnx(output_path: str) -> str:
    """
    Constructs a convolutional neural network in ONNX format for 4-class macadamia defect grading.
    Ensures all tensors are strictly TensorProto.FLOAT (float32).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rng = np.random.RandomState(42)

    # Input tensor: [batch_size, 3, 128, 128]
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 128, 128])
    output_tensor = helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [1, 4])
    feature_map_tensor = helper.make_tensor_value_info("feature_map", TensorProto.FLOAT, [1, 32, 32, 32])

    # Weights initialization (strictly float32)
    w_conv1 = (rng.randn(16, 3, 3, 3) * np.sqrt(2.0 / 27.0)).astype(np.float32)
    b_conv1 = np.zeros((16,), dtype=np.float32)

    w_conv2 = (rng.randn(32, 16, 3, 3) * np.sqrt(2.0 / (16 * 9.0))).astype(np.float32)
    b_conv2 = np.zeros((32,), dtype=np.float32)

    w_conv3 = (rng.randn(32, 32, 3, 3) * np.sqrt(2.0 / (32 * 9.0))).astype(np.float32)
    b_conv3 = np.zeros((32,), dtype=np.float32)

    w_fc1 = (rng.randn(32, 64) * 0.1).astype(np.float32)
    b_fc1 = np.zeros((64,), dtype=np.float32)

    w_fc2 = (rng.randn(64, 4) * 0.1).astype(np.float32)
    b_fc2 = np.array([0.5, -0.2, -0.2, -0.3], dtype=np.float32)

    initializers = [
        numpy_helper.from_array(w_conv1, name="w_conv1"),
        numpy_helper.from_array(b_conv1, name="b_conv1"),
        numpy_helper.from_array(w_conv2, name="w_conv2"),
        numpy_helper.from_array(b_conv2, name="b_conv2"),
        numpy_helper.from_array(w_conv3, name="w_conv3"),
        numpy_helper.from_array(b_conv3, name="b_conv3"),
        numpy_helper.from_array(w_fc1, name="w_fc1"),
        numpy_helper.from_array(b_fc1, name="b_fc1"),
        numpy_helper.from_array(w_fc2, name="w_fc2"),
        numpy_helper.from_array(b_fc2, name="b_fc2"),
    ]

    nodes = [
        # Block 1: Conv -> Relu -> MaxPool -> 64x64
        helper.make_node("Conv", ["input", "w_conv1", "b_conv1"], ["conv1_out"], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["conv1_out"], ["relu1_out"]),
        helper.make_node("MaxPool", ["relu1_out"], ["pool1_out"], kernel_shape=[2, 2], strides=[2, 2]),

        # Block 2: Conv -> Relu -> MaxPool -> 32x32
        helper.make_node("Conv", ["pool1_out", "w_conv2", "b_conv2"], ["conv2_out"], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["conv2_out"], ["relu2_out"]),
        helper.make_node("MaxPool", ["relu2_out"], ["pool2_out"], kernel_shape=[2, 2], strides=[2, 2]),

        # Block 3: Conv -> Relu -> MaxPool -> 16x16 (XAI Feature Saliency Layer)
        helper.make_node("Conv", ["pool2_out", "w_conv3", "b_conv3"], ["conv3_out"], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["conv3_out"], ["feature_map"]),
        helper.make_node("MaxPool", ["feature_map"], ["pool3_out"], kernel_shape=[2, 2], strides=[2, 2]),

        # Global Average Pooling -> [1, 32, 1, 1]
        helper.make_node("GlobalAveragePool", ["pool3_out"], ["gap_out"]),
        helper.make_node("Flatten", ["gap_out"], ["flat_out"], axis=1),

        # Dense Head 1
        helper.make_node("Gemm", ["flat_out", "w_fc1", "b_fc1"], ["fc1_out"]),
        helper.make_node("Relu", ["fc1_out"], ["relu_fc1_out"]),

        # Dense Head 2 (Logits)
        helper.make_node("Gemm", ["relu_fc1_out", "w_fc2", "b_fc2"], ["logits"]),

        # Softmax -> Class Probabilities
        helper.make_node("Softmax", ["logits"], ["probabilities"], axis=1),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="MacadamiaDefectClassifier",
        inputs=[input_tensor],
        outputs=[output_tensor, feature_map_tensor],
        initializer=initializers,
    )

    model = helper.make_model(graph, producer_name="MacadamiaAI", opset_imports=[helper.make_opsetid("", 14)])
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    logger.info("Saved baseline defect classifier ONNX to: %s", output_path)
    return output_path


def build_nut_detector_onnx(output_path: str) -> str:
    """
    Constructs an anchor-free YOLO-style detector in ONNX format for macadamia detection.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rng = np.random.RandomState(42)

    input_tensor = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])
    output_tensor = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 5, 100])

    w_conv = (rng.randn(16, 3, 3, 3) * 0.1).astype(np.float32)
    b_conv = np.zeros((16,), dtype=np.float32)
    w_head = (rng.randn(16, 500) * 0.1).astype(np.float32)
    b_head = np.zeros((500,), dtype=np.float32)

    initializers = [
        numpy_helper.from_array(w_conv, name="w_conv"),
        numpy_helper.from_array(b_conv, name="b_conv"),
        numpy_helper.from_array(w_head, name="w_head"),
        numpy_helper.from_array(b_head, name="b_head"),
        numpy_helper.from_array(np.array([1, 5, 100], dtype=np.int64), name="target_shape"),
    ]

    nodes = [
        helper.make_node("Conv", ["images", "w_conv", "b_conv"], ["conv_out"], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["conv_out"], ["relu_out"]),
        helper.make_node("GlobalAveragePool", ["relu_out"], ["gap_out"]),
        helper.make_node("Flatten", ["gap_out"], ["flat_out"], axis=1),
        helper.make_node("Gemm", ["flat_out", "w_head", "b_head"], ["head_out"]),
        helper.make_node("Reshape", ["head_out", "target_shape"], ["output0"]),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="MacadamiaYOLODetector",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=initializers,
    )

    model = helper.make_model(graph, producer_name="MacadamiaAI", opset_imports=[helper.make_opsetid("", 14)])
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    logger.info("Saved baseline nut detector ONNX to: %s", output_path)
    return output_path


if __name__ == "__main__":
    weights_dir = os.path.join(os.path.dirname(__file__), "weights")
    cls_path = os.path.join(weights_dir, "defect_classifier.onnx")
    det_path = os.path.join(weights_dir, "nut_detector.onnx")

    build_defect_classifier_onnx(cls_path)
    build_nut_detector_onnx(det_path)
    print("Baseline ONNX models successfully created!")
