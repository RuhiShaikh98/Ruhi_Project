from ultralytics import YOLO
import os

def run_inference(model_path, source, output_dir):
    model = YOLO(model_path)
    results = model.predict(source=source, save=True, project=output_dir)
    return results

if __name__ == "__main__":
    model_path = "/home/ruhi_musa/models/yolov8_model.pt"
    source = "/home/ruhi_musa/DATASET/TEST"  # Replace with a specific image or video file for inference
    if not os.path.exists(source):
        raise FileNotFoundError(f"Source path does not exist: {source}")
    output_dir = "/home/ruhi_musa/inference_results"
    run_inference(model_path, source, output_dir)
