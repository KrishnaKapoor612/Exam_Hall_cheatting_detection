import cv2
import time
from ultralytics import YOLO
from config import INPUT_SOURCE
import statistics

def benchmark_model(model_name):
    print(f"\n Benchmarking {model_name} ... (running 100 frames)")
    model = YOLO(model_name)
    
    cap = cv2.VideoCapture(INPUT_SOURCE)
    fps_list = []
    inference_list = []
    frame_count = 0
    
    # Warmup
    for _ in range(10):
        ret, frame = cap.read()
        if ret:
            model.track(frame, imgsz=320, persist=True, verbose=False)
    
    start_total = time.time()
    
    while frame_count < 100:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        
        t0 = time.time()
        results = model.track(frame, imgsz=320, persist=True, verbose=False)
        t1 = time.time()
        
        inference_list.append(t1 - t0)
        fps_list.append(1.0 / (t1 - t0))
    
    cap.release()
    
    avg_fps = statistics.mean(fps_list)
    avg_inference_ms = statistics.mean(inference_list) * 1000
    total_time = time.time() - start_total
    
    print(f"✅ {model_name} Results:")
    print(f"   Average FPS          : {avg_fps:.2f}")
    print(f"   Average Inference    : {avg_inference_ms:.1f} ms")
    print(f"   Latency per frame    : {avg_inference_ms:.1f} ms")
    print(f"   Total time for 100 frames : {total_time:.2f} seconds")
    
    return {
        "model": model_name,
        "fps": round(avg_fps, 2),
        "inference_ms": round(avg_inference_ms, 1),
        "latency_ms": round(avg_inference_ms, 1)
    }

# ================== RUN COMPARISON ==================
print("🚀 Starting YOLOv8 vs YOLO11 Comparison (CPU)")

yolov8_results = benchmark_model("yolov8n-pose.pt")
yolo11_results = benchmark_model("yolo11n-pose.pt")

print("\n" + "="*60)
print("FINAL COMPARISON TABLE")
print("="*60)
print(f"{'Metric':<20} {'YOLOv8n-pose':<18} {'YOLO11n-pose':<18} {'Winner'}")
print("-"*60)
print(f"{'FPS':<20} {yolov8_results['fps']:<18} {yolo11_results['fps']:<18} {'YOLO11' if yolo11_results['fps'] > yolov8_results['fps'] else 'YOLOv8'}")
print(f"{'Inference Time (ms)':<20} {yolov8_results['inference_ms']:<18} {yolo11_results['inference_ms']:<18} {'YOLO11' if yolo11_results['inference_ms'] < yolov8_results['inference_ms'] else 'YOLOv8'}")
print(f"{'Latency (ms)':<20} {yolov8_results['latency_ms']:<18} {yolo11_results['latency_ms']:<18} {'YOLO11' if yolo11_results['latency_ms'] < yolov8_results['latency_ms'] else 'YOLOv8'}")
print("="*60)