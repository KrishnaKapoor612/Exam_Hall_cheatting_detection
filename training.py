from ultralytics import YOLO
import matplotlib.pyplot as plt
import os

def train_model():
    print("Starting YOLOv8 Pose Training...")
    
    model = YOLO("yolov8n-pose.pt")  # start from official pretrained pose model
    
    # Train on your custom dataset (create data.yaml first)
    results = model.train(
        data="data.yaml",      # ← change to your dataset path
        epochs=50,
        imgsz=640,
        batch=16,
        name="exam_hall_pose",
        patience=10,
        optimizer="AdamW"
    )
    
    # Generate training graphs
    os.makedirs("graphs", exist_ok=True)
    
    # Loss vs Epoch
    plt.figure(figsize=(10, 5))
    plt.plot(results.results_dict['train/box_loss'], label='Box Loss')
    plt.plot(results.results_dict['train/pose_loss'], label='Pose Loss')
    plt.title('Training Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("graphs/loss.png")
    plt.close()
    
    # Accuracy (Pose AP) vs Epoch
    plt.figure(figsize=(10, 5))
    plt.plot(results.results_dict['metrics/mAP50-95(P)'], label='Pose mAP50-95')
    plt.title('Pose Accuracy vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.legend()
    plt.savefig("graphs/accuracy.png")
    plt.close()
    
    print("✅ Training completed! Graphs saved in /graphs/ folder")

if __name__ == "__main__":
    train_model()