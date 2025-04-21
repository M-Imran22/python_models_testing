# train.py
from ultralytics import YOLO


def main():
    # 1. Load the YOLOv8n “nano” model (pretrained on ImageNet for classification)
    model = YOLO('yolov8s-cls.pt')

    # 2. Train the model on your dataset
    # Specify the dataset root directory directly
    dataset_root = r'D:\Code\all_code\Projects\model_testing\dataset'

    # Train the model
    results = model.train(
        data=dataset_root,  # Path to the dataset root directory
        epochs=20,
        batch=32,
        imgsz=224,
        project='runs',
        name='exp1',
        device="cpu"  # Use CPU as specified in your output
    )

    # 3. After training, you can also evaluate on your val split
    metrics = model.val(
        model=f"runs/exp1/weights/best.pt",
        data=dataset_root,
        batch=32,
        imgsz=224
    )
    print(metrics)

    # 4. And run predictions on new images:
    preds = model.predict(
        source='runs/exp1/',
        # save classification results (confidences) into runs/predict/
        save=True,
        device='cpu',
        imgsz=224
    )
    print(f"Saved predictions to: runs/predict/{preds[0].orig_img.shape}")


if __name__ == "__main__":
    main()
