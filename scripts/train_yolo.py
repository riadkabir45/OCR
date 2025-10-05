from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(
        data='data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,  # Increased batch size for GPU
        patience=10,
        save_period=5,
        device=0,  # Force GPU device 0 (RTX 4090)
        workers=0  # Disable multiprocessing to avoid Windows issues
    )

    print("Training completed!")
    print(f"Results: {results}")