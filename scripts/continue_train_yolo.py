from ultralytics import YOLO
import glob
import os

def find_latest_model():
    """Find the most recently trained model."""
    # Look for best.pt and last.pt in runs/detect/train*/weights/
    best_paths = glob.glob('runs/detect/train*/weights/best.pt')
    last_paths = glob.glob('runs/detect/train*/weights/last.pt')
    
    if not best_paths and not last_paths:
        print("No trained models found. Please train a model first using train_yolo.py")
        return None
    
    # Get the most recent training run
    all_paths = best_paths + last_paths
    latest_path = max(all_paths, key=os.path.getmtime)
    
    # Prefer last.pt over best.pt for continuing training (has optimizer state)
    train_dir = os.path.dirname(latest_path)
    last_pt = os.path.join(train_dir, 'last.pt')
    best_pt = os.path.join(train_dir, 'best.pt')
    
    if os.path.exists(last_pt):
        return last_pt
    elif os.path.exists(best_pt):
        return best_pt
    else:
        return latest_path

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Continue training YOLO model from last checkpoint')
    parser.add_argument('--model', type=str, default=None, 
                       help='Path to model checkpoint (if None, uses latest)')
    parser.add_argument('--epochs', type=int, default=25, 
                       help='Number of additional epochs to train (default: 25)')
    parser.add_argument('--batch', type=int, default=16, 
                       help='Batch size (default: 16)')
    parser.add_argument('--lr', type=float, default=None, 
                       help='Learning rate (if None, uses model default)')
    parser.add_argument('--patience', type=int, default=10, 
                       help='Early stopping patience (default: 10)')
    parser.add_argument('--no_early_stop', action='store_true',
                       help='Disable early stopping (train for full epochs)')
    parser.add_argument('--save_period', type=int, default=5, 
                       help='Save checkpoint every N epochs (default: 5)')
    
    args = parser.parse_args()
    
    # Find model to continue from
    if args.model:
        model_path = args.model
        if not os.path.exists(model_path):
            print(f"Error: Model file '{model_path}' not found")
            exit(1)
    else:
        model_path = find_latest_model()
        if not model_path:
            exit(1)
    
    print(f"Continuing training from: {model_path}")
    print(f"Additional epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Learning rate: {args.lr if args.lr else 'model default'}")
    if args.no_early_stop:
        print("Early stopping: DISABLED")
    else:
        print(f"Patience: {args.patience}")
    
    # Load the trained model
    model = YOLO(model_path)
    
    # Prepare training arguments
    train_args = {
        'data': 'data.yaml',
        'epochs': args.epochs,
        'imgsz': 640,
        'batch': args.batch,
        'patience': 0 if args.no_early_stop else args.patience,  # Disable early stopping if requested
        'save_period': args.save_period,
        'device': 0,  # Force GPU device 0 (RTX 4090)
        'workers': 0,  # Disable multiprocessing to avoid Windows issues
        # Don't use resume=True, start fresh training with pretrained weights
    }
    
    # Add learning rate if specified
    if args.lr:
        train_args['lr0'] = args.lr
    
    print("\nStarting continued training...")
    print("Training arguments:", train_args)
    
    # Continue training
    results = model.train(**train_args)
    
    print("\nContinued training completed!")
    print(f"Results: {results}")
    
    # Print model locations
    print(f"\nNew models saved in: runs/detect/train*/weights/")
    print("- best.pt: Best performing model on validation")
    print("- last.pt: Latest checkpoint (for further continuation)")