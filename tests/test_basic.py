# test_basic.py - Just to verify your setup works
import cv2
from ultralytics import YOLO
import torch

print("🧪 Testing your setup...")

# 1. Check if MPS (Metal) is available on your M1
print(f"Metal Performance Shaders available: {torch.backends.mps.is_available()}")

# 2. Load YOLO model (this will download automatically)
print("Loading YOLO model...")
model = YOLO('yolov8s.pt')  # Start with small model for testing
print("✅ Model loaded!")

# 3. Test with your webcam
print("Testing webcam detection...")
cap = cv2.VideoCapture(0)

if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Run detection on single frame
        results = model(frame, classes=[0])  # Only detect persons
        
        # Draw results
        annotated_frame = results[0].plot()
        
        # Show the result
        cv2.imshow('Test Detection', annotated_frame)
        cv2.waitKey(3000)  # Show for 3 seconds
        cv2.destroyAllWindows()
        
        print("✅ Detection test successful!")
    else:
        print("❌ Could not capture from webcam")
else:
    print("❌ Could not open webcam")

cap.release()
print("🎉 Setup test complete!")