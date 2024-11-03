import cv2
import numpy as np
import torch

# Load the YOLO model (YOLOv5 for example)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # Use 'yolov5s' or replace with custom model

# Set the model to detect only "cone" (based on the classes file; customize if needed)
# YOLOv5 supports COCO classes, modify if you need a custom dataset.

# Start video capture from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB (YOLO requires RGB format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run inference
    results = model(rgb_frame)

    # Render results on the frame
    results.render()

    # Display the frame with detected objects
    cv2.imshow('Cone Detection', np.squeeze(results.ims))

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()

