"""
Object Detection with Deep Learning YOLO Model
"""

import cv2
import torch
from yolov5 import YOLOv5
from PIL import Image
import numpy as np
import json
import os
import pandas as pd
from ultralytics import YOLO
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
# Load YOLOv8 Model
model = YOLO("yolov8x.pt")


def process_vid(video_path):
    # infer the filename
    filename = video_path.split("/")[-1]

    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Prepare metadata storage
    metadata = []
    progress_interval = max(1, frame_count // 10)  # Report progress 10 times
    
    # Process the video frame by frame
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert the frame to PIL image
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
        # Perform inference
        results = model(img, verbose=False)
    
        # Get bounding boxes and labels
        if results:
            boxes = results[0].boxes
            
            # Extract necessary attributes
            xyxy = boxes.xyxy.cpu().numpy()  # Bounding box coordinates
            conf = boxes.conf.cpu().numpy()  # Confidence scores
            cls = boxes.cls.cpu().numpy()    # Class labels
    
            # Save metadata if any objects are detected
            if len(xyxy) > 0:
                frame_metadata = {
                    'frame_number': frame_number,
                    'objects': []
                }
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i]
                    obj_metadata = {
                        'label': model.names[int(cls[i])],
                        'confidence': float(conf[i]),
                        'bounding_box': [int(x1), int(y1), int(x2), int(y2)]
                    }
                    frame_metadata['objects'].append(obj_metadata)
                metadata.append(frame_metadata)
    
        # Report progress
        if frame_number % progress_interval == 0:
            print(f"Processed {frame_number}/{frame_count} frames ({frame_number / frame_count:.2%}) Objects found: {len(metadata)}")
        
        frame_number += 1
    
    # Release everything if job is finished
    cap.release()
    cv2.destroyAllWindows()
        
    # Save metadata to a JSON file
    output_metadata_path = f"output_metadata/output_metadata_{filename}.json"
    with open(output_metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"{len(metadata)} objects detected. Metadata saved to {output_metadata_path}.")

def process_videos_in_directory(directory_path):
    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.mp4'):
            print(f"{pd.to_datetime('now', utc=True)} Start processing {filename} now")
            video_path = os.path.join(directory_path, filename)
            process_vid(video_path)
            print(f"{pd.to_datetime('now', utc=True)} Finished with {filename}")

if __name__ == "__main__":
    directory_path = "path/to/video"  # Replace with your directory path
    process_videos_in_directory(directory_path)
