import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import List, Dict, Tuple

class ExamDetector:
    """
    Person detection system optimized for classroom exam monitoring.
    Uses YOLOv8 with Metal Performance Shaders acceleration on M1.
    """
    
    def __init__(self, model_size: str = 'yolov8m.pt', confidence_threshold: float = 0.6):
        """
        Initialize detector with specified YOLO model.
        
        Args:
            model_size: YOLO model variant (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
            confidence_threshold: Minimum confidence for valid detections
        """
        self.confidence_threshold = confidence_threshold
        self.model_size = model_size
        
        print(f"🤖 Initializing ExamDetector with {model_size}")
        
        # Load YOLO model
        self.model = YOLO(model_size)
        
        # Configure M1 acceleration
        if torch.backends.mps.is_available():
            print("⚡ Using Metal Performance Shaders acceleration")
            self.device = 'mps'
        else:
            print("📱 Using CPU")
            self.device = 'cpu'
        
        print(f"✅ ExamDetector ready (confidence threshold: {confidence_threshold})")
    
    def detect_people(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect all people in the given frame.
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            List of detected people with bounding boxes, centers, and metadata
        """
        # Run YOLO inference - only detect persons (class 0)
        results = self.model(
            frame,
            classes=[0],  # Person class only
            conf=self.confidence_threshold,
            verbose=False,
            device=self.device
        )
        
        people = []
        
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes
                
                for box in boxes:
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    # Calculate derived properties
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    
                    # Estimate person orientation based on bounding box shape
                    # Wider boxes suggest person is turned sideways
                    aspect_ratio = width / height if height > 0 else 0
                    
                    person_data = {
                        'bbox': [x1, y1, x2, y2],
                        'center': [center_x, center_y],
                        'confidence': confidence,
                        'width': width,
                        'height': height,
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'frame_timestamp': cv2.getTickCount()  # For timing analysis
                    }
                    
                    people.append(person_data)
        
        return people
    
    def draw_detections(self, frame: np.ndarray, people: List[Dict], 
                       show_details: bool = True) -> np.ndarray:
        """
        Draw detection results on frame with detailed annotations.
        
        Args:
            frame: Input frame
            people: List of detected people
            show_details: Whether to show detailed info
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Color scheme for different confidence levels
        def get_color(confidence):
            if confidence > 0.8:
                return (0, 255, 0)  # High confidence - green
            elif confidence > 0.6:
                return (0, 255, 255)  # Medium confidence - yellow
            else:
                return (0, 165, 255)  # Low confidence - orange
        
        for i, person in enumerate(people):
            x1, y1, x2, y2 = person['bbox']
            confidence = person['confidence']
            center = person['center']
            color = get_color(confidence)
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw center point
            cv2.circle(annotated_frame, tuple(center), 4, color, -1)
            
            # Draw person ID number
            cv2.putText(annotated_frame, str(i+1), (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            if show_details:
                # Draw confidence and dimensions
                info_text = f"ID:{i+1} Conf:{confidence:.2f}"
                cv2.putText(annotated_frame, info_text, (x1, y2+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Draw aspect ratio (useful for detecting turned heads/bodies)
                aspect_text = f"AR:{person['aspect_ratio']:.2f}"
                cv2.putText(annotated_frame, aspect_text, (x1, y2+35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw summary statistics
        summary_color = (255, 255, 255)  # White
        cv2.putText(annotated_frame, f"People Detected: {len(people)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, summary_color, 2)
        
        # Draw frame info
        frame_height, frame_width = frame.shape[:2]
        cv2.putText(annotated_frame, f"Frame: {frame_width}x{frame_height}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, summary_color, 1)
        
        return annotated_frame
    
    def get_detection_stats(self, people: List[Dict]) -> Dict:
        """
        Calculate detection statistics for analysis.
        
        Args:
            people: List of detected people
            
        Returns:
            Dictionary with detection statistics
        """
        if not people:
            return {
                'count': 0,
                'avg_confidence': 0,
                'avg_area': 0,
                'avg_aspect_ratio': 0
            }
        
        confidences = [p['confidence'] for p in people]
        areas = [p['area'] for p in people]
        aspect_ratios = [p['aspect_ratio'] for p in people]
        
        return {
            'count': len(people),
            'avg_confidence': np.mean(confidences),
            'max_confidence': np.max(confidences),
            'min_confidence': np.min(confidences),
            'avg_area': np.mean(areas),
            'total_area': np.sum(areas),
            'avg_aspect_ratio': np.mean(aspect_ratios),
            'detection_density': len(people) / (1920 * 1080)  # Normalized for 1080p
        }
    
    def filter_detections(self, people: List[Dict], 
                         min_area: int = 1000, 
                         max_aspect_ratio: float = 2.0) -> List[Dict]:
        """
        Filter detections to remove noise and invalid detections.
        
        Args:
            people: Raw detections
            min_area: Minimum bounding box area
            max_aspect_ratio: Maximum width/height ratio
            
        Returns:
            Filtered list of valid detections
        """
        filtered_people = []
        
        for person in people:
            # Filter by area (remove tiny detections)
            if person['area'] < min_area:
                continue
                
            # Filter by aspect ratio (remove unrealistic shapes)
            if person['aspect_ratio'] > max_aspect_ratio:
                continue
                
            # Additional filtering can be added here
            filtered_people.append(person)
        
        return filtered_people