import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
import time

class PersonTracker:
    """
    Multi-person tracking system using centroid-based tracking with Kalman filtering.
    Optimized for classroom exam monitoring scenarios.
    """
    
    def __init__(self, max_disappeared: int = 50, max_distance: float = 200):
        """
        Initialize the person tracker.
        
        Args:
            max_disappeared: Maximum frames a person can disappear before deletion
            max_distance: Maximum distance for person association (pixels)
        """
        self.next_id = 0
        self.tracks = {}  # Active tracks
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
        # Performance tracking
        self.frame_count = 0
        self.total_tracks_created = 0
        
        print(f"🎯 PersonTracker initialized (max_distance: {max_distance}, max_disappeared: {max_disappeared})")
        
        # Improved association parameters
        self.area_weight = 0.3  # Weight for area similarity in association
        self.confidence_weight = 0.2  # Weight for confidence similarity
    
    def update(self, detections: List[Dict]) -> Dict[int, Dict]:
        """
        Update tracker with new detections from current frame.
        
        Args:
            detections: List of person detections from detector
            
        Returns:
            Dictionary of active tracks with track_id as key
        """
        self.frame_count += 1
        current_time = time.time()
        
        # Handle case with no detections
        if len(detections) == 0:
            return self._handle_no_detections()
        
        # Handle case with no existing tracks
        if len(self.tracks) == 0:
            return self._initialize_tracks(detections, current_time)
        
        # Associate detections with existing tracks
        self._associate_detections(detections, current_time)
        
        # Clean up disappeared tracks
        self._cleanup_disappeared_tracks()
        
        return self.tracks.copy()
    
    def _handle_no_detections(self) -> Dict[int, Dict]:
        """Handle frame with no detections - increment disappeared counter."""
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id]['disappeared'] += 1
            if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                del self.tracks[track_id]
        
        return self.tracks.copy()
    
    def _initialize_tracks(self, detections: List[Dict], current_time: float) -> Dict[int, Dict]:
        """Initialize new tracks when no existing tracks exist."""
        for detection in detections:
            self._create_new_track(detection, current_time)
        
        return self.tracks.copy()
    
    def _create_new_track(self, detection: Dict, current_time: float) -> int:
        """Create a new track from detection."""
        track_id = self.next_id
        self.next_id += 1
        self.total_tracks_created += 1
        
        center = detection['center']
        
        self.tracks[track_id] = {
            'track_id': track_id,
            'center': center,
            'bbox': detection['bbox'],
            'confidence': detection['confidence'],
            'disappeared': 0,
            'created_frame': self.frame_count,
            'last_seen_frame': self.frame_count,
            'created_time': current_time,
            'last_update_time': current_time,
            
            # Position history for movement analysis
            'position_history': deque([center], maxlen=30),
            'confidence_history': deque([detection['confidence']], maxlen=10),
            
            # Movement metrics
            'total_movement': 0.0,
            'avg_movement_per_frame': 0.0,
            'velocity': [0.0, 0.0],  # pixels per frame [x, y]
            'movement_history': deque(maxlen=20),
            
            # Behavior flags
            'suspicious_movement_count': 0,
            'stationary_frames': 0,
            'direction_changes': 0,
            
            # Physical properties
            'area': detection['area'],
            'aspect_ratio': detection['aspect_ratio'],
            'avg_area': detection['area'],
            'area_history': deque([detection['area']], maxlen=10)
        }
        
        return track_id
    
    def _associate_detections(self, detections: List[Dict], current_time: float):
        """Associate detections with existing tracks using Hungarian algorithm approach."""
        # Calculate distance matrix between existing tracks and new detections
        track_ids = list(self.tracks.keys())
        
        if len(track_ids) == 0:
            # Create new tracks for all detections
            for detection in detections:
                self._create_new_track(detection, current_time)
            return
        
        # Simple greedy assignment (can be upgraded to Hungarian algorithm)
        used_detections = set()
        
        for track_id in track_ids:
            track_center = self.tracks[track_id]['center']
            best_detection_idx = None
            min_distance = float('inf')
            
            # Find closest detection to this track with improved similarity
            for i, detection in enumerate(detections):
                if i in used_detections:
                    continue
                
                detection_center = detection['center']
                
                # Calculate distance
                distance = np.sqrt((track_center[0] - detection_center[0])**2 + 
                                 (track_center[1] - detection_center[1])**2)
                
                # Calculate area similarity (helps with webcam inconsistencies)
                area_similarity = 1.0 - abs(track['area'] - detection['area']) / max(track['area'], detection['area'])
                
                # Calculate confidence similarity
                conf_similarity = 1.0 - abs(track['confidence'] - detection['confidence'])
                
                # Combined similarity score
                combined_score = distance - (area_similarity * self.area_weight * 100) - (conf_similarity * self.confidence_weight * 100)
                
                if distance < self.max_distance and combined_score < min_distance:
                    min_distance = combined_score
                    best_detection_idx = i
            
            if best_detection_idx is not None:
                # Update existing track
                self._update_track(track_id, detections[best_detection_idx], current_time)
                used_detections.add(best_detection_idx)
            else:
                # No detection associated - increment disappeared
                self.tracks[track_id]['disappeared'] += 1
        
        # Create new tracks for unassociated detections
        for i, detection in enumerate(detections):
            if i not in used_detections:
                self._create_new_track(detection, current_time)
    
    def _update_track(self, track_id: int, detection: Dict, current_time: float):
        """Update existing track with new detection."""
        track = self.tracks[track_id]
        old_center = track['center']
        new_center = detection['center']
        
        # Calculate movement
        movement = np.sqrt((new_center[0] - old_center[0])**2 + 
                          (new_center[1] - old_center[1])**2)
        
        # Calculate velocity (pixels per frame)
        velocity_x = new_center[0] - old_center[0]
        velocity_y = new_center[1] - old_center[1]
        
        # Update position history
        track['position_history'].append(new_center)
        track['confidence_history'].append(detection['confidence'])
        track['movement_history'].append(movement)
        track['area_history'].append(detection['area'])
        
        # Update basic properties
        track['center'] = new_center
        track['bbox'] = detection['bbox']
        track['confidence'] = detection['confidence']
        track['disappeared'] = 0
        track['last_seen_frame'] = self.frame_count
        track['last_update_time'] = current_time
        
        # Update movement metrics
        track['total_movement'] += movement
        track['velocity'] = [velocity_x, velocity_y]
        
        frames_tracked = self.frame_count - track['created_frame'] + 1
        track['avg_movement_per_frame'] = track['total_movement'] / frames_tracked
        
        # Update behavior analysis
        if movement < 5:  # Very small movement threshold
            track['stationary_frames'] += 1
        else:
            track['stationary_frames'] = 0
        
        # Detect direction changes (simplified)
        if len(track['movement_history']) >= 3:
            recent_movements = list(track['movement_history'])[-3:]
            if max(recent_movements) > 20:  # Significant movement
                track['suspicious_movement_count'] += 1
        
        # Update area metrics
        track['area'] = detection['area']
        track['aspect_ratio'] = detection['aspect_ratio']
        track['avg_area'] = np.mean(list(track['area_history']))
    
    def _cleanup_disappeared_tracks(self):
        """Remove tracks that have disappeared for too long."""
        tracks_to_remove = []
        
        for track_id, track in self.tracks.items():
            if track['disappeared'] > self.max_disappeared:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
    
    def get_track_by_id(self, track_id: int) -> Optional[Dict]:
        """Get specific track by ID."""
        return self.tracks.get(track_id)
    
    def get_active_tracks(self) -> Dict[int, Dict]:
        """Get all currently active tracks."""
        return {tid: track for tid, track in self.tracks.items() if track['disappeared'] == 0}
    
    def get_tracking_statistics(self) -> Dict:
        """Get overall tracking statistics."""
        active_tracks = self.get_active_tracks()
        
        if not active_tracks:
            return {
                'total_active_tracks': 0,
                'avg_track_age': 0,
                'total_tracks_created': self.total_tracks_created
            }
        
        track_ages = [self.frame_count - track['created_frame'] for track in active_tracks.values()]
        avg_movements = [track['avg_movement_per_frame'] for track in active_tracks.values()]
        
        return {
            'total_active_tracks': len(active_tracks),
            'total_tracks_created': self.total_tracks_created,
            'avg_track_age': np.mean(track_ages),
            'max_track_age': np.max(track_ages) if track_ages else 0,
            'avg_movement_per_frame': np.mean(avg_movements) if avg_movements else 0,
            'frame_count': self.frame_count
        }
    
    def draw_tracks(self, frame: np.ndarray, show_trails: bool = True, 
                   show_ids: bool = True) -> np.ndarray:
        """
        Draw tracking results on frame.
        
        Args:
            frame: Input frame
            show_trails: Whether to show movement trails
            show_ids: Whether to show track IDs
            
        Returns:
            Annotated frame with tracking visualization
        """
        annotated_frame = frame.copy()
        active_tracks = self.get_active_tracks()
        
        # Color palette for tracks
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (0, 128, 255), (255, 20, 147), (34, 139, 34), (255, 69, 0)
        ]
        
        for track_id, track in active_tracks.items():
            color = colors[track_id % len(colors)]
            center = track['center']
            bbox = track['bbox']
            
            # Draw bounding box
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw center point
            cv2.circle(annotated_frame, tuple(center), 6, color, -1)
            
            if show_ids:
                # Draw track ID
                cv2.putText(annotated_frame, f"ID:{track_id}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Draw movement info
                movement_info = f"Mov:{track['avg_movement_per_frame']:.1f}"
                cv2.putText(annotated_frame, movement_info, (x1, y2+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            if show_trails and len(track['position_history']) > 1:
                # Draw movement trail
                points = list(track['position_history'])
                for i in range(1, len(points)):
                    cv2.line(annotated_frame, tuple(points[i-1]), tuple(points[i]), 
                           color, 2)
        
        # Draw tracking statistics
        stats = self.get_tracking_statistics()
        stats_color = (255, 255, 255)
        cv2.putText(annotated_frame, f"Active Tracks: {stats['total_active_tracks']}", 
                   (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, stats_color, 2)
        cv2.putText(annotated_frame, f"Total Created: {stats['total_tracks_created']}", 
                   (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, stats_color, 2)
        
        return annotated_frame