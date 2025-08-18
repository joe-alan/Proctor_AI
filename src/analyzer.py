import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from collections import deque, defaultdict
import time
import math

class BehaviorAnalyzer:
    """
    Advanced behavior analysis system for exam monitoring.
    Detects suspicious activities like excessive movement, head turning, clustering, etc.
    """
    
    def __init__(self, 
                 movement_threshold: float = 15.0,
                 clustering_threshold: float = 120.0,
                 attention_threshold: float = 0.4,
                 alert_cooldown: int = 30,
                 alert_persistence: int = 90):
        """
        Initialize behavior analyzer with detection thresholds.
        
        Args:
            movement_threshold: Pixels moved per frame to trigger movement alert
            clustering_threshold: Distance threshold for clustering detection
            attention_threshold: Aspect ratio change threshold for attention detection
            alert_cooldown: Minimum frames between same alert types
            alert_persistence: Frames to keep alerts visible on screen
        """
        self.movement_threshold = movement_threshold
        self.clustering_threshold = clustering_threshold  
        self.attention_threshold = attention_threshold
        self.alert_cooldown = alert_cooldown
        self.alert_persistence = alert_persistence
        
        # Alert tracking
        self.alerts_history = deque(maxlen=1000)
        self.recent_alerts = defaultdict(int)  # Track recent alerts per person
        self.alert_counters = defaultdict(int)
        self.persistent_alerts = deque(maxlen=10)  # Keep recent alerts visible
        
        # Behavior tracking
        self.person_behaviors = {}
        self.frame_count = 0
        
        # Analysis windows
        self.movement_window = 10  # frames
        self.attention_window = 15  # frames
        
        print(f"🧠 BehaviorAnalyzer initialized")
        print(f"   Movement threshold: {movement_threshold} pixels")
        print(f"   Clustering threshold: {clustering_threshold} pixels")
        print(f"   Attention threshold: {attention_threshold}")
        print(f"   Alert persistence: {alert_persistence} frames")
    
    def analyze_behavior(self, tracks: Dict[int, Dict], frame_num: int) -> List[Dict]:
        """
        Analyze behavior patterns and generate alerts.
        
        Args:
            tracks: Active tracks from PersonTracker
            frame_num: Current frame number
            
        Returns:
            List of alert dictionaries
        """
        self.frame_count = frame_num
        current_alerts = []
        
        if not tracks:
            return current_alerts
        
        # Update behavior tracking for each person
        for track_id, track in tracks.items():
            self._update_person_behavior(track_id, track)
        
        # Analyze individual behaviors
        for track_id, track in tracks.items():
            alerts = self._analyze_individual_behavior(track_id, track)
            current_alerts.extend(alerts)
        
        # Analyze group behaviors
        group_alerts = self._analyze_group_behavior(tracks)
        current_alerts.extend(group_alerts)
        
        # Store alerts in history and persistent display
        for alert in current_alerts:
            alert['frame'] = frame_num
            alert['timestamp'] = time.time()
            alert['expires_at'] = frame_num + self.alert_persistence
            self.alerts_history.append(alert)
            self.persistent_alerts.append(alert)
        
        # Clean up expired persistent alerts
        current_persistent = []
        for alert in self.persistent_alerts:
            if alert['expires_at'] > frame_num:
                current_persistent.append(alert)
        self.persistent_alerts = deque(current_persistent, maxlen=10)
        
        # Update recent alerts tracking
        self._update_alert_tracking(current_alerts)
        
        return current_alerts
    
    def _update_person_behavior(self, track_id: int, track: Dict):
        """Update behavior tracking data for a person."""
        if track_id not in self.person_behaviors:
            self.person_behaviors[track_id] = {
                'movement_history': deque(maxlen=self.movement_window),
                'attention_history': deque(maxlen=self.attention_window),
                'position_variance': 0.0,
                'attention_changes': 0,
                'total_alerts': 0,
                'last_alert_frame': -1
            }
        
        behavior = self.person_behaviors[track_id]
        
        # Update movement history
        if len(track['position_history']) >= 2:
            recent_positions = list(track['position_history'])[-2:]
            movement = np.sqrt((recent_positions[1][0] - recent_positions[0][0])**2 + 
                             (recent_positions[1][1] - recent_positions[0][1])**2)
            behavior['movement_history'].append(movement)
        
        # Update attention history (using aspect ratio as proxy)
        behavior['attention_history'].append(track['aspect_ratio'])
        
        # Calculate position variance (stability metric)
        if len(track['position_history']) >= 5:
            positions = np.array(list(track['position_history'])[-5:])
            behavior['position_variance'] = np.var(positions, axis=0).mean()
    
    def _analyze_individual_behavior(self, track_id: int, track: Dict) -> List[Dict]:
        """Analyze individual person's behavior for suspicious activities."""
        alerts = []
        behavior = self.person_behaviors[track_id]
        
        # Skip if too recent alert for this person
        if self.frame_count - behavior['last_alert_frame'] < self.alert_cooldown:
            return alerts
        
        # 1. Excessive Movement Analysis
        if len(behavior['movement_history']) >= self.movement_window:
            avg_movement = np.mean(list(behavior['movement_history']))
            max_movement = np.max(list(behavior['movement_history']))
            
            if avg_movement > self.movement_threshold or max_movement > self.movement_threshold * 2:
                severity = min(avg_movement / self.movement_threshold, 3.0)
                alerts.append({
                    'type': 'excessive_movement',
                    'track_id': track_id,
                    'severity': severity,
                    'description': f'Person {track_id} moving excessively (avg: {avg_movement:.1f}px)',
                    'position': track['center'],
                    'confidence': min(0.9, severity / 3.0)
                })
        
        # 2. Attention/Head Turning Analysis
        if len(behavior['attention_history']) >= self.attention_window:
            attention_values = list(behavior['attention_history'])
            attention_variance = np.var(attention_values)
            
            # Detect rapid aspect ratio changes (head turning)
            if attention_variance > self.attention_threshold:
                severity = min(attention_variance / self.attention_threshold, 3.0)
                alerts.append({
                    'type': 'attention_deviation',
                    'track_id': track_id,
                    'severity': severity,
                    'description': f'Person {track_id} frequent head turning (var: {attention_variance:.2f})',
                    'position': track['center'],
                    'confidence': min(0.8, severity / 3.0)
                })
        
        # 3. Unusual Posture Analysis
        current_aspect_ratio = track['aspect_ratio']
        if current_aspect_ratio > 1.5:  # Very wide bounding box
            alerts.append({
                'type': 'unusual_posture',
                'track_id': track_id,
                'severity': min(current_aspect_ratio / 1.5, 3.0),
                'description': f'Person {track_id} unusual posture (AR: {current_aspect_ratio:.2f})',
                'position': track['center'],
                'confidence': 0.6
            })
        
        # 4. Talking Detection (based on rapid head position changes)
        if len(behavior['attention_history']) >= self.attention_window:
            attention_values = list(behavior['attention_history'])
            # Look for rapid, small oscillations that might indicate talking
            recent_changes = [abs(attention_values[i] - attention_values[i-1]) 
                            for i in range(1, min(8, len(attention_values)))]
            
            if len(recent_changes) >= 5:
                talking_pattern = np.std(recent_changes) > 0.05 and np.mean(recent_changes) > 0.02
                if talking_pattern:
                    alerts.append({
                        'type': 'suspected_talking',
                        'track_id': track_id,
                        'severity': 2.5,
                        'description': f'Person {track_id} shows talking-like head movements',
                        'position': track['center'],
                        'confidence': 0.7
                    })
        
        # 5. Phone Usage Detection (hand-to-face gesture)
        current_aspect_ratio = track['aspect_ratio']
        if len(behavior['attention_history']) >= 3:
            recent_ratios = list(behavior['attention_history'])[-3:]
            # Detect sudden narrowing of bounding box (phone to ear)
            if all(r < 0.6 for r in recent_ratios) and current_aspect_ratio < 0.5:
                alerts.append({
                    'type': 'phone_usage',
                    'track_id': track_id,
                    'severity': 3.0,
                    'description': f'Person {track_id} possible phone usage detected',
                    'position': track['center'],
                    'confidence': 0.8
                })
        
        # 6. Paper Sharing Movement (reaching toward others)
        if len(behavior['movement_history']) >= 5:
            recent_movements = list(behavior['movement_history'])[-5:]
            # Detect pattern of movement then return (paper sharing)
            if max(recent_movements) > 25 and recent_movements[-1] < 8:
                alerts.append({
                    'type': 'paper_sharing',
                    'track_id': track_id,
                    'severity': 2.8,
                    'description': f'Person {track_id} suspicious reaching movement',
                    'position': track['center'],
                    'confidence': 0.6
                })
        
        # 7. Extended Looking Around (exam anxiety or cheating)
        if len(track['position_history']) >= 10:
            positions = np.array(list(track['position_history'])[-10:])
            position_spread = np.std(positions, axis=0).mean()
            
            if position_spread > 15:  # High variability in head position
                alerts.append({
                    'type': 'excessive_looking_around',
                    'track_id': track_id,
                    'severity': 2.0,
                    'description': f'Person {track_id} looking around extensively',
                    'position': track['center'],
                    'confidence': 0.5
                })
        
        # 8. Cheating Gestures (specific movement patterns)
        if track['suspicious_movement_count'] > 3:
            alerts.append({
                'type': 'cheating_gestures',
                'track_id': track_id,
                'severity': 2.7,
                'description': f'Person {track_id} repeated suspicious gestures',
                'position': track['center'],
                'confidence': 0.7
            })
        
        # 9. Stationary Too Long (Paradoxically suspicious in some contexts)
        if track['stationary_frames'] > 300:  # 10 seconds at 30fps
            alerts.append({
                'type': 'too_stationary',
                'track_id': track_id,
                'severity': 1.0,
                'description': f'Person {track_id} completely motionless for extended time',
                'position': track['center'],
                'confidence': 0.4
            })
        
        # Update alert tracking
        if alerts:
            behavior['last_alert_frame'] = self.frame_count
            behavior['total_alerts'] += len(alerts)
        
        return alerts
    
    def _analyze_group_behavior(self, tracks: Dict[int, Dict]) -> List[Dict]:
        """Analyze group behaviors like clustering, coordination, etc."""
        alerts = []
        
        if len(tracks) < 2:
            return alerts
        
        track_list = list(tracks.values())
        
        # 1. Clustering Analysis - People too close together
        for i, track1 in enumerate(track_list):
            for j, track2 in enumerate(track_list[i+1:], i+1):
                center1 = np.array(track1['center'])
                center2 = np.array(track2['center'])
                distance = np.linalg.norm(center1 - center2)
                
                if distance < self.clustering_threshold:
                    severity = max(1.0, (self.clustering_threshold - distance) / 50.0)
                    
                    alerts.append({
                        'type': 'clustering',
                        'track_id': [track1['track_id'], track2['track_id']],
                        'severity': min(severity, 3.0),
                        'description': f'People {track1["track_id"]} and {track2["track_id"]} too close ({distance:.0f}px)',
                        'position': ((center1 + center2) / 2).astype(int).tolist(),
                        'confidence': min(0.7, severity / 3.0)
                    })
        
        # 2. Synchronized Movement Analysis
        if len(tracks) >= 2:
            movements = []
            for track in track_list:
                if len(track['movement_history']) > 0:
                    recent_movement = list(track['movement_history'])[-1] if track['movement_history'] else 0
                    movements.append(recent_movement)
            
            if len(movements) >= 2:
                try:
                    # Ensure we have valid arrays for correlation
                    movements_array = np.array(movements[:2])
                    if len(movements_array) == 2 and np.var(movements_array) > 0:
                        correlation_matrix = np.corrcoef(movements_array)
                        if correlation_matrix.shape == (2, 2):
                            movement_sync = correlation_matrix[0, 1]
                        else:
                            movement_sync = 0
                    else:
                        movement_sync = 0
                except:
                    movement_sync = 0
                
                # High correlation in movement might indicate coordination
                if not np.isnan(movement_sync) and movement_sync > 0.8 and np.mean(movements) > 5:
                    alerts.append({
                        'type': 'synchronized_movement',
                        'track_id': [track['track_id'] for track in track_list[:2]],
                        'severity': min(movement_sync * 2, 3.0),
                        'description': f'Synchronized suspicious movement detected',
                        'position': [0, 0],  # Will be updated in drawing
                        'confidence': 0.6
                    })
        
        return alerts
    
    def _update_alert_tracking(self, alerts: List[Dict]):
        """Update recent alert tracking to prevent spam."""
        for alert in alerts:
            alert_key = f"{alert['type']}_{alert['track_id']}"
            self.recent_alerts[alert_key] = self.frame_count
    
    def get_alert_summary(self) -> Dict:
        """Get summary of all alerts generated."""
        if not self.alerts_history:
            return {'total_alerts': 0}
        
        alert_types = defaultdict(int)
        severity_levels = []
        
        for alert in self.alerts_history:
            alert_types[alert['type']] += 1
            severity_levels.append(alert['severity'])
        
        return {
            'total_alerts': len(self.alerts_history),
            'alert_types': dict(alert_types),
            'avg_severity': np.mean(severity_levels) if severity_levels else 0,
            'max_severity': np.max(severity_levels) if severity_levels else 0,
            'recent_alerts': len([a for a in self.alerts_history if time.time() - a['timestamp'] < 60])
        }
    
    def get_person_risk_score(self, track_id: int) -> float:
        """Calculate risk score for a specific person."""
        if track_id not in self.person_behaviors:
            return 0.0
        
        behavior = self.person_behaviors[track_id]
        
        # Calculate risk based on multiple factors
        alert_frequency = behavior['total_alerts'] / max(1, self.frame_count / 100)
        movement_risk = behavior['position_variance'] / 1000.0
        
        person_alerts = [a for a in self.alerts_history if a.get('track_id') == track_id]
        recent_severity = np.mean([a['severity'] for a in person_alerts[-5:]]) if person_alerts else 0
        
        risk_score = min(1.0, (alert_frequency * 0.4 + movement_risk * 0.3 + recent_severity * 0.3))
        
        return risk_score
    
    def draw_alerts(self, frame: np.ndarray, current_alerts: List[Dict], 
                   show_risk_scores: bool = True) -> np.ndarray:
        """
        Draw alert visualizations on frame including persistent alerts.
        
        Args:
            frame: Input frame
            current_alerts: Current frame alerts (for immediate feedback)
            show_risk_scores: Whether to show individual risk scores
            
        Returns:
            Annotated frame with alert visualizations
        """
        annotated_frame = frame.copy()
        
        # Use persistent alerts for display (includes recent alerts)
        display_alerts = list(self.persistent_alerts)
        
        # Define alert colors with more exam-specific types
        alert_colors = {
            'excessive_movement': (0, 0, 255),        # Red
            'attention_deviation': (0, 165, 255),     # Orange  
            'clustering': (255, 0, 255),              # Magenta
            'unusual_posture': (0, 255, 255),         # Yellow
            'synchronized_movement': (128, 0, 128),   # Purple
            'suspected_talking': (0, 100, 255),       # Dark Orange
            'phone_usage': (0, 0, 200),               # Dark Red
            'paper_sharing': (255, 100, 100),         # Light Red
            'excessive_looking_around': (100, 255, 100), # Light Green
            'cheating_gestures': (200, 0, 200),       # Dark Magenta
            'too_stationary': (128, 128, 128)         # Gray
        }
        
        # Draw individual alerts
        alert_y_offset = 150
        for alert in display_alerts:
            color = alert_colors.get(alert['type'], (255, 255, 255))
            position = alert['position']
            
            # Draw alert marker at position
            if isinstance(position, list) and len(position) == 2:
                cv2.circle(annotated_frame, tuple(position), 15, color, 3)
                
                # Draw severity indicator
                severity_radius = int(alert['severity'] * 10)
                cv2.circle(annotated_frame, tuple(position), severity_radius, color, 1)
                
                # Add alert type text near the marker
                alert_text = alert['type'].replace('_', ' ').upper()
                cv2.putText(annotated_frame, alert_text, 
                           (position[0] + 20, position[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw alert summary panel
        panel_height = 140
        cv2.rectangle(annotated_frame, (10, alert_y_offset), 
                     (450, alert_y_offset + panel_height), (0, 0, 0), -1)
        cv2.rectangle(annotated_frame, (10, alert_y_offset), 
                     (450, alert_y_offset + panel_height), (255, 255, 255), 2)
        
        # Alert summary text
        summary = self.get_alert_summary()
        text_color = (255, 255, 255)
        
        cv2.putText(annotated_frame, "EXAM MONITORING ALERTS", (15, alert_y_offset + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        cv2.putText(annotated_frame, f"Total Alerts: {summary['total_alerts']}", 
                   (15, alert_y_offset + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        
        cv2.putText(annotated_frame, f"Recent (1min): {summary.get('recent_alerts', 0)}", 
                   (15, alert_y_offset + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        
        cv2.putText(annotated_frame, f"Persistent Alerts: {len(display_alerts)}", 
                   (15, alert_y_offset + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        
        cv2.putText(annotated_frame, f"Avg Severity: {summary.get('avg_severity', 0):.2f}", 
                   (15, alert_y_offset + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        
        # Current frame alerts (immediate feedback)
        if current_alerts:
            cv2.putText(annotated_frame, f"NEW ALERTS: {len(current_alerts)}", 
                       (15, alert_y_offset + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Show most recent alert description
            latest_alert = current_alerts[-1]
            alert_desc = latest_alert['type'].replace('_', ' ').title()
            cv2.putText(annotated_frame, f"Latest: {alert_desc}", 
                       (15, alert_y_offset + 125), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Draw enhanced legend
        legend_x = frame.shape[1] - 280
        legend_y = 30
        cv2.putText(annotated_frame, "EXAM ALERT TYPES:", (legend_x, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show top 6 most relevant alert types for exams
        priority_alerts = [
            ('excessive_movement', 'Excessive Movement'),
            ('suspected_talking', 'Suspected Talking'),
            ('attention_deviation', 'Looking Around'),
            ('phone_usage', 'Phone Usage'),
            ('paper_sharing', 'Paper Sharing'),
            ('clustering', 'Student Clustering')
        ]
        
        y_offset = 50
        for alert_type, display_name in priority_alerts:
            if alert_type in alert_colors:
                color = alert_colors[alert_type]
                cv2.circle(annotated_frame, (legend_x + 10, legend_y + y_offset), 5, color, -1)
                cv2.putText(annotated_frame, display_name, 
                           (legend_x + 25, legend_y + y_offset + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                y_offset += 18
        
        return annotated_frame
    
    def export_alert_report(self, filename: str = None) -> str:
        """Export detailed alert report to file."""
        if not filename:
            filename = f"exam_alert_report_{int(time.time())}.txt"
        
        with open(filename, 'w') as f:
            f.write("EXAM MONITORING - BEHAVIOR ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            summary = self.get_alert_summary()
            f.write(f"Total Alerts Generated: {summary['total_alerts']}\n")
            f.write(f"Average Severity: {summary.get('avg_severity', 0):.2f}\n")
            f.write(f"Maximum Severity: {summary.get('max_severity', 0):.2f}\n\n")
            
            f.write("Alert Breakdown by Type:\n")
            for alert_type, count in summary.get('alert_types', {}).items():
                f.write(f"  {alert_type}: {count}\n")
            
            f.write(f"\nDetailed Alert History:\n")
            f.write("-" * 30 + "\n")
            
            for alert in self.alerts_history:
                f.write(f"Frame {alert['frame']}: {alert['description']} "
                       f"(Severity: {alert['severity']:.2f})\n")
        
        return filename