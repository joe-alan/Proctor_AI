#!/usr/bin/env python3
"""
Complete exam monitoring system test.
Tests detection + tracking + behavior analysis together.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

import cv2
import time
from detector import ExamDetector
from tracker import PersonTracker
from analyzer import BehaviorAnalyzer

def test_complete_exam_monitoring():

    print("  - Press 'q' to quit, 's' to save screenshot, 'r' for report")
    print("=" * 60)
    
    # Initialize all components
    print("🤖 Initializing AI components...")
    detector = ExamDetector(model_size='yolov8m.pt', confidence_threshold=0.6)
    tracker = PersonTracker(max_disappeared=50, max_distance=200)
    analyzer = BehaviorAnalyzer(
        movement_threshold=12.0,    # Sensitive to movement
        clustering_threshold=150.0,  # Alert when people < 150px apart
        attention_threshold=0.3,     # Sensitive to head turning
        alert_cooldown=20,          # Allow alerts every 20 frames
        alert_persistence=90        # Keep alerts visible for 90 frames (~3 seconds)
    )
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return
    
    print("✅ System initialized successfully!")
    print("🎬 Starting live monitoring...")
    print("🔥 New Features: Persistent alerts, talking detection, phone usage detection!")
    
    frame_count = 0
    start_time = time.time()
    total_alerts = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to capture frame")
                break
            
            frame_start = time.time()
            
            # Step 1: Detect people
            people = detector.detect_people(frame)
            
            # Step 2: Update tracking
            tracks = tracker.update(people)
            
            # Step 3: Analyze behavior and generate alerts
            alerts = analyzer.analyze_behavior(tracks, frame_count)
            total_alerts += len(alerts)
            
            # Step 4: Create comprehensive visualization
            # Start with detection visualization
            annotated_frame = detector.draw_detections(frame, people, show_details=False)
            
            # Add tracking visualization
            annotated_frame = tracker.draw_tracks(annotated_frame, show_trails=True, show_ids=True)
            
            # Add behavior analysis visualization with persistent alerts
            annotated_frame = analyzer.draw_alerts(annotated_frame, analyzer.persistent_alerts)
            
            # Add performance metrics
            processing_time = time.time() - frame_start
            fps = 1.0 / processing_time if processing_time > 0 else 0
            
            # Performance overlay
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.putText(annotated_frame, f"Processing: {processing_time*1000:.1f}ms", 
                       (10, frame.shape[0] - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Alert notifications with enhanced feedback
            if alerts:
                alert_text = f"🚨 {len(alerts)} NEW ALERTS!"
                cv2.putText(annotated_frame, alert_text, (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Print alerts to console with more detail
                for alert in alerts:
                    alert_type = alert['type'].replace('_', ' ').title()
                    print(f"🚨 ALERT: {alert_type} - {alert['description']} (Severity: {alert['severity']:.2f})")
            
            # Show number of persistent alerts
            num_persistent = len(analyzer.persistent_alerts)
            if num_persistent > 0:
                cv2.putText(annotated_frame, f"📌 {num_persistent} Persistent Alerts", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Display the complete monitoring interface
            cv2.imshow('Complete Exam Monitoring System - Enhanced', annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_name = f"exam_monitor_screenshot_{frame_count}.jpg"
                cv2.imwrite(screenshot_name, annotated_frame)
                print(f"📸 Screenshot saved: {screenshot_name}")
            elif key == ord('r'):
                report_file = analyzer.export_alert_report()
                print(f"📄 Alert report saved: {report_file}")
            elif key == ord('h'):
                print("\n📚 HELP - Alert Types:")
                print("   🔴 Excessive Movement - Moving around too much")
                print("   🟠 Attention Deviation - Looking around frequently")
                print("   🟣 Suspected Talking - Head movements suggesting speech")
                print("   🔴 Phone Usage - Hand-to-ear gestures")
                print("   🟡 Paper Sharing - Reaching toward others")
                print("   🟢 Looking Around - Environmental scanning")
                print("   Press 'h' anytime to see this help!")
            
            frame_count += 1
            
            # Enhanced status update every 60 frames
            if frame_count % 60 == 0:
                elapsed = time.time() - start_time
                avg_fps = frame_count / elapsed
                
                tracking_stats = tracker.get_tracking_statistics()
                alert_summary = analyzer.get_alert_summary()
                
                print(f"\n📊 Status Update (Frame {frame_count}):")
                print(f"   ⚡ Average FPS: {avg_fps:.1f}")
                print(f"   👥 Active Tracks: {tracking_stats['total_active_tracks']}")
                print(f"   🚨 Total Alerts: {alert_summary['total_alerts']}")
                print(f"   📌 Persistent Alerts: {len(analyzer.persistent_alerts)}")
                print(f"   🔥 Recent Alerts: {alert_summary.get('recent_alerts', 0)}")
                
                # Show alert type breakdown
                alert_types = alert_summary.get('alert_types', {})
                if alert_types:
                    print(f"   📈 Alert Types: {dict(list(alert_types.items())[:3])}")
                
                # Individual risk scores
                active_tracks = tracker.get_active_tracks()
                for track_id in active_tracks:
                    risk_score = analyzer.get_person_risk_score(track_id)
                    if risk_score > 0.1:  # Only show significant risks
                        print(f"   ⚠️  Person {track_id} Risk Score: {risk_score:.2f}")
    
    except KeyboardInterrupt:
        print("\n⏹️ Monitoring stopped by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Final comprehensive report
        total_time = time.time() - start_time
        final_tracking_stats = tracker.get_tracking_statistics()
        final_alert_summary = analyzer.get_alert_summary()
        
        print(f"\n" + "=" * 60)
        print("📈 FINAL EXAM MONITORING REPORT")
        print("=" * 60)
        
        print(f"🎬 Session Summary:")
        print(f"   Total Duration: {total_time:.1f} seconds")
        print(f"   Frames Processed: {frame_count}")
        print(f"   Average FPS: {frame_count/total_time:.1f}")
        
        print(f"\n👥 Tracking Performance:")
        print(f"   People Tracked: {final_tracking_stats['total_tracks_created']}")
        print(f"   Final Active: {final_tracking_stats['total_active_tracks']}")
        print(f"   Average Track Age: {final_tracking_stats.get('avg_track_age', 0):.1f} frames")
        
        print(f"\n🧠 Enhanced Behavior Analysis:")
        print(f"   Total Alerts: {final_alert_summary['total_alerts']}")
        print(f"   Alert Types: {final_alert_summary.get('alert_types', {})}")
        print(f"   Average Severity: {final_alert_summary.get('avg_severity', 0):.2f}")
        print(f"   Maximum Severity: {final_alert_summary.get('max_severity', 0):.2f}")
        print(f"   Final Persistent Alerts: {len(analyzer.persistent_alerts)}")
        
        # Show most common alert types
        alert_types = final_alert_summary.get('alert_types', {})
        if alert_types:
            print(f"\n🏆 Most Common Alert Types:")
            sorted_alerts = sorted(alert_types.items(), key=lambda x: x[1], reverse=True)
            for i, (alert_type, count) in enumerate(sorted_alerts[:5], 1):
                alert_name = alert_type.replace('_', ' ').title()
                print(f"   {i}. {alert_name}: {count} times")
        
        # Export final report
        final_report = analyzer.export_alert_report("final_exam_report.txt")
        print(f"\n📄 Detailed report saved: {final_report}")

if __name__ == "__main__":
    test_complete_exam_monitoring()