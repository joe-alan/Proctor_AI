import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

import cv2
import time
from detector import ExamDetector
from tracker import PersonTracker

def test_live_detection_tracking():
    """Test detection and tracking with live webcam feed."""
    
    print("🚀 Starting live detection + tracking test...")
    print("Press 'q' to quit, 's' to save screenshot")
    
    # Initialize components
    detector = ExamDetector(model_size='yolov8m.pt', confidence_threshold=0.6)
    tracker = PersonTracker(max_disappeared=30, max_distance=150)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return
    
    frame_count = 0
    fps_start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.time()
            
            # Step 1: Detect people
            people = detector.detect_people(frame)
            
            # Step 2: Update tracker
            tracks = tracker.update(people)
            
            # Step 3: Create visualization
            # First draw detections
            annotated_frame = detector.draw_detections(frame, people, show_details=True)
            
            # Then draw tracking info
            annotated_frame = tracker.draw_tracks(annotated_frame, show_trails=True, show_ids=True)
            
            # Calculate and display FPS
            processing_time = time.time() - start_time
            fps = 1.0 / processing_time if processing_time > 0 else 0
            
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Show processing info
            cv2.putText(annotated_frame, f"Process Time: {processing_time*1000:.1f}ms", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            # Display frame
            cv2.imshow('Exam Monitor - Detection + Tracking Test', annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_name = f"test_screenshot_{frame_count}.jpg"
                cv2.imwrite(screenshot_name, annotated_frame)
                print(f"📸 Screenshot saved: {screenshot_name}")
            
            frame_count += 1
            
            # Print stats every 30 frames
            if frame_count % 30 == 0:
                elapsed = time.time() - fps_start_time
                avg_fps = frame_count / elapsed
                stats = tracker.get_tracking_statistics()
                
                print(f"📊 Frame {frame_count}: Avg FPS: {avg_fps:.1f}, "
                      f"Active tracks: {stats['total_active_tracks']}, "
                      f"Total created: {stats['total_tracks_created']}")
    
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Final statistics
        total_time = time.time() - fps_start_time
        final_stats = tracker.get_tracking_statistics()
        
        print(f"\n📈 Test Complete!")
        print(f"Total frames processed: {frame_count}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average FPS: {frame_count/total_time:.1f}")
        print(f"Total tracks created: {final_stats['total_tracks_created']}")
        print(f"Final active tracks: {final_stats['total_active_tracks']}")

def test_with_video_file(video_path):
    """Test with a video file instead of live camera."""
    
    print(f"🎬 Testing with video file: {video_path}")
    
    # Initialize components
    detector = ExamDetector(model_size='yolov8m.pt', confidence_threshold=0.6)
    tracker = PersonTracker(max_disappeared=30, max_distance=150)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video properties: {total_frames} frames @ {fps} FPS")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            people = detector.detect_people(frame)
            tracks = tracker.update(people)
            
            # Create visualization
            annotated_frame = detector.draw_detections(frame, people)
            annotated_frame = tracker.draw_tracks(annotated_frame, show_trails=True)
            
            # Show progress
            progress = (frame_count / total_frames) * 100
            cv2.putText(annotated_frame, f"Progress: {progress:.1f}%", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Video Test - Detection + Tracking', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
            
            # Print progress every 100 frames
            if frame_count % 100 == 0:
                stats = tracker.get_tracking_statistics()
                print(f"📊 Progress: {progress:.1f}% - Active tracks: {stats['total_active_tracks']}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        final_stats = tracker.get_tracking_statistics()
        print(f"\n✅ Video processing complete!")
        print(f"Frames processed: {frame_count}/{total_frames}")
        print(f"Total tracks created: {final_stats['total_tracks_created']}")

if __name__ == "__main__":
    print("🧪 Exam Monitor - Detection + Tracking Test")
    print("="*50)
    
    # Test with live camera by default
    # To test with video file, uncomment the line below and provide video path
    # test_with_video_file("your_test_video.mp4")
    
    test_live_detection_tracking()
    