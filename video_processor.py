import cv2

class VideoProcessor:
    """Handles video processing operations"""
    
    def __init__(self, pose_detector, smoother, analyzer, visualizer):
        self.pose_detector = pose_detector
        self.smoother = smoother
        self.analyzer = analyzer
        self.visualizer = visualizer
        self.frame_count = 0
        self.detection_count = 0
    
    def process_video(self, video_path, output_path=None):
        """Process video with squat analysis"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        self.frame_count = 0
        self.detection_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            
            # Detect pose
            landmarks = self.pose_detector.detect_pose(frame)
            
            if landmarks:
                self.detection_count += 1
                
                # Smooth landmarks
                smoothed_landmarks = self.smoother.smooth_landmarks(landmarks)
                
                # Analyze form
                analysis = self.analyzer.analyze_squat_form(smoothed_landmarks)
                
                # Apply smoothing to analysis results
                if 'knee_angle' in analysis:
                    analysis['knee_angle'] = self.smoother.smooth_knee_angle(analysis['knee_angle'])
                    analysis['depth_status'] = self.smoother.smooth_depth_status(analysis['knee_angle'])
                
                # Visualize
                frame = self.visualizer.draw_clean_stick_figure(frame, smoothed_landmarks)
                frame = self.visualizer.draw_form_feedback(frame, analysis)
            else:
                cv2.putText(frame, 'No pose detected', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            # Add frame counter
            cv2.putText(frame, f'Frame: {self.frame_count} | Detections: {self.detection_count}', 
                       (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('Squat Analysis', frame)
            
            if output_path:
                out.write(frame)
            
            # Progress indicator
            if self.frame_count % 30 == 0:
                progress = (self.frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"Progress: {progress:.1f}% - Detections: {self.detection_count}/{self.frame_count}")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"\nProcessing complete!")
        print(f"Total frames: {self.frame_count}")
        print(f"Successful detections: {self.detection_count}")
        print(f"Detection rate: {(self.detection_count/self.frame_count)*100:.1f}%")
