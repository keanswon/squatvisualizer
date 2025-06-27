from pose_detector import PoseDetector
from smoothing import LandmarkSmoother
from squat_analyzer import SquatAnalyzer
from visualizer import SquatVisualizer
from video_processor import VideoProcessor
import os

class SquatAnalysisSystem:
    """Main system that orchestrates all components"""
    
    def __init__(self, depth_threshold=90, shallow_threshold=120):
        self.pose_detector = PoseDetector()
        self.smoother = LandmarkSmoother()
        self.analyzer = SquatAnalyzer(depth_threshold, shallow_threshold)
        self.visualizer = SquatVisualizer()
        
        # Configure smoother with thresholds manually
        self.smoother.DEPTH_THRESHOLD = depth_threshold
        self.smoother.SHALLOW_THRESHOLD = shallow_threshold
        
        self.video_processor = VideoProcessor(
            self.pose_detector, 
            self.smoother, 
            self.analyzer, 
            self.visualizer
        )
    
    def analyze_video(self, video_path, output_path=None):
        """Analyze squat form in video"""
        self.video_processor.process_video(video_path, output_path)

def main():
    print("Improved MediaPipe Squat Analyzer")
    print("This version focuses on knee angles for accurate depth assessment")
    
    video_path = input("Enter video file name: ")
    
    # Check if it's a full path or just filename
    if not os.path.isabs(video_path):
        video_path = os.path.join(os.getcwd(), "videos", video_path)
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return
    
    save_output = input("Save output video? (y/n): ").lower() == 'y'
    output_path = None
    
    if save_output:
        output_path = input("Enter output file path (e.g., improved_squat.mp4): ")
        if not os.path.isabs(output_path):
            output_path = os.path.join(os.getcwd(), output_path)

    # Create system with default thresholds
    system = SquatAnalysisSystem(depth_threshold=90, shallow_threshold=120)
    
    print(f"Depth thresholds: <{system.analyzer.DEPTH_THRESHOLD}° = depth reached, >{system.analyzer.SHALLOW_THRESHOLD}° = not deep enough")
    
    system.analyze_video(video_path, output_path)

if __name__ == "__main__":
    main()