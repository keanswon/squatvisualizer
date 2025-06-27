from ImprovedSquatAnalyzer import ImprovedSquatAnalyzer
import os

def main():
    analyzer = ImprovedSquatAnalyzer()
    
    print("Improved MediaPipe Squat Analyzer")
    print("This version focuses on knee angles for accurate depth assessment")
    print(f"Depth thresholds: <{analyzer.DEPTH_THRESHOLD}° = depth reached, >{analyzer.SHALLOW_THRESHOLD}° = not deep enough")
    
    video_path = input("Enter video file name: ")
    video_path = os.path.join(os.getcwd(), "videos", video_path)
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return
    
    save_output = input("Save output video? (y/n): ").lower() == 'y'
    
    if save_output:
        output_path = input("Enter output file path (e.g., improved_squat.mp4): ")
        analyzer.process_video(video_path, output_path)
    else:
        analyzer.process_video(video_path)


if __name__ == "__main__":
    main()