#!/usr/bin/env python3
"""
YouTube Video Downloader Script
Downloads YouTube videos and Shorts to a 'videos' folder
"""

import os
import sys
import subprocess

def install_requirements():
    """Install yt-dlp if not already installed"""
    try:
        import yt_dlp
    except ImportError:
        print("Installing yt-dlp...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp"])
        import yt_dlp

def download_youtube_video(url):
    """Download YouTube video to videos folder"""
    # Import after potential installation
    import yt_dlp
    
    # Create videos directory if it doesn't exist
    videos_dir = os.path.join(os.getcwd(), "videos")
    os.makedirs(videos_dir, exist_ok=True)
    
    # Configure yt-dlp options
    ydl_opts = {
        'outtmpl': os.path.join(videos_dir, '%(title)s.%(ext)s'),
        'format': 'best[height<=720]',  # Download best quality up to 720p
        'writeinfojson': False,  # Don't save metadata
        'writesubtitles': False,  # Don't download subtitles
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"Downloading: {url}")
            ydl.download([url])
            print(f"✅ Download completed! Video saved to: {videos_dir}")
            
    except Exception as e:
        print(f"❌ Error downloading video: {str(e)}")
        return False
    
    return True

def main():
    """Main function"""
    # Install requirements
    install_requirements()
    
    # Get URL from command line argument or user input
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = input("Enter YouTube URL (video or Short): ").strip()
    
    # Validate URL
    if not url or not ("youtube.com" in url or "youtu.be" in url):
        print("❌ Please provide a valid YouTube URL")
        return
    
    # Download the video
    success = download_youtube_video(url)
    
    if success:
        print("🎉 Download successful!")
    else:
        print("💥 Download failed!")

if __name__ == "__main__":
    main()