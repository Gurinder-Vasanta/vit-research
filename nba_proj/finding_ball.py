import os
import cv2
import subprocess 
# import skvideo.io

# purpose of this one is to teach the model when the ball is somewhere near the rim
def download_video(url, output_path):
    command = ["yt-dlp", "-f", "616", "-o", output_path, url]
    subprocess.run(command, check=True)
    print(f"Video downloaded to {output_path}")

vid_url = 'https://www.youtube.com/watch?v=SdvAG7dgrT0'
output_path = 'hoop_ball_finding.mp4'

download_video(vid_url,output_path)