import os
import cv2
import subprocess 
# import skvideo.io

# purpose of this one is to teach the model when the ball is somewhere near the rim
def download_video(url, output_path):
    command = ["yt-dlp", "-f", "136", "-o", output_path, url]
    subprocess.run(command, check=True)
    print(f"Video downloaded to {output_path}")

vid_url = 'https://www.youtube.com/watch?v=H56iR3188P0'

# okc game: https://www.youtube.com/watch?v=I33o9UnUe1A&list=PLlVlyGVtvuVniS7jx4DyESaKRUANIzyhE (vid2)
# heat game: https://www.youtube.com/watch?v=H56iR3188P0 (vid3; removed playlist stuff)
output_path = 'data/raw_videos/heat_2014.mp4'

download_video(vid_url,output_path)