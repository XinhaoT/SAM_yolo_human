import cv2
import argparse
import os

parser = argparse.ArgumentParser(description='Extract images from a given .mp4 video')
parser.add_argument('--filepath', type=str, required=True, help='the video path')
parser.add_argument('--frames_interval', type=int, default=3, help='interval of frames for the neighbouring extracted images')
args = parser.parse_args()

count = 0

cap = cv2.VideoCapture(args.filepath)
os.makedirs(os.path.join("extracted_frames", args.filepath.split("/")[-1][:-4]))

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        count += 1
        if count % args.frames_interval == 0:
            file_name = os.path.join("extracted_frames", args.filepath.split("/")[-1][:-4], f'{count:06d}.jpg')
            cv2.imwrite(file_name, frame)
    else:
        break

cap.release()

