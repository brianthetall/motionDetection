import cv2
import numpy as np
import subprocess
from datetime import datetime

def save_frame(frame):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"frame_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    print(f"Frame saved as {filename}")

def detect_motion(prev_frame, curr_frame, threshold=200, min_contour_area=200):
    # Calculate the absolute difference between current frame and the previous frame
    diff = cv2.absdiff(prev_frame, curr_frame)
    blur = cv2.GaussianBlur(diff, (5, 5), 0)
    _, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any substantial contours are found
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            return True
    return False

def main():
    # Replace with your RTSP stream URL
    rtsp_url = "rtsp://admin:admin@172.172.172.30/1"
    
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_GSTREAMER)  # Using GStreamer backend
    if not cap.isOpened():
        print("Error: Unable to open video stream.")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read video frame.")
        return

    prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Unable to read video frame.")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if detect_motion(prev_frame, gray):
            print("Motion detected! Recording...")
            save_frame(frame)

        prev_frame = gray

    cap.release()

if __name__ == "__main__":
    main()
    
