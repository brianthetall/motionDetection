import cv2
import numpy as np
import subprocess
from datetime import datetime

def detect_motion(prev_frame, curr_frame, threshold=50):

    # Assuming prev_frame and curr_frame are already in grayscale
    diff = cv2.absdiff(prev_frame, curr_frame)
    blur = cv2.GaussianBlur(diff, (5, 5), 0)
    _, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours) > 0

def record_video(rtsp_url, duration=30):
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}.mkv"

    # Command to use FFmpeg to record video from RTSP stream
    command = ['ffmpeg', '-i', rtsp_url, '-t', str(duration), '-acodec', 'copy', '-vcodec', 'copy', '-r', '1', filename]
    subprocess.run(command)

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
            record_video(rtsp_url, 30)  # Record for 30 seconds
            break

        prev_frame = gray
        #cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
