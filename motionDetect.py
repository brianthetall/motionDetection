import sys
import cv2
import numpy as np
import subprocess
from datetime import datetime

def get_output_layers(net):
    layer_names = net.getLayerNames()
    out_layers_indices = net.getUnconnectedOutLayers()

    # Check if the output layer indices are wrapped in a numpy array
    if out_layers_indices.ndim > 1:
        out_layers_indices = out_layers_indices.flatten()

    return [layer_names[i - 1] for i in out_layers_indices]

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
output_layers = get_output_layers(net)

def detect_human_yolo(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    humanFound = False
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id == 0 and confidence > 0.5:  # Class ID 0 is 'person'
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                humanFound = True

    return humanFound, frame

def detect_humanoid_motion(frame, body_cascade, min_size=(50,200)):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=min_size)
    detected = False

    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Yellow rectangle
        detected = True

    return detected, frame


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
    rtsp_url = sys.argv[1] #"rtsp://admin:admin@172.172.172.30/11"
    cap = cv2.VideoCapture(rtsp_url)
    consistent_count = 0
    threshold_consistency = 1  # Number of consecutive frames to confirm detection

    frame_skip = 6  # Skip every N frames
    frame_count = 0

    if not cap.isOpened():
        print("Error: Unable to open video stream.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Unable to read video frame.")
            #continue
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  # Skip this frame
        
        #detected, processed_frame = detect_humanoid_motion(frame, body_cascade)
        detected, processed_frame = detect_human_yolo(frame)
        if detected:
            consistent_count += 1
            if consistent_count >= threshold_consistency:
                print("Consistent humanoid motion detected! Saving frame...")
                save_frame(processed_frame)  # Save the frame with the yellow rectangle
                consistent_count = 0
        else:
            consistent_count = 0

    cap.release()

if __name__ == "__main__":
    main()
    
