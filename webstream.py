from flask import Flask, render_template, Response
import cv2
import imutils
from datetime import datetime
import os
import time

app = Flask(__name__)

# Initialize camera
camera = cv2.VideoCapture(0)

# Load MobileNet SSD model
net = cv2.dnn.readNetFromCaffe("models/MobileNetSSD_deploy.prototxt",
                               "models/MobileNetSSD_deploy.caffemodel")

# Class labels
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Ensure static/videos folder exists
if not os.path.exists("static/videos"):
    os.makedirs("static/videos")

def generate_frames():
    writer = None
    recording = False
    last_detection_time = 0
    output_path = None
    fps = 20

    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = imutils.resize(frame, width=500)
        (h, w) = frame.shape[:2]

        # Prepare frame for MobileNet SSD
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843,
                                     (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        person_detected = False

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(detections[0, 0, i, 1])
                label = CLASSES[idx]

                if label != "person":
                    continue

                person_detected = True
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (startX, startY, endX, endY) = box.astype("int")

                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                text = f"{label}: {int(confidence * 100)}%"
                cv2.putText(frame, text, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Handle recording logic
        current_time = time.time()
        if person_detected:
            last_detection_time = current_time
            if not recording:
                output_path = f"static/videos/person_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                recording = True
                print(f"[INFO] Started recording: {output_path}")

        elif recording and (current_time - last_detection_time > 2):
            recording = False
            writer.release()
            print(f"[INFO] Stopped recording: {output_path}")

        if recording and writer is not None:
            writer.write(frame)

        # Add timestamp
        timestamp = datetime.now().strftime("%A %d %B %Y %I:%M:%S%p")
        cv2.putText(frame, timestamp, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Encode and yield the frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recordings')
def recordings():
    video_folder = 'static/videos'
    videos = [f for f in os.listdir(video_folder) if f.endswith('.avi')]
    videos.sort(reverse=True)  # Show latest videos first
    return render_template('recordings.html', videos=videos)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
