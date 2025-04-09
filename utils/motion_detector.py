import cv2
import imutils
import numpy as np
from datetime import datetime
import os

class MotionDetector:
    def __init__(self, source=0, output_dir="videos"):
        self.source = source
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run(self):
        cap = cv2.VideoCapture(self.source)
        first_frame = None
        writer = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = imutils.resize(frame, width=500)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if first_frame is None:
                first_frame = gray
                continue

            frame_delta = cv2.absdiff(first_frame, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            motion_detected = False
            for contour in contours:
                if cv2.contourArea(contour) < 500:
                    continue
                motion_detected = True
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Timestamp annotation
            timestamp = datetime.now().strftime("%A %d %B %Y %I:%M:%S%p")
            cv2.putText(frame, timestamp, (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if motion_detected:
                if writer is None:
                    filename = os.path.join(self.output_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi")
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    writer = cv2.VideoWriter(filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
                    print(f"[INFO] Motion detected. Recording started: {filename}")
                writer.write(frame)
            elif writer is not None:
                print("[INFO] Motion stopped. Recording saved.")
                writer.release()
                writer = None

            cv2.imshow("Smart Security Camera", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
